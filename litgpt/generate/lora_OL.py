# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import json
import sys
import time
from pathlib import Path
from typing import Optional

import jsonlines as jsonl
import lightning as L
import torch
import yaml
from lightning.fabric.plugins import BitsandbytesPrecision

from litgpt import PromptStyle, Tokenizer
from litgpt.data.base import InferenceDataset
from litgpt.generate.base import generate
from litgpt.lora import GPT, Config
from litgpt.prompts import has_prompt_style, load_prompt_style
from litgpt.utils import CLI, check_valid_checkpoint_dir, get_default_supported_precision, load_checkpoint


def main(
    input_file: Path,
    lora_dir: Path = Path("logs/runs/lora/e3c"),
    max_new_tokens: int = 100,
    top_k: Optional[int] = 50,
    top_p: float = 1.0,
    temperature: float = 0.8,
) -> None:
    """Generates a response based on a given instruction and an optional input. This script will only work with
    checkpoints from the instruction-tuned GPT model. See ``litgpt.finetune.full``.

    Args:
        lora_dir: Path to the checkpoint with trained weights, which are the output of
            ``litgpt.finetune.lora_OL``.
        max_new_tokens: The number of generation steps to take.
        top_k: The number of top most probable tokens to consider in the sampling process.
        top_p: If specified, it represents the cumulative probability threshold to consider in the sampling process.
            In top-p sampling, the next token is sampled from the highest probability tokens
            whose cumulative probability exceeds the threshold `top_p`. When specified,
            it must be `0 <= top_p <= 1`. Here, `top_p=0` is equivalent
            to sampling the most probable token, while `top_p=1` samples from the whole distribution.
            It can be used in conjunction with `top_k` and `temperature` with the following order
            of application:

            1. `top_k` sampling
            2. `temperature` scaling
            3. `top_p` sampling

            For more details, see https://arxiv.org/abs/1904.09751
            or https://huyenchip.com/2024/01/16/sampling.html#top_p
        temperature: A value controlling the randomness of the sampling process. Higher values result in more random
            samples.
    """
    with open(lora_dir / "hyperparameters.yaml", "r") as f:
        training_cfg = yaml.safe_load(f)

    precision = training_cfg["precision"] or get_default_supported_precision(training=False)

    plugins = None
    if training_cfg["quantize"] is not None and training_cfg["quantize"].startswith("bnb."):
        if "mixed" in precision:
            raise ValueError("Quantization and mixed precision is not supported.")
        dtype = {"16-true": torch.float16, "bf16-true": torch.bfloat16, "32-true": torch.float32}[precision]
        plugins = BitsandbytesPrecision(training_cfg["quantize"][4:], dtype)
        precision = None

    fabric = L.Fabric(devices=1, precision=precision, plugins=plugins)
    fabric.launch()

    checkpoint_dir = Path(training_cfg["checkpoint_dir"])
    check_valid_checkpoint_dir(checkpoint_dir)
    config = Config.from_file(
        checkpoint_dir / "model_config.yaml",
        lora_r=training_cfg["lora"]["r"],
        lora_alpha=training_cfg["lora"]["alpha"],
        lora_dropout=training_cfg["lora"]["dropout"],
        lora_query=training_cfg["lora"]["query"],
        lora_key=training_cfg["lora"]["key"],
        lora_value=training_cfg["lora"]["value"],
        lora_projection=training_cfg["lora"]["projection"],
        lora_mlp=training_cfg["lora"]["mlp"],
        lora_head=training_cfg["lora"]["head"],
    )

    checkpoint_path = checkpoint_dir / "lit_model.pth"

    tokenizer = Tokenizer(lora_dir)
    prompt_style = load_prompt_style(lora_dir) if has_prompt_style(lora_dir) else PromptStyle.from_config(config)

    with open(input_file, "r") as f:
        data = InferenceDataset(json.load(f), tokenizer=tokenizer, prompt_style=prompt_style)
    longest_seq_length = max([len(d["encoded_prompt"]) for d in data])

    max_returned_tokens = longest_seq_length + max_new_tokens

    fabric.print(f"Loading model {str(checkpoint_path)!r} with {config.__dict__}", file=sys.stderr)
    t0 = time.perf_counter()
    with fabric.init_module(empty_init=True):
        model = GPT(config)
    fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.", file=sys.stderr)
    with fabric.init_tensor():
        # set the max_seq_length to limit the memory usage to what we need
        model.max_seq_length = max_returned_tokens
        # enable the kv cache
        model.set_kv_cache(batch_size=1)
    model.eval()

    model = fabric.setup(model)

    t0 = time.perf_counter()
    load_checkpoint(fabric, model, checkpoint_path, strict=False)
    load_checkpoint(fabric, model, lora_dir / "lit_model.pth.lora", strict=False)
    fabric.print(f"Time to load the model weights: {time.perf_counter() - t0:.02f} seconds.", file=sys.stderr)

    L.seed_everything(1234)
    generated_samples = []
    with jsonl.open(lora_dir / "predictions.jsonl", "w") as writer:
        for sample in data:
            y = generate(
                model,
                sample["encoded_prompt"].to(fabric.device),
                max_returned_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                eos_id=tokenizer.eos_id,
            )

            output = tokenizer.decode(y)
            output = output.split("### Response:")[1].strip()
            sample["prediction"] = output
            sample.pop("encoded_prompt", None)
            generated_samples.append(sample)
            writer.write(sample)

    with open(lora_dir / "predictions.json", "w") as f:
        json.dump(generated_samples, f)
    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB", file=sys.stderr)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    CLI(main)
