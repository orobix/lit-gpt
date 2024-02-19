# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import hydra
import jsonlines as jsonl
import lightning as L
import torch
from lightning.fabric.plugins import BitsandbytesPrecision
from omegaconf import DictConfig, OmegaConf

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from generate.base import generate
from lit_gpt import Tokenizer
from lit_gpt.lora import GPT, Config
from lit_gpt.utils import (
    check_valid_checkpoint_dir,
    dotdict,
    get_default_supported_precision,
    gptq_quantization,
    lazy_load,
    tokenize_dataset,
)


@hydra.main(
    version_base="1.3", config_path="../configs", config_name="inference_config.yaml"
)
def setup(cfg: DictConfig):
    cfg = dotdict(OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))
    cfg.model.lora_path = Path(cfg.model.lora_path)
    cfg.data.outfile_path = Path(cfg.data.outfile_path)
    cfg.model.checkpoint_dir = Path(cfg.model.checkpoint_dir)
    training_cfg = dotdict(OmegaConf.load(cfg.model.lora_path.parent / "config.yaml"))
    cfg.model.lora = training_cfg.lora

    precision = cfg.model.precision or get_default_supported_precision(training=False)

    plugins = None
    if cfg.model.quantize is not None and cfg.model.quantize.startswith("bnb."):
        if "mixed" in precision:
            raise ValueError("Quantization and mixed precision is not supported.")
        dtype = {
            "16-true": torch.float16,
            "bf16-true": torch.bfloat16,
            "32-true": torch.float32,
        }[precision]
        plugins = BitsandbytesPrecision(cfg.model.quantize[4:], dtype)
        precision = None

    fabric = L.Fabric(devices=1, precision=precision, plugins=plugins)
    fabric.launch(main, cfg=cfg)


def main(fabric: L.Fabric, cfg: dotdict[str, Any]) -> None:
    check_valid_checkpoint_dir(cfg.model.checkpoint_dir)

    config = Config.from_json(
        cfg.model.checkpoint_dir / "lit_config.json",
        r=cfg.model.lora.r,
        alpha=cfg.model.lora.alpha,
        dropout=cfg.model.lora.dropout,
        to_query=cfg.model.lora.query,
        to_key=cfg.model.lora.key,
        to_value=cfg.model.lora.value,
        to_projection=cfg.model.lora.projection,
        to_mlp=cfg.model.lora.mlp,
        to_head=cfg.model.lora.head,
    )

    if cfg.model.quantize == "gptq.int4":
        model_file = "lit_model_gptq.4bit.pth"
        if not (cfg.model.checkpoint_dir / model_file).is_file():
            raise ValueError("Please run `python quantize/gptq.py` first")
    else:
        model_file = "lit_model.pth"

    tokenizer = Tokenizer(cfg.model.checkpoint_dir)
    tokenized_dataset = tokenize_dataset(
        fabric,
        tokenizer,
        cfg.data.inference_filepath,
    )

    prompt_lengths = [sample["encoded_prompt"].size(0) for sample in tokenized_dataset]
    max_returned_tokens = max(prompt_lengths) + cfg.generation.max_generated_tokens

    fabric.print(
        f"Loading model {str(cfg.model.checkpoint_path)!r} with {config.__dict__}",
        file=sys.stderr,
    )
    t0 = time.perf_counter()
    with fabric.init_module(empty_init=True), gptq_quantization(
        cfg.model.quantize == "gptq.int4"
    ):
        model = GPT(config)
    fabric.print(
        f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.",
        file=sys.stderr,
    )
    with fabric.init_tensor():
        # set the max_seq_length to limit the memory usage to what we need
        model.max_seq_length = max_returned_tokens
        # enable the kv cache
        model.set_kv_cache(batch_size=1)
    model.eval()

    t0 = time.perf_counter()
    checkpoint = lazy_load(cfg.model.checkpoint_dir / "lit_model.pth")
    lora_checkpoint = lazy_load(cfg.model.lora_path)
    checkpoint.update(lora_checkpoint.get("model", lora_checkpoint))
    model.load_state_dict(checkpoint)
    fabric.print(
        f"Time to load the model weights: {time.perf_counter() - t0:.02f} seconds.",
        file=sys.stderr,
    )

    model = fabric.setup(model)

    if fabric.global_rank == 0:
        os.makedirs(cfg.data.outfile_path.parent, exist_ok=True)
    with jsonl.open(cfg.data.outfile_path, "w") as writer:
        for sample in tokenized_dataset:
            sample["prediction"] = predict(
                model,
                tokenizer,
                sample["encoded_prompt"],
                cfg.generation.max_generated_tokens,
                cfg.generation.top_k,
                cfg.generation.temperature,
            )
            sample.pop("encoded_prompt", None)
            writer.write(sample)


def predict(
    model,
    tokenizer: Tokenizer,
    encoded: torch.Tensor,
    max_generated_tokens: int,
    top_k: Optional[int] = 200,
    temperature: float = 0.8,
) -> str:
    """Generates a response based on a given instruction and an optional input.
    This script will only work with checkpoints from the instruction-tuned GPT-LoRA model.
    See `finetune/lora.py`.

    Args:
        tokenizer (Tokenizer): The tokenizer.
        encoded (Tensor): The encoded prompt.
        max_generated_tokens (int): The maximum number of generated tokens.
        checkpoint_dir: The path to the checkpoint folder with pretrained GPT weights.
        top_k (int, optional): The number of top most probable tokens to consider in the sampling process.
        temperature (float): A value controlling the randomness of the sampling process.
            Higher values result in more random samples.
    """
    L.seed_everything(1234)
    y = generate(
        model,
        encoded,
        encoded.size(0) + max_generated_tokens,
        temperature=temperature,
        top_k=top_k,
        eos_id=tokenizer.eos_id,
    )

    output = tokenizer.decode(y)
    output = output.split("### Response:")[1].strip()

    return output


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    setup()
