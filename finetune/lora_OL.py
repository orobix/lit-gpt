# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import os

from dotenv import find_dotenv, load_dotenv

_ = load_dotenv(find_dotenv(".env"))  # read local .env file

import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import hydra
import lightning as L
import mlflow
import torch
from lightning.fabric.loggers import CSVLogger
from lightning.fabric.plugins import BitsandbytesPrecision
from lightning.fabric.strategies import FSDPStrategy
from lightning.fabric.utilities import ThroughputMonitor
from omegaconf import DictConfig, OmegaConf

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))


from generate.base import generate
from lit_gpt.lora import GPT, Block, Config, lora_filter, mark_only_lora_as_trainable
from lit_gpt.mlflow_utils import CustomMLFlowLogger
from lit_gpt.tokenizer import Tokenizer
from lit_gpt.utils import (
    check_valid_checkpoint_dir,
    chunked_cross_entropy,
    create_balanced_batch,
    dotdict,
    get_default_supported_precision,
    load_checkpoint,
    num_parameters,
)
from scripts.prepare_data import generate_prompt, prepare


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def setup(cfg: DictConfig) -> None:

    cfg = dotdict(OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))
    cfg.experiment.out_dir = Path(cfg.experiment.out_dir)
    cfg.logging_and_checkpoint.checkpoint_dir = Path(
        cfg.logging_and_checkpoint.checkpoint_dir
    )
    cfg.data.data_dir = Path(cfg.data.data_dir)

    if cfg.data.json.train_file_path:
        prepare(
            destination_path=cfg.data.data_dir,
            checkpoint_dir=cfg.logging_and_checkpoint.checkpoint_dir,
            test_split_fraction=cfg.data.valset_split_percentage,
            data_file_name=cfg.data.json.train_file_path,
            data_file_name_val=cfg.data.json.val_file_path,
        )

    # load datasets
    train_data = torch.load(cfg.data.data_dir / "train.pt")
    val_data = torch.load(cfg.data.data_dir / "test.pt")
    if cfg.data.balanced_batch.classes:
        processed_train_data = create_balanced_batch(
            train_data,
            cfg.data.balanced_batch,
            cfg.data.batch_size * cfg.data.micro_batch_size,
        )
    else:
        processed_train_data = train_data

    # compute hyper-parameters on the fly
    cfg.data.epoch_size = len(processed_train_data)
    if cfg.logging_and_checkpoint.eval_interval is None:
        cfg.logging_and_checkpoint.eval_interval = cfg.data.epoch_size // (
            cfg.data.batch_size * cfg.experiment.devices
        )
    if cfg.logging_and_checkpoint.save_interval is None:
        cfg.logging_and_checkpoint.save_interval = cfg.data.epoch_size // (
            cfg.data.batch_size * cfg.experiment.devices
        )
    cfg.data.gradient_accumulation_iters = (
        cfg.data.batch_size // cfg.data.micro_batch_size
    )
    cfg.experiment.max_iters = (
        cfg.experiment.num_epochs
        * (cfg.data.epoch_size // cfg.data.micro_batch_size)
        // cfg.experiment.devices
    )

    precision = cfg.experiment.precision or get_default_supported_precision(
        training=True
    )

    plugins = None
    if cfg.experiment.quantize is not None and cfg.experiment.quantize.startswith(
        "bnb."
    ):
        if "mixed" in precision:
            raise ValueError("Quantization and mixed precision is not supported.")
        dtype = {
            "16-true": torch.float16,
            "bf16-true": torch.bfloat16,
            "32-true": torch.float32,
        }[precision]
        plugins = BitsandbytesPrecision(cfg.experiment.quantize[4:], dtype)
        precision = None

    if cfg.experiment.devices > 1:
        if cfg.experiment.quantize:
            raise NotImplementedError(
                "Quantization is currently not supported for multi-GPU training. Please set devices=1 when using the"
                " --quantize flag."
            )
        strategy = FSDPStrategy(
            auto_wrap_policy={Block},
            activation_checkpointing_policy={Block},
            state_dict_type="full",
            limit_all_gathers=True,
            cpu_offload=False,
        )
    else:
        strategy = "auto"

    print(cfg.experiment)
    logger = CSVLogger(
        cfg.experiment.out_dir.parent,
        cfg.experiment.out_dir.name,
        flush_logs_every_n_steps=cfg.logging_and_checkpoint.log_interval,
    )
    mlf_logger = CustomMLFlowLogger(
        experiment_name=cfg.logging_and_checkpoint.experiment_name,
        tracking_uri=cfg.logging_and_checkpoint.mlflow_tracking_uri,
        run_name=cfg.logging_and_checkpoint.run_name,
        synchronous=cfg.logging_and_checkpoint.synchronous,
    )
    mlf_logger.log_hyperparams(cfg)
    fabric = L.Fabric(
        devices=cfg.experiment.devices,
        strategy=strategy,
        precision=precision,
        loggers=[logger, mlf_logger],
        plugins=plugins,
    )
    fabric.print(cfg)

    fabric.launch(main, cfg, train_data, val_data)


def main(
    fabric: L.Fabric,
    cfg: dotdict[str, Any],
    train_data: Sequence[Dict[str, Any]],
    val_data: Sequence[Dict[str, Any]],
) -> None:
    lora_cfg = cfg.lora
    out_dir = cfg.experiment.out_dir
    checkpoint_dir = cfg.logging_and_checkpoint.checkpoint_dir
    check_valid_checkpoint_dir(checkpoint_dir)

    fabric.seed_everything(1337)  # same seed for every process to init model (FSDP)

    if fabric.global_rank == 0:
        os.makedirs(out_dir, exist_ok=True)
        with open(out_dir / "config.yaml", "w") as f:
            OmegaConf.save(cfg.as_dict(), f)

    if fabric.global_rank == 0:
        with mlflow.start_run(run_id=fabric.loggers[-1].run_id):
            mlflow.log_artifacts(cfg.data.data_dir)

    if not any(
        (
            lora_cfg.query,
            lora_cfg.key,
            lora_cfg.value,
            lora_cfg.projection,
            lora_cfg.mlp,
            lora_cfg.head,
        )
    ):
        fabric.print("Warning: all LoRA layers are disabled!")
    config = Config.from_name(
        name=checkpoint_dir.name,
        r=lora_cfg.r,
        alpha=lora_cfg.alpha,
        dropout=lora_cfg.dropout,
        to_query=lora_cfg.query,
        to_key=lora_cfg.key,
        to_value=lora_cfg.value,
        to_projection=lora_cfg.projection,
        to_mlp=lora_cfg.mlp,
        to_head=lora_cfg.head,
    )

    checkpoint_path = checkpoint_dir / "lit_model.pth"
    print(
        f"{fabric.global_rank}: Loading model {str(checkpoint_path)!r} with {config.__dict__}"
    )
    with fabric.init_module(empty_init=(cfg.experiment.devices > 1)):
        model = GPT(config)
    mark_only_lora_as_trainable(model)

    fabric.print(
        f"Number of trainable parameters: {num_parameters(model, requires_grad=True):,}"
    )
    fabric.print(
        f"Number of non trainable parameters: {num_parameters(model, requires_grad=False):,}"
    )

    model = fabric.setup_module(model)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if isinstance(fabric.strategy.precision, BitsandbytesPrecision):
        import bitsandbytes as bnb

        optimizer = bnb.optim.PagedAdamW(
            trainable_params,
            lr=cfg.experiment.learning_rate,
            weight_decay=cfg.experiment.weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=cfg.experiment.learning_rate,
            weight_decay=cfg.experiment.weight_decay,
        )
    optimizer = fabric.setup_optimizers(optimizer)
    scheduler = get_lr_scheduler(
        optimizer,
        warmup_steps=cfg.experiment.warmup_steps,
        max_steps=cfg.experiment.max_iters // cfg.data.gradient_accumulation_iters,
    )

    # strict=False because missing keys due to LoRA weights not contained in state dict
    load_checkpoint(fabric, model, checkpoint_path, strict=False)

    fabric.seed_everything(1337 + fabric.global_rank)

    train_time = time.perf_counter()
    train(
        fabric,
        model,
        optimizer,
        scheduler,
        train_data,
        val_data,
        cfg,
    )
    fabric.print(f"Training time: {(time.perf_counter()-train_time):.2f}s")
    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")

    # Save the final LoRA checkpoint at the end of training
    save_path = out_dir / "lit_model_lora_finetuned.pth"
    save_lora_checkpoint(fabric, model, save_path)


def train(
    fabric: L.Fabric,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    train_data: Sequence[Dict[str, Any]],
    val_data: Sequence[Dict[str, Any]],
    cfg: dotdict[str, Any],
) -> None:
    tokenizer = Tokenizer(cfg.logging_and_checkpoint.checkpoint_dir)
    longest_seq_length, longest_seq_ix = get_longest_seq_length(train_data)
    model.max_seq_length = min(
        longest_seq_length, cfg.data.max_seq_length or float("inf")
    )
    fabric.print(
        f"The longest sequence length in the train data is {longest_seq_length}, the model's maximum sequence length is"
        f" {model.max_seq_length} and context length is {model.config.block_size}"
    )

    validate(fabric, model, val_data, cfg, tokenizer)  # sanity check

    throughput = ThroughputMonitor(fabric, window_size=50)
    step_count = 0
    total_lengths = 0
    total_t0 = time.perf_counter()

    losses = []
    iter_num = 0

    # Epochs management added to simplify experiments setting
    for _ in range(0, cfg.experiment.num_epochs):
        # Random shuffle added to increase the variability in training
        if cfg.data.balanced_batch.classes:
            processed_train_data = create_balanced_batch(
                train_data,
                cfg.data.balanced_batch,
                cfg.data.batch_size * cfg.data.micro_batch_size,
            )
        else:
            processed_train_data = random.shuffle(train_data)

        # Only for the first iteration
        if iter_num == 0:
            # get the longest sequence length and its index
            # use it in the first microbatch to check whether or not the model fits in the GPU
            longest_seq_length, longest_seq_ix = get_longest_seq_length(
                processed_train_data
            )

        for i in range(0, len(processed_train_data), cfg.data.micro_batch_size):
            iter_num += 1
            iter_t0 = time.perf_counter()

            # The default microbatch definition selects elements in random way (idx selected in the get_batch function). To guarantee the inclusion of the entire dataset in a epoch a continuous selection of data ids has been added
            idx = None
            if not cfg.data.random_microbatch:
                idx = list(range(i, i + cfg.data.micro_batch_size))
                idx = [i % len(processed_train_data) for i in idx]
            input_ids, targets = get_batch(
                fabric,
                processed_train_data,
                cfg,
                idx,
                longest_seq_ix if iter_num == 1 else None,
            )

            is_accumulating = iter_num % cfg.data.gradient_accumulation_iters != 0
            with fabric.no_backward_sync(model, enabled=is_accumulating):
                logits = model(input_ids, lm_head_chunk_size=128)
                # shift the targets such that output n predicts token n+1
                logits[-1] = logits[-1][..., :-1, :]
                loss = chunked_cross_entropy(logits, targets[..., 1:])
                fabric.backward(loss / cfg.data.gradient_accumulation_iters)
                losses.append(loss.item())

            if not is_accumulating:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                step_count += 1

            total_lengths += input_ids.numel()
            if iter_num % cfg.logging_and_checkpoint.log_interval == 0:
                loss_item = loss.item()  # expensive device-to-host synchronization
                t1 = time.perf_counter()
                throughput.update(
                    time=t1 - total_t0,
                    batches=iter_num,
                    samples=iter_num * cfg.data.micro_batch_size,
                    lengths=total_lengths,
                )
                throughput.compute_and_log(step=iter_num)
                fabric.print(
                    f"iter {iter_num} step {step_count}: loss {loss_item:.4f}, iter time:"
                    f" {(t1 - iter_t0) * 1000:.2f}ms{' (optimizer.step)' if not is_accumulating else ''}"
                )
                # fabric.log('train_loss', loss_item, iter_num)
                fabric.log("train_loss", sum(losses) / len(losses), iter_num)
                losses = []

            if (
                not is_accumulating
                and step_count % cfg.logging_and_checkpoint.eval_interval == 0
            ):
                t0 = time.perf_counter()
                val_loss = validate(fabric, model, val_data, cfg, tokenizer)
                fabric.log("val_loss", val_loss, iter_num)
                t1 = time.perf_counter() - t0
                fabric.print(
                    f"step {iter_num}: val loss {val_loss.item():.4f}, val time: {t1 * 1000:.2f}ms\n"
                )
                fabric.barrier()
            if (
                not is_accumulating
                and step_count % cfg.logging_and_checkpoint.save_interval == 0
            ):
                checkpoint_path = (
                    cfg.experiment.out_dir / f"iter-{iter_num:06d}-ckpt.pth"
                )
                save_lora_checkpoint(fabric, model, checkpoint_path)


# FSDP has issues with `inference_mode`
@torch.no_grad()
def validate(
    fabric: L.Fabric,
    model: GPT,
    val_data: Sequence[Dict[str, Any]],
    cfg: dotdict[str, Any],
    tokenizer: Tokenizer,
) -> torch.Tensor:
    fabric.print("Validating ...\n")
    model.eval()
    losses = []
    for i in range(0, len(val_data), cfg.data.micro_batch_size):
        # print(f"{fabric.global_rank}:{i} in validation")
        idx = list(range(i, (i + cfg.data.micro_batch_size)))
        idx = [i % len(val_data) for i in idx]
        input_ids, targets = get_batch(fabric, val_data, cfg, idx)
        logits = model(input_ids)
        losses.append(
            chunked_cross_entropy(logits[..., :-1, :], targets[..., 1:], chunk_size=0)
        )
    val_loss = sum(losses) / len(losses)

    # produce an example:
    qualitative_val_sample_idx = min(
        cfg.eval.qualitative_val_sample_idx or 0, len(val_data) - 1
    )
    sample = val_data[qualitative_val_sample_idx]
    fabric.print(f"----- Instruction: -----\n{sample['instruction']}")
    fabric.print(f"----- Input: -----\n{sample['input']}")
    prompt = generate_prompt(
        {"instruction": sample["instruction"], "input": sample["input"]}
    )
    encoded = tokenizer.encode(prompt, device=fabric.device)
    with fabric.init_tensor():
        # do not set `max_seq_length=max_returned_token` because memory is not a concern here
        model.set_kv_cache(batch_size=1)
    output = generate(
        model,
        encoded[: cfg.data.max_seq_length],
        max_returned_tokens=min(
            len(encoded) + cfg.eval.max_new_tokens, model.max_seq_length
        ),
        temperature=cfg.eval.temperature,
        eos_id=tokenizer.eos_id,
    )
    model_prediction = tokenizer.decode(output).split("### Response:")[1].strip()
    fabric.print(f"----- Model prediction: -----\n{model_prediction}")
    fabric.print(f"----- GT: -----\n{sample['output']}\n")

    model.train()
    return val_loss


def get_batch(
    fabric: L.Fabric,
    data: Sequence[Dict[str, Any]],
    cfg: dotdict[str, Any],
    ix: Optional[List[int]] = None,
    longest_seq_ix: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not ix:  # random selection of data in a microbatch
        x = torch.randint(len(data), (cfg.data.micro_batch_size,))

    if longest_seq_ix is not None:
        # force the longest sample at the beginning so potential OOMs happen right away
        ix[0] = longest_seq_ix

    input_ids = [data[i]["input_ids"].type(torch.int64) for i in ix]
    labels = [data[i]["labels"].type(torch.int64) for i in ix]

    # this could be `longest_seq_length` to have a fixed size for all batches
    max_len = max(len(s) for s in input_ids)

    def pad_right(x, pad_id):
        # pad right based on the longest sequence
        n = max_len - len(x)
        return torch.cat((x, torch.full((n,), pad_id, dtype=x.dtype)))

    x = torch.stack([pad_right(x, pad_id=0) for x in input_ids])
    y = torch.stack([pad_right(x, pad_id=-1) for x in labels])

    # print(f"{fabric.global_rank}: pad done")

    # Truncate if needed
    if cfg.data.max_seq_length:
        x = x[:, : cfg.data.max_seq_length]
        y = y[:, : cfg.data.max_seq_length]
    # print(f"{fabric.global_rank}: moving to cuda")
    if fabric.device.type == "cuda" and x.device.type == "cpu":
        x, y = fabric.to_device((x.pin_memory(), y.pin_memory()))
    else:
        x, y = fabric.to_device((x, y))
    # print(f"{fabric.global_rank}: moved to cuda")
    return x, y


def get_lr_scheduler(
    optimizer, warmup_steps: int, max_steps: int
) -> torch.optim.lr_scheduler.LRScheduler:
    # linear warmup followed by cosine annealing
    scheduler1 = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: step / warmup_steps
    )
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=(max_steps - warmup_steps)
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, [scheduler1, scheduler2], milestones=[warmup_steps]
    )


def get_longest_seq_length(data: Sequence[Dict[str, Any]]) -> Tuple[int, int]:
    # find out the minimum max_seq_length required during fine-tuning (saves memory!)
    lengths = [len(d["input_ids"]) for d in data]
    longest_seq_length = max(lengths)
    longest_seq_ix = lengths.index(longest_seq_length)
    return longest_seq_length, longest_seq_ix


def save_lora_checkpoint(
    fabric: L.Fabric, model: torch.nn.Module, file_path: Path
) -> None:
    fabric.print(f"Saving LoRA weights to {str(file_path)!r}")
    fabric.save(file_path, {"model": model}, filter={"model": lora_filter})


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    setup()
