# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainArgs:
    """Training-related arguments"""

    save_interval: Optional[int] = 1000
    """Number of optimizer steps between saving checkpoints"""
    log_interval: int = 1
    """Number of iterations between logging calls"""
    global_batch_size: int = 64
    """Number of samples between optimizer steps across data-parallel ranks"""
    micro_batch_size: int = 4
    """Number of samples per data-parallel rank"""
    lr_warmup_steps: Optional[int] = 100
    """Number of iterations with learning rate warmup active"""
    lr_warmup_fraction: Optional[float] = None
    """The fraction of an epoch to use for learning rate warmup"""
    epochs: Optional[int] = None
    """Number of epochs to train on"""
    # TODO: `pretrain` is the only script using `max_tokens` explicitly. replace it with epoch_size*epochs?
    max_tokens: Optional[int] = None
    """Total number of tokens to train on"""
    max_steps: Optional[int] = None
    """Limits the number of optimizer steps to run"""
    max_seq_length: Optional[int] = None
    """Limits the length of samples"""
    tie_embeddings: Optional[bool] = None
    """Whether to tie the embedding weights with the language modeling head weights"""

    # Optimization args
    learning_rate: float = 1e-3
    weight_decay: float = 0.02
    beta1: float = 0.9
    beta2: float = 0.95
    max_norm: Optional[float] = None
    min_lr: float = 6e-5

    def __post_init__(self) -> None:
        if self.lr_warmup_fraction and self.lr_warmup_steps:
            raise ValueError(
                "Can't provide both `--train.lr_warmup_fraction` and `--train.lr_warmup_steps`. Choose one."
            )
        if self.lr_warmup_fraction and not (0 <= self.lr_warmup_fraction <= 1):
            raise ValueError("`--train.lr_warmup_fraction` must be between 0 and 1.")

    def gradient_accumulation_iters(self, devices: int) -> int:
        """Number of iterations between gradient synchronizations"""
        gradient_accumulation_iters = self.batch_size(devices) // self.micro_batch_size
        assert gradient_accumulation_iters > 0
        return gradient_accumulation_iters

    def batch_size(self, devices: int) -> int:
        """Number of samples between optimizer steps per data-parallel rank"""
        batch_size = self.global_batch_size // devices
        assert batch_size > 0
        return batch_size

    def warmup_iters(self, devices: int, max_iters: int, train_dataloader) -> int:
        """Number of iterations to warm up the learning rate."""
        if self.lr_warmup_fraction:
            return min(max_iters, math.ceil(self.lr_warmup_fraction * len(train_dataloader)))
        if self.lr_warmup_steps:
            return min(max_iters, self.lr_warmup_steps * self.gradient_accumulation_iters(devices))
        return 0


@dataclass
class EvalArgs:
    """Evaluation-related arguments"""

    interval: int | None = None
    """Number of optimizer steps between evaluation calls"""
    max_new_tokens: Optional[int] = None
    """Number of tokens to generate"""
    max_iters: int | float = float("inf")
    """Number of iterations"""
    initial_validation: bool = False
    """Whether to evaluate on the validation set at the beginning of the training"""
    qualitative_val_sample_idx: int | None = None
    temperature: float = 0.0


@dataclass
class LoraArgs:
    r: int = 128
    alpha: int = 256
    dropout: float = 0.05
    query: bool = True
    key: bool = True
    value: bool = True
    projection: bool = True
    mlp: bool = True
    head: bool = True


@dataclass
class MLFlowArgs:
    experiment_name: str
    run_name: str
    tracking_uri: str = "http://192.168.3.78:5100"
    run_id: str | None = None
    synchronous: bool = False
