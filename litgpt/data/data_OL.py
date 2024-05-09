# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file
from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Union

import numpy as np
import torch
from lightning_utilities.core.imports import RequirementCache
from torch.utils.data import DataLoader, random_split

from litgpt.data import DataModule, SFTDataset, get_sft_collate_fn
from litgpt.prompts import PromptStyle
from litgpt.tokenizer import Tokenizer


@dataclass
class BalanceDatasetConfig:
    """Balance dataset configs."""

    classes: Sequence[str] = field(default_factory=list)
    percentage: float = 0.5


@dataclass
class OL(DataModule):
    """OL data module for supervised finetuning."""

    mask_prompt: bool = False
    """Whether to mask the prompt section from the label (with ``ignore_index``)."""
    val_split_fraction: float = 0.03865  # to get exactly 2000 validation samples,
    """The fraction of the dataset to use for the validation dataset. The rest is used for training."""
    prompt_style: Union[str, PromptStyle] = "ol"
    """The style to apply to instruction prompts. See `litgpt.prompts` for a list of available styles."""
    ignore_index: int = -100
    """The index to use for elements to be ignored in the label."""
    seed: int = 42
    """The random seed for creating the train/val splits and shuffling the dataset."""
    num_workers: int = 4
    """How many DataLoader processes to use for loading."""
    download_dir: Path = Path("./data/e3c")
    """The directory in which the downloaded dataset gets saved."""
    file_url: str | None = field(repr=False, default=None)
    """The URL from where to download the dataset."""
    file_name: str = field(repr=False, default="e3c.json")
    """The name of the dataset file."""
    file_name_validation: str | None = field(repr=False, default=None)
    """The name of the validation dataset file."""
    balance_cfg: BalanceDatasetConfig | None = field(repr=False, default=None)
    """The configuration of the balance dataset, if any."""

    tokenizer: Optional[Tokenizer] = field(default=None, init=False, repr=False)
    batch_size: int = field(default=1, init=False, repr=False)
    max_seq_length: int = field(default=-1, init=False, repr=False)
    train_dataset: Optional[SFTDataset] = field(default=None, init=False, repr=False)
    test_dataset: Optional[SFTDataset] = field(default=None, init=False, repr=False)
    sft_dataset_fn: Callable[..., SFTDataset] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.prompt_style, str):
            self.prompt_style = PromptStyle.from_name(self.prompt_style)

    def connect(
        self, tokenizer: Optional[Tokenizer] = None, batch_size: int = 1, max_seq_length: Optional[int] = None
    ) -> None:
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_seq_length = -1 if max_seq_length is None else max_seq_length

    def prepare_data(self) -> None:
        self.download_dir.mkdir(parents=True, exist_ok=True)
        download_if_missing(self.download_dir / self.file_name, self.file_url)

    def setup(self, stage: str = "") -> None:
        with open(self.download_dir / self.file_name, "r", encoding="utf-8") as file:
            data = json.load(file)

        if self.file_name_validation is None:
            # Partition the dataset into train and test
            train_data, test_data = random_split(
                data,
                [1.0 - self.val_split_fraction, self.val_split_fraction],
                generator=torch.Generator().manual_seed(self.seed),
            )
            train_data, test_data = list(train_data), list(test_data)
        else:
            train_data = data
            with open(self.download_dir / self.file_name_validation, "r", encoding="utf-8") as file:
                test_data = json.load(file)

        self.sft_dataset_fn = partial(
            SFTDataset,
            tokenizer=self.tokenizer,
            prompt_style=self.prompt_style,
            max_seq_length=self.max_seq_length,
            mask_prompt=self.mask_prompt,
            ignore_index=self.ignore_index,
        )

        self.train_dataset = self.sft_dataset_fn(data=train_data)
        self.test_dataset = self.sft_dataset_fn(data=test_data)

    def train_dataloader(self) -> DataLoader:
        train_dataset = self._create_balanced_batch(self.train_dataset.data, self.batch_size)
        return DataLoader(
            self.sft_dataset_fn(train_dataset),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=get_sft_collate_fn(max_seq_length=self.max_seq_length, ignore_index=self.ignore_index),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=get_sft_collate_fn(max_seq_length=self.max_seq_length, ignore_index=self.ignore_index),
        )

    def _create_balanced_batch(self, data: Sequence[Dict[str, Any]], batch_size: int) -> Sequence[Dict[str, Any]]:
        # retrieves the indices of the elements of the specified clesses and the other classes in data
        idx_specified_classes = np.array([i for i in range(len(data)) if data[i]["class"] in self.balance_cfg.classes])
        idx_other_classes = np.array([i for i in range(len(data)) if data[i]["class"] not in self.balance_cfg.classes])

        # If you are unmbalancing the dataset instead of balancing it, an error is raised
        if self.balance_cfg.percentage < len(idx_specified_classes) / len(data):
            raise ValueError(
                f"Percentage you specified ({self.balance_cfg.percentage}) is lower than "
                f"the original percentage in the dataset ({len(idx_specified_classes) / len(data)})."
            )

        np.random.shuffle(idx_specified_classes)
        np.random.shuffle(idx_other_classes)

        len_major_class = max(len(idx_specified_classes), len(idx_other_classes))

        # number of elements of the two groups
        el_per_specified_classes = int(batch_size * self.balance_cfg.percentage)
        el_other_classes = batch_size - el_per_specified_classes

        # indices for composing the batch
        # np.array([0, 1, ..., el_per_batch - 1]
        batch_idx_specified_classes = np.arange(0, el_per_specified_classes)
        batch_idx_other_classes = np.arange(0, el_other_classes)

        # Map the two groups in "bigger" and "smaller" groups
        if len(idx_specified_classes) > len(idx_other_classes):
            idx_bigger_group = idx_specified_classes
            el_per_batch_bigger_group = el_per_specified_classes
            batch_idx_bigger_group = batch_idx_specified_classes
            idx_smaller_group = idx_other_classes
            el_per_batch_smaller_group = el_other_classes
            batch_idx_smaller_group = batch_idx_other_classes
        else:
            idx_bigger_group = idx_other_classes
            el_per_batch_bigger_group = el_other_classes
            batch_idx_bigger_group = batch_idx_other_classes
            idx_smaller_group = idx_specified_classes
            el_per_batch_smaller_group = el_per_specified_classes
            batch_idx_smaller_group = batch_idx_specified_classes

        j = 0
        data_idx = list()
        # create a list of indices for composing the balanced dataset
        for i in range(0, len_major_class, el_per_batch_bigger_group):
            # access for each group with the batch indices array
            # at each iteration different elements are retrieved from the two groups
            # for example:
            # el_per_batch_bigger_group = 24
            # el_per_batch_smaller_group = 8
            # <iteration 0>
            #   i = 0
            #   j = 0
            #   batch_idx_bigger_group + i = np.array(
            #       [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
            #   )
            #   batch_idx_smaller_group + j = np.array([0, 1, 2, 3, 4, 5, 6, 7])
            #   # use `batch_idx_bigger_group` to access to the `idx_bigger_group`
            #   # use `batch_idx_smaller_group` to access to the `idx_smaller_group`
            # <iteration 1>
            #   i = 24
            #   j = 8
            #   batch_idx_bigger_group + i = np.array(
            #       [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]
            #   )
            #   batch_idx_smaller_group + j = np.array([8, 9, 10, 11, 12, 13, 14, 15])
            # ...
            batch = np.concatenate(
                (
                    idx_bigger_group[(batch_idx_bigger_group + i) % len(idx_bigger_group)],
                    idx_smaller_group[(batch_idx_smaller_group + j) % len(idx_smaller_group)],
                ),
                axis=None,
            )
            j += el_per_batch_smaller_group

            np.random.shuffle(batch)
            data_idx.extend(batch.tolist())

        # convert to array for accessing with a np.array
        balanced_data = np.array(data)[data_idx].tolist()

        return balanced_data


def download_if_missing(file_path: Path, file_url: str, mode: str = "w", stream: bool = False) -> None:
    """Downloads the raw json data file and saves it in the given destination."""
    if file_path.exists() and file_path.stat().st_size > 0:
        return
    requests_available = RequirementCache("requests")
    if not requests_available:
        raise ModuleNotFoundError(str(requests_available))
    import requests

    response = requests.get(file_url, stream=stream)
    with open(file_path, mode, encoding=None if mode == "wb" else "utf-8") as f:
        if stream:
            # credit: https://github.com/karpathy/llama2.c/blob/b3c4b6/tinystories.py#L25-L38
            from tqdm import tqdm

            pbar = tqdm(
                desc=str(file_path),
                total=int(response.headers.get("content-length", 0)),
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
            )
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                pbar.update(size)
            pbar.close()
        else:
            f.write(response.text)
