"""Data pipeline — JSONL dataset, domain-balanced sampling, DataLoader."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Sampler, WeightedRandomSampler

from .tokenizer import Tokenizer


class TextDataset(Dataset):
    """Reads all JSONL files from data/<domain>/ sub-directories,
    tokenizes on-the-fly, and packs/pads to max_seq_len."""

    def __init__(
        self,
        data_dir: str = "data",
        max_seq_len: int = 512,
        tokenizer_name: str = "gpt2",
        domains: list[str] | None = None,
        split: str = "train",
        val_split: float = 0.1,
    ):
        self.tokenizer = Tokenizer(tokenizer_name, max_len=max_seq_len)
        self.max_seq_len = max_seq_len
        self.samples: list[dict] = []  # {"text": ..., "domain": ...}
        self.domain_indices: dict[str, list[int]] = {}

        data_path = Path(data_dir)
        for domain_dir in sorted(data_path.iterdir()):
            if not domain_dir.is_dir():
                continue
            domain = domain_dir.name
            if domains and domain not in domains:
                continue

            self.domain_indices[domain] = []
            for jsonl_file in sorted(domain_dir.glob("*.jsonl")):
                with open(jsonl_file, "r", encoding="utf-8") as f:
                    for line in f:
                        entry = json.loads(line)
                        idx = len(self.samples)
                        self.samples.append(
                            {"text": entry["text"], "domain": domain}
                        )
                        self.domain_indices[domain].append(idx)

        # Deterministic train/val split
        n = len(self.samples)
        indices = list(range(n))
        # Use a fixed seed for reproducibility
        rng = torch.Generator().manual_seed(42)
        perm = torch.randperm(n, generator=rng).tolist()

        split_idx = int(n * (1 - val_split))
        if split == "train":
            self._indices = sorted(perm[:split_idx])
        else:
            self._indices = sorted(perm[split_idx:])

        # Rebuild domain index for this split
        split_set = set(self._indices)
        self.split_domain_indices: dict[str, list[int]] = {}
        for i, global_idx in enumerate(self._indices):
            domain = self.samples[global_idx]["domain"]
            if domain not in self.split_domain_indices:
                self.split_domain_indices[domain] = []
            self.split_domain_indices[domain].append(i)

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        global_idx = self._indices[idx]
        text = self.samples[global_idx]["text"]
        domain = self.samples[global_idx]["domain"]

        ids = self.tokenizer.encode(text, add_eos=True)

        # Truncate
        if len(ids) > self.max_seq_len:
            ids = ids[: self.max_seq_len]

        real_len = len(ids)

        # Pad
        pad_len = self.max_seq_len - real_len
        attention_mask = [1] * real_len + [0] * pad_len
        ids = ids + [self.tokenizer.pad_id] * pad_len

        input_ids = torch.tensor(ids[:-1], dtype=torch.long)
        labels = torch.tensor(ids[1:], dtype=torch.long)
        attention_mask = torch.tensor(attention_mask[:-1], dtype=torch.long)

        # Mask padding positions so cross_entropy ignores them
        if real_len - 1 < len(labels):
            labels[real_len - 1:] = -100

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

    @property
    def domains(self) -> list[str]:
        return list(self.split_domain_indices.keys())


def build_domain_balanced_sampler(dataset: TextDataset) -> Sampler:
    """WeightedRandomSampler that gives equal weight to each domain."""
    n_domains = len(dataset.split_domain_indices)
    if n_domains == 0:
        raise ValueError("No domains found in dataset")

    weights = torch.zeros(len(dataset))
    for domain, indices in dataset.split_domain_indices.items():
        w = 1.0 / (n_domains * len(indices))
        for i in indices:
            weights[i] = w

    return WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)


def build_dataloader(
    data_dir: str = "data",
    max_seq_len: int = 512,
    batch_size: int = 32,
    tokenizer_name: str = "gpt2",
    domains: list[str] | None = None,
    split: str = "train",
    val_split: float = 0.1,
    num_workers: int = 0,
    domain_balanced: bool = True,
) -> DataLoader:
    """Convenience function to build a DataLoader."""
    dataset = TextDataset(
        data_dir=data_dir,
        max_seq_len=max_seq_len,
        tokenizer_name=tokenizer_name,
        domains=domains,
        split=split,
        val_split=val_split,
    )

    sampler = build_domain_balanced_sampler(dataset) if domain_balanced and split == "train" else None
    shuffle = (sampler is None) and (split == "train")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train"),
        persistent_workers=(num_workers > 0),
    )
