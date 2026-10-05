

from __future__ import annotations

import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = ["read_meta", "check_split", "NeBLaDataset"]


def read_meta(root: str) -> dict:
    with open(os.path.join(root, "dataset.json")) as fh:
        return json.load(fh)


def check_split(meta: dict, train: list, val: list, test: list | None = None) -> None:
    """Refuse empty, unknown or overlapping case lists."""
    known = set(meta["cases"])
    groups = {"train": list(train or []), "val": list(val or []), "test": list(test or [])}
    if not groups["train"] or not groups["val"]:
        raise ValueError("give both --train-cases and --val-cases (validation must be separate cases)")
    for name, cases in groups.items():
        unknown = set(cases) - known
        if unknown:
            raise ValueError(f"{name}: unknown case(s) {sorted(unknown)}")
    names = list(groups)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            both = set(groups[names[i]]) & set(groups[names[j]])
            if both:
                raise ValueError(f"cases in both {names[i]} and {names[j]}: {sorted(both)}")


class NeBLaDataset(Dataset):
    def __init__(self, root: str, cases: list):
        self.root = root
        self.meta = read_meta(root)
        self.cases = list(cases)
        D, H, W = self.meta["volume_shape"]
        self.shape = (D, H, W)
        for c in self.cases:
            s = np.load(os.path.join(root, "simpx", c + ".npy"), mmap_mode="r")
            v = np.load(os.path.join(root, "volumes", c + ".npy"), mmap_mode="r")
            if s.shape != (D, self.meta["n_rays"]) or v.shape != (D, H, W):
                raise ValueError(f"{c}: simpx {s.shape} / volume {v.shape} do not match {self.shape}")

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, i):
        c = self.cases[i]
        s = np.load(os.path.join(self.root, "simpx", c + ".npy")).astype(np.float32)
        v = np.load(os.path.join(self.root, "volumes", c + ".npy")).astype(np.float32)
        return {"case": c, "simpx": torch.from_numpy(s)[None], "volume": torch.from_numpy(v)[None]}
