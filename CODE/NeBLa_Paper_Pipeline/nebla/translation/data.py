"""Image folders for the translation module and the teeth segmenter.

Every image is read as grayscale, resized to the SimPX size (128 x 256,
rows x columns) and scaled to [0, 1]. ``.npy`` files are used as they are
(SimPX are already 128 x 256 in [0, 1]).
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

__all__ = ["load_px", "UnpairedPX", "SegPairs", "IMG_EXT"]

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".npy")


def load_px(path: str, size=(128, 256), flip_lr: bool = False) -> np.ndarray:
    """(H, W) float32 in [0, 1]."""
    if path.endswith(".npy"):
        a = np.load(path).astype(np.float32)
        if a.max() > 1.5:
            a = a / 255.0
        if a.shape != tuple(size):
            a = np.asarray(Image.fromarray(a).resize((size[1], size[0]), Image.BILINEAR), np.float32)
    else:
        im = Image.open(path)
        if im.mode in ("I;16", "I;16B", "I;16L", "I", "F"):                              
            a = np.asarray(im, np.float32)
            a = (a - a.min()) / max(float(a.max() - a.min()), 1e-6)
            a = np.asarray(Image.fromarray(a).resize((size[1], size[0]), Image.BILINEAR), np.float32)
        else:
            im = im.convert("L").resize((size[1], size[0]), Image.BILINEAR)
            a = np.asarray(im, np.float32) / 255.0
    if flip_lr:
        a = a[:, ::-1]
    return np.clip(a, 0.0, 1.0).copy()


def _files(folder, ext=IMG_EXT, cases=None):
    out = sorted(os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(ext)
                 and (cases is None or os.path.splitext(f)[0] in set(cases)))
    if not out:
        raise ValueError(f"no images in {folder}")
    if cases is not None:
        found = {os.path.splitext(os.path.basename(f))[0] for f in out}
        missing = set(cases) - found
        if missing:
            raise ValueError(f"no file for case(s) {sorted(missing)} in {folder}")
    return out


class UnpairedPX(Dataset):
    """Real PX (domain X) and SimPX (domain Y), drawn independently. Returns [-1, 1] tensors."""

    def __init__(self, real_dir: str, simpx_dir: str, size=(128, 256), flip_real: bool = False,
                 simpx_cases=None):
                                                                                    
                                                                         
        self.real, self.sim = _files(real_dir), _files(simpx_dir, (".npy",), simpx_cases)
        self.size, self.flip_real = size, flip_real

    def __len__(self):
        return max(len(self.real), len(self.sim))

    def __getitem__(self, i):
        x = load_px(self.real[i % len(self.real)], self.size, self.flip_real)
        y = load_px(random.choice(self.sim), self.size)
        to = lambda a: torch.from_numpy(a)[None] * 2.0 - 1.0              
        return {"x": to(x), "y": to(y)}


class SegPairs(Dataset):
    """(image, teeth mask) pairs with matching file names in two folders. Returns [0, 1] tensors."""

    def __init__(self, image_dir: str, mask_dir: str, size=(128, 256), augment: bool = True):
        imgs = _files(image_dir)
        masks = {os.path.splitext(os.path.basename(m))[0]: m for m in _files(mask_dir)}
        self.pairs = [(p, masks[os.path.splitext(os.path.basename(p))[0]]) for p in imgs
                      if os.path.splitext(os.path.basename(p))[0] in masks]
        if not self.pairs:
            raise ValueError("no image/mask pairs with matching names")
        self.size, self.augment = size, augment

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        ip, mp = self.pairs[i]
        img = load_px(ip, self.size)
        m = (load_px(mp, self.size) > 0.5).astype(np.float32)
        if self.augment and random.random() < 0.5:                                 
            img, m = img[:, ::-1].copy(), m[:, ::-1].copy()
        return {"image": torch.from_numpy(img)[None], "mask": torch.from_numpy(m)[None]}
