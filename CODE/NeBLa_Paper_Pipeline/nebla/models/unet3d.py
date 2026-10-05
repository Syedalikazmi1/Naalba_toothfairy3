"""3D UNet refinement network, following wolny/pytorch-3dunet ``UNet3D``.

The NeBLa repository names pytorch-3dunet for this component; the paper gives
four levels with feature widths 64, 128, 256, 512 (Table 6). This module
reproduces pytorch-3dunet's default design so it runs without that package:

    * ``DoubleConv`` with layer order "gcr" (GroupNorm -> Conv3d -> ReLU), twice;
      in the encoder the first conv outputs max(out // 2, in) channels;
      GroupNorm uses 8 groups (1 group when a layer has fewer than 8 channels);
      convolutions have no bias (GroupNorm follows).
    * encoder level 1 has no pooling; levels 2-4 start with MaxPool3d(2).
      The 4th level (512) is the bottom of the U - there is no extra bottleneck.
    * decoder: nearest-neighbour upsampling to the skip's size, concatenation
      [skip, x], DoubleConv -> f_maps[i]  (3 decoder levels).
    * final 1x1x1 convolution, then sigmoid, so the output is a density in
      [0, 1] like the normalised CBCT target.

``use_checkpoint=True`` recomputes each block in the backward pass, which cuts
activation memory several-fold at the full 128 x 256 x 256 size.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

__all__ = ["DoubleConv3d", "UNet3D"]


def _gn(channels: int, num_groups: int) -> nn.GroupNorm:
    g = num_groups if channels >= num_groups else 1
    if channels % g:
        raise ValueError(f"{channels} channels not divisible into {g} groups")
    return nn.GroupNorm(g, channels)


class DoubleConv3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, encoder: bool, num_groups: int = 8):
        super().__init__()
        if encoder:
            mid = max(out_channels // 2, in_channels)
        else:
            mid = out_channels
        self.block = nn.Sequential(
            _gn(in_channels, num_groups),
            nn.Conv3d(in_channels, mid, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            _gn(mid, num_groups),
            nn.Conv3d(mid, out_channels, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet3D(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1,
                 f_maps=(64, 128, 256, 512), num_groups: int = 8,
                 final_sigmoid: bool = True, use_checkpoint: bool = False):
        super().__init__()
        f_maps = tuple(int(f) for f in f_maps)
        self.use_checkpoint = use_checkpoint
        self.pool = nn.MaxPool3d(2)
        self.encoders = nn.ModuleList()
        prev = in_channels
        for f in f_maps:
            self.encoders.append(DoubleConv3d(prev, f, encoder=True, num_groups=num_groups))
            prev = f
        self.decoders = nn.ModuleList()
        rev = list(reversed(f_maps))
        for i in range(len(rev) - 1):
            self.decoders.append(DoubleConv3d(rev[i] + rev[i + 1], rev[i + 1], encoder=False,
                                              num_groups=num_groups))
        self.final_conv = nn.Conv3d(f_maps[0], out_channels, 1)
        self.final_sigmoid = final_sigmoid

    def _run(self, block, x):
        if self.use_checkpoint and self.training and x.requires_grad:
            return checkpoint(block, x, use_reentrant=False)
        return block(x)

    def forward(self, x):
        skips = []
        for i, enc in enumerate(self.encoders):
            if i > 0:
                x = self.pool(x)
            x = self._run(enc, x)
            skips.append(x)
        skips = skips[:-1][::-1]                                                 
        for dec, skip in zip(self.decoders, skips):
            x = F.interpolate(x, size=skip.shape[2:], mode="nearest")
            x = self._run(dec, torch.cat([skip, x], dim=1))
                                                                                
                                                            
        with torch.autocast(x.device.type, enabled=False):
            x = self.final_conv(x.float())
            return torch.sigmoid(x) if self.final_sigmoid else x
