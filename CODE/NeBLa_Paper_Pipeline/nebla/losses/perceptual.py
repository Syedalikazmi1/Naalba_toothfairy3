

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["VGGPerceptualLoss"]

_VGG16_SLICES = {"relu1_2": 4, "relu2_2": 9, "relu3_3": 16, "relu4_3": 23}


class VGGPerceptualLoss(nn.Module):
    def __init__(self, layers=("relu1_2", "relu2_2", "relu3_3"), min_side: int = 224,
                 allow_untrained: bool = False):
        super().__init__()
        self.min_side = min_side
        self.blocks = nn.ModuleList()
        try:
            from torchvision.models import VGG16_Weights, vgg16
            features = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features.eval()
            self.is_pretrained = True
        except Exception as exc:                
            if not allow_untrained:
                raise RuntimeError(
                    f"pretrained VGG-16 could not be loaded ({exc}). Install torchvision and allow the "
                    "weight download, or pass allow_untrained=True for a smoke test only.") from exc
            from torchvision.models import vgg16
            features = vgg16(weights=None).features.eval()
            self.is_pretrained = False
        prev = 0
        for name in layers:
            end = _VGG16_SLICES[name]
            self.blocks.append(nn.Sequential(*list(features[prev:end])))
            prev = end
        for p in self.parameters():
            p.requires_grad_(False)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

    def train(self, mode: bool = True):                                                  
        super().train(mode)
        self.blocks.eval()
        return self

    def _prepare(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float().clamp(0.0, 1.0)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        h, w = x.shape[-2:]
        s = self.min_side / min(h, w)
        if s > 1:
            x = F.interpolate(x, size=(round(h * s), round(w * s)), mode="bilinear", align_corners=False)
        return (x - self.mean) / self.std

    def _features(self, x):
        feats = []
        for blk in self.blocks:
            x = blk(x)
            feats.append(x)
        return feats

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = self._features(self._prepare(pred))
        with torch.no_grad():
            t = self._features(self._prepare(target))
        return torch.stack([F.mse_loss(a, b) for a, b in zip(p, t)]).mean()
