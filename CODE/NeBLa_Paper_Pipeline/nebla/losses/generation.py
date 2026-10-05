

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .perceptual import VGGPerceptualLoss

__all__ = ["GenerationLossConfig", "GenerationLoss", "mip_views"]


def mip_views(volume: torch.Tensor) -> list:
    """(B, 1, D, H, W) -> [axial (B,1,H,W), coronal (B,1,D,W), sagittal (B,1,D,H)]."""
    if volume.dim() != 5:
        raise ValueError(f"expected (B, 1, D, H, W), got {tuple(volume.shape)}")
    return [volume.amax(dim=2), volume.amax(dim=3), volume.amax(dim=4)]


@dataclass
class GenerationLossConfig:
    lambda_proj: float = 10.0
    lambda_perc: float = 1.0
    perceptual_layers: tuple = ("relu1_2", "relu2_2", "relu3_3")
    allow_untrained_perceptual: bool = False                                                


class GenerationLoss(nn.Module):
    def __init__(self, cfg: GenerationLossConfig | None = None):
        super().__init__()
        self.cfg = cfg or GenerationLossConfig()
        self.perceptual = None
        if self.cfg.lambda_perc > 0:
            self.perceptual = VGGPerceptualLoss(layers=self.cfg.perceptual_layers,
                                                allow_untrained=self.cfg.allow_untrained_perceptual)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        if pred.shape != target.shape:
            raise ValueError(f"prediction {tuple(pred.shape)} vs target {tuple(target.shape)}")
        pred, target = pred.float(), target.float()
        l_mse = F.mse_loss(pred, target)
        pm, tm = mip_views(pred), mip_views(target)
        l_proj = sum(F.mse_loss(p, t) for p, t in zip(pm, tm)) / 3.0
        if self.perceptual is not None:
            l_perc = torch.stack([self.perceptual(p, t) for p, t in zip(pm, tm)]).mean()
        else:
            l_perc = torch.zeros((), device=pred.device)
        total = l_mse + self.cfg.lambda_proj * l_proj + self.cfg.lambda_perc * l_perc
        return {"loss": total, "mse": l_mse.detach(), "proj": l_proj.detach(), "perc": l_perc.detach()}
