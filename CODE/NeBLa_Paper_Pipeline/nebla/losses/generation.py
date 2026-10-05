"""Generation-module loss, paper Eq. 8-9.

    L = L_MSE + lambda_1 * L_proj + lambda_2 * L_perc,    lambda_1 = 10, lambda_2 = 1

* L_MSE  - voxel-wise squared error between sigma(I, x) and the CBCT value that
           generated I.
* L_proj - MSE between maximum intensity projections along the axial, coronal
           and sagittal axes (mean of the three).
* L_perc - perceptual loss (Johnson et al. 2016) with frozen VGG-16 features,
           computed on the same three MIP images (VGG is 2D).

Assumptions (the paper does not fix them): every term is a MEAN, not the sum
written in Eq. 9. Means keep the three terms on comparable scales; sums would
weight L_MSE (8.4 M voxels) far more than the MIP terms (32-65 k pixels), so
the reduction changes the effective balance of lambda_1 and lambda_2. The
three MIP terms are averaged, not summed, and the perceptual loss is taken on
MIPs. Report these choices with your results.
"""

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
