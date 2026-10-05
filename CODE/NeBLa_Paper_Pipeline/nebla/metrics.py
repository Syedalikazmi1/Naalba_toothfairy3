

from __future__ import annotations

import warnings

import numpy as np
import torch
import torch.nn.functional as F

__all__ = ["psnr", "dice", "ssim", "MetricBundle", "evaluate_volume"]

DEFAULT_THRESHOLD = 0.2


def psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> float:
    mse = F.mse_loss(pred.float(), target.float()).item()
    if mse <= 0:
        return float("inf")
    return float(10.0 * np.log10(data_range**2 / mse))


def dice(pred: torch.Tensor, target: torch.Tensor, threshold: float = DEFAULT_THRESHOLD) -> float:
    p = (pred.float() > threshold).float()
    t = (target.float() > threshold).float()
    denom = p.sum() + t.sum()
    if denom.item() == 0:
        return 1.0
    return float((2.0 * (p * t).sum() / denom).item())


def _gaussian_window(size: int, sigma: float, device, dtype) -> torch.Tensor:
    coords = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2.0
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    return (g[:, None] @ g[None, :]).view(1, 1, size, size)


def ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
) -> float:
    """SSIM for ``(B, 1, H, W)`` tensors."""
    pred, target = pred.float(), target.float()
    window = _gaussian_window(window_size, sigma, pred.device, pred.dtype)
    pad = window_size // 2

    mu1 = F.conv2d(pred, window, padding=pad)
    mu2 = F.conv2d(target, window, padding=pad)
    mu1_sq, mu2_sq, mu1_mu2 = mu1**2, mu2**2, mu1 * mu2
    sigma1 = F.conv2d(pred * pred, window, padding=pad) - mu1_sq
    sigma2 = F.conv2d(target * target, window, padding=pad) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=pad) - mu1_mu2

    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    smap = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1 + sigma2 + c2)
    )
    return float(smap.mean().item())


class MetricBundle:
    """Lazily-constructed LPIPS backbones plus the closed-form metrics."""

    def __init__(self, lpips_nets=("alex", "vgg", "squeeze"), device: str | torch.device = "cpu"):
        self.device = torch.device(device)
        self.lpips_nets = tuple(lpips_nets)
        self._lpips: dict[str, object] = {}
        self._warned = False

    def _get_lpips(self, net: str):
        if net in self._lpips:
            return self._lpips[net]
        try:
            import lpips as lpips_lib

            model = lpips_lib.LPIPS(net=net, verbose=False).to(self.device).eval()
        except Exception as exc:                
            if not self._warned:
                warnings.warn(
                    f"LPIPS unavailable ({exc}); install `lpips` and allow weight "
                    "downloads to reproduce Tables 1 and 2.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._warned = True
            model = None
        self._lpips[net] = model
        return model

    def lpips(self, pred: torch.Tensor, target: torch.Tensor, net: str = "vgg") -> float | None:
        model = self._get_lpips(net)
        if model is None:
            return None
        p = pred.clamp(0, 1).repeat(1, 3, 1, 1) * 2 - 1
        t = target.clamp(0, 1).repeat(1, 3, 1, 1) * 2 - 1
        with torch.no_grad():
            return float(model(p.to(self.device), t.to(self.device)).mean().item())


def _mips(volume: torch.Tensor) -> list[torch.Tensor]:
    return [volume.amax(dim=2), volume.amax(dim=3), volume.amax(dim=4)]


def evaluate_volume(
    pred: torch.Tensor,
    target: torch.Tensor,
    bundle: MetricBundle | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    lpips_nets=("vgg",),
) -> dict:
    """Score one ``(B, 1, D, H, W)`` prediction against its ground truth.

    Returns PSNR (dB), SSIM (%), Dice (%) and one LPIPS value per requested
    backbone, matching the columns of Table 1.
    """
    if pred.dim() != 5 or target.dim() != 5:
        raise ValueError("expected (B, 1, D, H, W) tensors")
    if pred.shape != target.shape:
        raise ValueError(f"prediction {tuple(pred.shape)} and target {tuple(target.shape)} differ")
    pred, target = pred.float().cpu(), target.float().cpu()

    out = {
        "psnr": psnr(pred, target),
        "dice": 100.0 * dice(pred, target, threshold),
    }
    mips_p, mips_t = _mips(pred), _mips(target)
    out["ssim"] = 100.0 * float(np.mean([ssim(p, t) for p, t in zip(mips_p, mips_t)]))

    if bundle is not None:
        for net in lpips_nets:
            vals = [bundle.lpips(p, t, net=net) for p, t in zip(mips_p, mips_t)]
            vals = [v for v in vals if v is not None]
            if vals:
                out[f"lpips_{net}"] = float(np.mean(vals))
    return out
