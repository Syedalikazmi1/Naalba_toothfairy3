

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .image_encoder import UNET
from .mlp import NeRF
from .point_embedder import get_embedder
from .unet3d import UNet3D

__all__ = ["GenerationConfig", "GenerationModule", "load_index"]

INDEX_KEYS = ("plane_index", "mask", "sample_weight", "count", "voxel_norm", "depth_norm",
              "volume_shape", "n_rays", "n_points")


@dataclass
class GenerationConfig:
    """Paper Table 6 defaults."""
    encoder_features: tuple = (64, 128, 256, 512)
    encoder_out_dim: int = 128                                                                    
    embed_multires: int = 7                                  
    mlp_depth: int = 8
    mlp_width: int = 128
    mlp_skips: tuple = (4,)
    refine_f_maps: tuple = (64, 128, 256, 512)
    refine_num_groups: int = 8
    refine_checkpoint: bool = False                                                           
    chunk_pixels: int = 2048                                                   
    mlp_checkpoint: bool = True


def load_index(path_or_dict) -> dict:
    """Read ``voxel_index.npz`` (or take the dict from ``build_voxel_index``)."""
    src = np.load(path_or_dict) if isinstance(path_or_dict, (str, os.PathLike)) else path_or_dict
    missing = [k for k in INDEX_KEYS if k not in src]
    if missing:
        raise KeyError(f"voxel index is missing {missing}; rebuild it with simpx/make_dataset.py")
    return {k: np.asarray(src[k]) for k in INDEX_KEYS}


class GenerationModule(nn.Module):
    def __init__(self, index, cfg: GenerationConfig | None = None):
        super().__init__()
        self.cfg = cfg or GenerationConfig()
        if self.cfg.encoder_out_dim != 128:
            raise ValueError("the official MLP's image_linear takes 128 inputs; keep encoder_out_dim = 128")
        idx = load_index(index)
        D, H, W = (int(v) for v in idx["volume_shape"])
        self.volume_shape = (D, H, W)
        self.n_rays = int(idx["n_rays"])
        self.n_points = int(idx["n_points"])

        self.encoder = UNET(in_channels=1, out_channels=self.cfg.encoder_out_dim,
                            features=list(self.cfg.encoder_features))
        self.embed_fn, embed_ch = get_embedder(self.cfg.embed_multires)
        self.mlp = NeRF(D=self.cfg.mlp_depth, W=self.cfg.mlp_width, input_ch=embed_ch,
                        output_ch=1, skips=list(self.cfg.mlp_skips))
        self.refine = UNet3D(1, 1, f_maps=self.cfg.refine_f_maps,
                             num_groups=self.cfg.refine_num_groups,
                             final_sigmoid=True, use_checkpoint=self.cfg.refine_checkpoint)

        R, S = self.n_rays, self.n_points
        t = lambda a, dt: torch.as_tensor(np.ascontiguousarray(a), dtype=dt)              
        self.register_buffer("plane_index", t(idx["plane_index"], torch.long).view(R, S), persistent=False)
        self.register_buffer("sample_mask", t(idx["mask"], torch.bool).view(R, S), persistent=False)
        self.register_buffer("sample_weight", t(idx["sample_weight"], torch.float32).view(R, S), persistent=False)
        self.register_buffer("voxel_norm", t(idx["voxel_norm"], torch.float32).view(R, S, 2), persistent=False)
        self.register_buffer("depth_norm", t(idx["depth_norm"], torch.float32).view(D), persistent=False)
        count = t(idx["count"], torch.float32).view(H * W)
        self.register_buffer("count", count, persistent=False)
        self.register_buffer("count_clamped", count.clamp(min=1.0), persistent=False)

                                                                              
    def _density_chunk(self, feat: torch.Tensor, d_idx: torch.Tensor, r_idx: torch.Tensor) -> torch.Tensor:
        """F for every sample of the pixels (d_idx, r_idx). feat: (C, D, R). Returns (n, S)."""
        n, S = d_idx.numel(), self.n_points
        xy = self.voxel_norm[r_idx]                                                         
        z = self.depth_norm[d_idx].view(n, 1, 1).expand(n, S, 1)                  
        gamma = self.embed_fn(torch.cat([z, xy], dim=-1))                          
        cond = feat[:, d_idx, r_idx].transpose(0, 1).unsqueeze(1)                             
        return self.mlp(gamma, cond).squeeze(-1)                               

                                                                              
    def aggregate(self, feat: torch.Tensor) -> torch.Tensor:
        """feat (B, 128, D, R) -> rho (B, 1, D, H, W)."""
        B, _, D, R = feat.shape
        Dv, H, W = self.volume_shape
        if (D, R) != (Dv, self.n_rays):
            raise ValueError(f"encoder output {(D, R)} does not match the geometry {(Dv, self.n_rays)}")
        dev = feat.device
        pairs = torch.arange(D * R, device=dev)
        d_all, r_all = pairs // R, pairs % R
        chunk = max(int(self.cfg.chunk_pixels), 1)
        out = []
        for b in range(B):
            fb = feat[b]
            parts = []
            for s in range(0, D * R, chunk):
                di, ri = d_all[s:s + chunk], r_all[s:s + chunk]
                if self.cfg.mlp_checkpoint and self.training and torch.is_grad_enabled():
                    dens = checkpoint(self._density_chunk, fb, di, ri, use_reentrant=False)
                else:
                    dens = self._density_chunk(fb, di, ri)
                parts.append(dens.float())
            dens = torch.cat(parts, 0)                                                     
            vox = d_all.view(-1, 1) * (H * W) + self.plane_index[r_all]                   
                                                                                   
                                                                                  
            val = dens * self.sample_weight[r_all]
            flat = torch.zeros(D * H * W, device=dev, dtype=torch.float32)
            flat = flat.index_add(0, vox.reshape(-1), val.reshape(-1))
            rho = flat.view(D, H, W) / self.count_clamped.view(1, H, W)
            out.append(rho)
        return torch.stack(out, 0).unsqueeze(1)

    def forward(self, simpx: torch.Tensor, return_rho: bool = False):
        if simpx.dim() != 4 or simpx.shape[1] != 1:
            raise ValueError(f"expected SimPX (B, 1, D, R), got {tuple(simpx.shape)}")
        feat = self.encoder(simpx)
        rho = self.aggregate(feat)
        sigma = self.refine(rho)
        return (sigma, rho) if return_rho else sigma

    @torch.no_grad()
    def predict(self, simpx: torch.Tensor) -> torch.Tensor:
        was = self.training
        self.eval()
        out = self.forward(simpx)
        self.train(was)
        return out

    def parameter_groups(self) -> dict:
        c = lambda m: sum(p.numel() for p in m.parameters())              
        return {"encoder": c(self.encoder), "mlp": c(self.mlp), "refine": c(self.refine), "total": c(self)}
