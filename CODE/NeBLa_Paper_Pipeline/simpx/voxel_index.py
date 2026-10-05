

from __future__ import annotations

import numpy as np

__all__ = ["build_voxel_index", "aggregate_reference", "INDEX_VERSION"]

INDEX_VERSION = "nebla-paper-index/1"


def build_voxel_index(grid: dict, volume_shape) -> dict:
    D, H, W = volume_shape
    pts = grid["points_rc"]
    mask = grid["inside"].copy()
    rows = np.clip(np.rint(pts[..., 0]), 0, H - 1).astype(np.int64)
    cols = np.clip(np.rint(pts[..., 1]), 0, W - 1).astype(np.int64)
    plane = rows * W + cols
    plane = np.where(mask, plane, 0)
    n_rays, n_pts = plane.shape

    weight = np.zeros((n_rays, n_pts), np.float64)
    count = np.zeros(H * W, np.float64)
    for r in range(n_rays):
        m = mask[r]
        vox, inv, mult = np.unique(plane[r][m], return_inverse=True, return_counts=True)
        weight[r, m] = 1.0 / mult[inv]
        count[vox] += 1.0

    voxel_norm = np.stack([2.0 * rows / max(H - 1, 1) - 1.0,
                           2.0 * cols / max(W - 1, 1) - 1.0], axis=-1)
    depth_norm = 2.0 * np.arange(D) / max(D - 1, 1) - 1.0
    return {
        "version": INDEX_VERSION,
        "volume_shape": np.array([D, H, W], np.int64),
        "n_rays": np.int64(n_rays), "n_points": np.int64(n_pts),
        "plane_index": plane.astype(np.int64),
        "mask": mask,
        "sample_weight": weight.astype(np.float32),
        "count": count.astype(np.float32),
        "voxel_norm": voxel_norm.astype(np.float32),
        "depth_norm": depth_norm.astype(np.float32),
    }


def aggregate_reference(F: np.ndarray, index: dict) -> np.ndarray:
    """NumPy reference of Eq. 7. ``F``: (D, n_rays, n_points) -> rho (D, H, W)."""
    D, H, W = (int(x) for x in index["volume_shape"])
    plane, mask, w = index["plane_index"], index["mask"], index["sample_weight"]
    flat = np.zeros(D * H * W, np.float64)
    vox = np.arange(D)[:, None, None] * (H * W) + plane[None]
    m = np.broadcast_to(mask[None], vox.shape)
    np.add.at(flat, vox[m], (F * w[None])[m])
    cnt = np.maximum(index["count"], 1.0).reshape(1, H, W)
    return flat.reshape(D, H, W) / cnt
