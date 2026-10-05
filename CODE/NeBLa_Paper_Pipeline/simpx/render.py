

from __future__ import annotations

import numpy as np

__all__ = ["sample_grid", "render_tau", "beer_lambert", "normalize_simpx", "search_beta",
           "NORMALIZE_MODES"]

NORMALIZE_MODES = ("global", "minmax", "none")

PAPER_N_POINTS = 200


def _entry_exit(p, d, H, W):
    """Parameters where lines p + t d enter / leave the box [0,H-1] x [0,W-1]."""
    t_lo = np.full(len(p), -np.inf)
    t_hi = np.full(len(p), np.inf)
    for ax, hi in ((0, H - 1.0), (1, W - 1.0)):
        o, di = p[:, ax], d[:, ax]
        par = np.abs(di) < 1e-12
        safe = np.where(par, 1.0, di)
        t1, t2 = (0.0 - o) / safe, (hi - o) / safe
        lo, up = np.minimum(t1, t2), np.maximum(t1, t2)
        inside = (o >= 0) & (o <= hi)
        lo = np.where(par, np.where(inside, -np.inf, np.inf), lo)
        up = np.where(par, np.where(inside, np.inf, -np.inf), up)
        t_lo, t_hi = np.maximum(t_lo, lo), np.minimum(t_hi, up)
    return t_lo, t_hi


def sample_grid(geom: dict, n_points: int = PAPER_N_POINTS, step: float | None = None) -> dict:
    """Sample positions (n_rays, n_points, 2) in (row, col), plus the inside mask."""
    D, H, W = geom["volume_shape"]
    p = np.asarray(geom["pivots_rc"], np.float64)
    d = np.asarray(geom["directions_rc"], np.float64)
    if step is None:
        step = max(H, W) / float(n_points)
    t_in, t_out = _entry_exit(p, d, H, W)
    if not np.all(np.isfinite(t_in)) or np.any(t_out <= t_in):
        raise ValueError("a ray misses the slice: check the geometry offsets")
    t = t_in[:, None] + np.arange(n_points)[None, :] * step
    pts = p[:, None, :] + d[:, None, :] * t[..., None]
    eps = 1e-6
    inside = ((pts[..., 0] >= -eps) & (pts[..., 0] <= H - 1 + eps)
              & (pts[..., 1] >= -eps) & (pts[..., 1] <= W - 1 + eps))
    pts = np.clip(pts, [0.0, 0.0], [H - 1.0, W - 1.0])
    return {"points_rc": pts, "inside": inside, "t": t, "t_enter": t_in, "t_exit": t_out,
            "step": float(step), "n_points": int(n_points),
            "chord_len": t_out - t_in}


def render_tau(volume: np.ndarray, grid: dict) -> np.ndarray:
    """Optical depth sum_i sigma_i * delta for every (slice, ray): shape (D, n_rays).

    Bilinear interpolation inside each axial slice; points outside the slice
    contribute 0 (none should be outside when chords are >= 256 voxels).
    """
    v = np.asarray(volume, np.float32)
    D, H, W = v.shape
    pts = grid["points_rc"].reshape(-1, 2)
    r, c = pts[:, 0], pts[:, 1]
    r0 = np.floor(r).astype(np.int64); c0 = np.floor(c).astype(np.int64)
    r1 = np.minimum(r0 + 1, H - 1); c1 = np.minimum(c0 + 1, W - 1)
    dr = (r - r0).astype(np.float32); dc = (c - c0).astype(np.float32)
    w00, w01, w10, w11 = (1 - dr) * (1 - dc), (1 - dr) * dc, dr * (1 - dc), dr * dc
    s = v[:, r0, c0] * w00 + v[:, r0, c1] * w01 + v[:, r1, c0] * w10 + v[:, r1, c1] * w11
    s *= grid["inside"].reshape(-1)[None, :]
    n_rays, n_pts = grid["inside"].shape
    return (s.reshape(D, n_rays, n_pts).sum(-1) * grid["step"]).astype(np.float64)


def beer_lambert(tau: np.ndarray, beta: float) -> np.ndarray:
    """Eq. 4: pixel = 1 - exp(-beta * tau)."""
    return (1.0 - np.exp(-float(beta) * np.asarray(tau, np.float64))).astype(np.float32)


def normalize_simpx(images, mode: str = "global", norm_range=None):
    """Normalise a list of SimPX images. Returns ``(images, (lo, hi))``.

    ``global``: (x - lo) / (hi - lo) with lo/hi the min/max over ALL images (or
    ``norm_range``, e.g. the values stored in an existing dataset.json, so new
    cases get the same map; values outside are clipped). ``minmax``: per image
    (lo/hi returned as None). ``none``: unchanged.
    """
    imgs = [np.asarray(x, np.float64) for x in images]
    if mode == "none":
        return [x.astype(np.float32) for x in imgs], (None, None)
    if mode == "minmax":
        out = [(x - x.min()) / max(float(x.max() - x.min()), 1e-12) for x in imgs]
        return [x.astype(np.float32) for x in out], (None, None)
    if mode != "global":
        raise ValueError(f"normalize must be one of {NORMALIZE_MODES}, got {mode!r}")
    if norm_range is None:
        lo, hi = min(float(x.min()) for x in imgs), max(float(x.max()) for x in imgs)
    else:
        lo, hi = (float(v) for v in norm_range)
    out = [np.clip((x - lo) / max(hi - lo, 1e-12), 0.0, 1.0) for x in imgs]
    return [x.astype(np.float32) for x in out], (lo, hi)


def search_beta(taus, target_mean: float = 0.357, grid=None, normalize: str = "none",
                norm_range=None) -> dict:
    """Hyperparameter search for ONE beta shared by all cases.

    Criterion: the average SimPX intensity over the cases, AFTER the chosen
    normalisation, equals ``target_mean`` (e.g. the mean of your real PX images).
    Returns the chosen beta and the full search table.
    """
    grid = np.geomspace(1e-3, 1.0, 400) if grid is None else np.asarray(grid)
    table = []
    for b in grid:
        imgs, _ = normalize_simpx([beer_lambert(t, b) for t in taus], normalize, norm_range)
        m = float(np.mean([x.mean() for x in imgs]))
        table.append((float(b), m))
    arr = np.array(table)
    i = int(np.argmin(np.abs(arr[:, 1] - target_mean)))
    lo_m, hi_m = float(arr[:, 1].min()), float(arr[:, 1].max())
                                                                                
                                                                                 
                                                                          
    reachable = lo_m - 1e-3 <= target_mean <= hi_m + 1e-3 and 0 < i < len(arr) - 1
    return {"beta": float(arr[i, 0]), "achieved_mean": float(arr[i, 1]),
            "target_mean": float(target_mean), "reachable": bool(reachable),
            "mean_range": (lo_m, hi_m), "table": arr}
