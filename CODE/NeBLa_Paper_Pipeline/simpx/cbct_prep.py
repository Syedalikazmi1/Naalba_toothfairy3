"""Bring every CBCT volume into ONE standard pose so the paper's fixed ray layout fits it.

The paper places the 21 rotation centres at fixed positions on the 256 x 256
axial slice for every patient. That only works when every volume shows the jaw
in the same place, as it does for a single-hospital dataset where patients are
positioned the same way. ToothFairy volumes are not positioned consistently
(the arches of cases 002/003/004 sit up to ~60 voxels apart), so each volume is
rotated and shifted in the axial plane once, during preprocessing:

    * the front of the arch faces row 0 (anterior = decreasing row),
    * the arch is centred left/right on column ``col_center`` (128),
    * the front of the incisors sits on row ``incisor_row``.

No scaling: a rigid in-plane transform, the same for every axial slice. The
target volume is transformed with the input, so the network learns in this
standard frame and nothing about the target is passed to the model.

One voxel size. The ray layout is fixed in VOXELS, so a jaw stored at 0.48 mm
is 25% larger on the curve than one stored at 0.6 mm. ``resample_spacing``
brings every case to one voxel size first (true spacing from the v3 sidecar),
WITHOUT cropping; ``align_volume(..., output_shape=...)`` then cuts the final
(128, 256, 256) block around the jaw, with the slices centred on the
dentition. Cropping before the alignment (``harmonise_spacing``, kept for
compatibility) can cut teeth when the jaw is off-centre in the scan: resampled
0.6 -> 0.48 mm, a jaw 70 voxels off-centre lost 23% of its teeth in the centre
crop; resampling first and cropping in the alignment lost none.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import affine_transform, zoom

__all__ = ["estimate_pose", "align_volume", "resample_spacing", "dentition_z_center",
           "dentition_width", "harmonise_spacing", "check_volume", "PAPER_DENTITION_WIDTH_VOX"]

                                                                               
                                                                              
                                                            
DEFAULT_INCISOR_ROW = 56.0
DEFAULT_COL_CENTER = 128.0
                                                                          
                                                                             
PAPER_DENTITION_WIDTH_VOX = 100.0


def check_volume(v: np.ndarray, name: str = "volume") -> dict:
    """Basic sanity checks shared by every stage. Raises on hard errors."""
    v = np.asarray(v)
    if v.ndim != 3:
        raise ValueError(f"{name}: expected (D, H, W), got {v.shape}")
    if not np.isfinite(v).all():
        raise ValueError(f"{name}: contains NaN/inf")
    lo, hi = float(v.min()), float(v.max())
    if lo < -1e-6 or hi > 1 + 1e-6:
        raise ValueError(f"{name}: values must be normalised to [0, 1], got [{lo:.3f}, {hi:.3f}]")
    return {"shape": tuple(v.shape), "min": lo, "max": hi, "mean": float(v.mean()),
            "frac_zero": float((v == 0).mean()), "frac_max": float((v >= hi).mean())}


def estimate_pose(volume: np.ndarray, quantile: float = 0.997, span_deg: float = 30.0,
                  n_angles: int = 241) -> dict:
    """In-plane pose of the dentition from its densest voxels (enamel, restorations).

    1. Keep voxels above the ``quantile`` of the foreground (the teeth) and take
       their axial footprint (all slices together, so both arches count).
    2. Search the in-plane rotation (+-``span_deg`` around the principal axis)
       at which a parabola fits the footprint best: that is the arch's axis.
    3. Anterior = the side of the parabola's vertex; the incisor point = the
       98th percentile of the anterior coordinate within the central 20% band.

    Uses no information that is not also aligned into the target volume.
    """
    v = np.asarray(volume, np.float32)
    fg = v[v > 0.05]
    if fg.size < 1000:
        raise ValueError("volume has almost no foreground")
    thr = float(np.quantile(fg, quantile))
    rows, cols = np.nonzero((v >= thr).any(0))
    if rows.size < 200:
        raise ValueError(f"only {rows.size} dense footprint pixels; cannot find the teeth")
    pts = np.stack([rows, cols], 1).astype(np.float64)
    cen = pts.mean(0)
    x = pts - cen
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    base = np.degrees(np.arctan2(vt[0][0], vt[0][1]))
    best = None
    for a in base + np.linspace(-span_deg, span_deg, n_angles):
        r = np.radians(a)
        u_s = np.array([np.sin(r), np.cos(r)])
        u_t = np.array([np.cos(r), -np.sin(r)])
        s, t = x @ u_s, x @ u_t
        co = np.polyfit(s, t, 2)
        res = float(np.mean((t - np.polyval(co, s)) ** 2))
        if best is None or res < best[0]:
            best = (res, co, u_s, u_t)
    res, co, u_s, u_t = best
    u_ap = -u_t if co[0] > 0 else u_t
    u_lr = u_s.copy()
    if u_lr @ np.array([u_ap[1], -u_ap[0]]) < 0:                                  
        u_lr = -u_lr
    s, t = x @ u_lr, x @ u_ap
    s0 = float(-co[1] / (2 * co[0])) * (1.0 if np.allclose(u_lr, u_s) else -1.0)
    width = float(s.max() - s.min())
    width_robust = float(np.percentile(s, 99.5) - np.percentile(s, 0.5))
    band = np.abs(s - s0) < 0.1 * width
    front = float(np.quantile(t[band], 0.98)) if band.sum() > 10 else float(t.max())
    vertex = cen + u_lr * s0 + u_ap * front
    return {"u_ap": u_ap, "u_lr": u_lr, "vertex": vertex, "threshold": thr,
            "n_footprint": int(rows.size), "fit_residual": res, "width": width,
            "width_robust": width_robust,
            "angle_deg": float(np.degrees(np.arctan2(u_ap[1], -u_ap[0])))}


def dentition_z_center(volume: np.ndarray, quantile: float = 0.997) -> float:
    """Mean slice index of the densest voxels (enamel, restorations)."""
    v = np.asarray(volume, np.float32)
    fg = v[v > 0.05]
    if fg.size < 1000:
        raise ValueError("volume has almost no foreground")
    z = np.nonzero(v >= float(np.quantile(fg, quantile)))[0]
    return float(z.mean())


def dentition_width(volume: np.ndarray) -> float:
    """Left-right width of the dense dental footprint, in voxels (0.5-99.5th percentile)."""
    return float(estimate_pose(volume)["width_robust"])


def resample_spacing(volume: np.ndarray, spacing_zyx_mm, target_mm: float) -> np.ndarray:
    """Resample to ``target_mm`` isotropic voxels. No crop or pad: the output grows or
    shrinks. ``align_volume(..., output_shape=...)`` cuts the final block afterwards."""
    v = np.asarray(volume, np.float32)
    factors = [float(s) / float(target_mm) for s in spacing_zyx_mm]
    return np.clip(zoom(v, factors, order=1, mode="nearest"), 0.0, 1.0).astype(np.float32)


def _matrix(pose, incisor_row, col_center, z_shift=0.0):
    """3x3 matrix + offset mapping OUTPUT (z, row, col) -> INPUT (z, row, col)."""
    u_ap, u_lr, v = pose["u_ap"], pose["u_lr"], pose["vertex"]
                                                                        
    m = np.array([[1.0, 0.0, 0.0],
                  [0.0, -u_ap[0], u_lr[0]],
                  [0.0, -u_ap[1], u_lr[1]]])
    off = np.array([float(z_shift),
                    v[0] - col_center * u_lr[0] + incisor_row * u_ap[0],
                    v[1] - col_center * u_lr[1] + incisor_row * u_ap[1]])
    return m, off


def align_volume(volume: np.ndarray, pose: dict | None = None,
                 incisor_row: float = DEFAULT_INCISOR_ROW,
                 col_center: float = DEFAULT_COL_CENTER, order: int = 1,
                 output_shape=None, z_center: float | None = None):
    """Rigidly rotate/shift every axial slice into the standard frame.

    ``output_shape`` (default: the input shape) is the block that is cut out.
    When it differs from the input (a resampled volume), the slices are
    centred on the dentition (``z_center``, default ``dentition_z_center``);
    with the input shape, slices are not shifted.

    Returns ``(aligned, info)``; ``info`` holds the pose, the 3x3 matrix and
    offset (so the transform can be inverted or audited) and how much dense
    tissue fell outside the field of view.
    """
    v = np.asarray(volume, np.float32)
    check_volume(v)
    pose = estimate_pose(v) if pose is None else pose
    out_shape = tuple(int(s) for s in (v.shape if output_shape is None else output_shape))
    z_shift = 0.0
    if out_shape != v.shape:
        zc = dentition_z_center(v) if z_center is None else float(z_center)
        z_shift = zc - (out_shape[0] - 1) / 2.0
    m, off = _matrix(pose, incisor_row, col_center, z_shift)
    det = float(np.linalg.det(m[1:, 1:]))
    if det < 0:
        raise ValueError("pose would mirror the volume (det < 0); refusing")
    out = affine_transform(v, m, offset=off, output_shape=out_shape, order=order,
                           mode="constant", cval=0.0).astype(np.float32)
    out = np.clip(out, 0.0, 1.0)
    thr = float(np.quantile(v[v > 0], 0.995)) if (v > 0).any() else 1.0
    lost = 1.0 - float((out >= thr).sum()) / max(float((v >= thr).sum()), 1.0)
                                                                                    
                                                                                    
                                                                                       
                                                                               
    pts = np.argwhere(v >= float(pose["threshold"])).astype(np.float64)
    if len(pts):
        o = (pts - off) @ np.linalg.inv(m).T
        inside = np.all((o >= -0.5) & (o <= np.array(out_shape, np.float64) - 0.5), axis=1)
        cropped = float(1.0 - inside.mean())
    else:
        cropped = 0.0
    info = {"pose": {k: (val.tolist() if isinstance(val, np.ndarray) else val)
                     for k, val in pose.items()},
            "matrix": m.tolist(), "offset": off.tolist(), "det": det, "z_shift": float(z_shift),
            "input_shape": list(v.shape), "output_shape": list(out_shape),
            "incisor_row": float(incisor_row), "col_center": float(col_center),
            "dense_tissue_lost_frac": max(lost, 0.0), "dense_tissue_cropped_frac": cropped}
    return out, info


def harmonise_spacing(volume: np.ndarray, spacing_zyx_mm, target_mm: float,
                      out_shape=(128, 256, 256)) -> np.ndarray:
    """Resample to ``target_mm`` isotropic voxels, then centre-crop / pad to ``out_shape``.

    Kept for compatibility only: the centre crop happens BEFORE the alignment
    and can cut teeth of an off-centre jaw. ``make_dataset.py`` uses
    ``resample_spacing`` + ``align_volume(output_shape=...)`` instead.
    Only call this with the TRUE spacing of the case (sidecar JSON).
    """
    v = np.asarray(volume, np.float32)
    factors = [float(s) / float(target_mm) for s in spacing_zyx_mm]
    r = zoom(v, factors, order=1, mode="nearest")
    out = np.zeros(out_shape, np.float32)
    src, dst = [], []
    for n_in, n_out in zip(r.shape, out_shape):
        if n_in >= n_out:
            a = (n_in - n_out) // 2
            src.append(slice(a, a + n_out)); dst.append(slice(0, n_out))
        else:
            a = (n_out - n_in) // 2
            src.append(slice(0, n_in)); dst.append(slice(a, a + n_in))
    out[tuple(dst)] = r[tuple(src)]
    return np.clip(out, 0.0, 1.0)
