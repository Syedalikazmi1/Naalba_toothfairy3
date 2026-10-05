"""SimPX ray geometry exactly as described in the NeBLa appendix (Park et al., AAAI 2024).

Step 1 - rotation centres
    f(x) = 0.01 (x + 100)^2  for -100 <= x <= 0
           0.01 (x - 100)^2  for  0 <= x <= 100
    c_i  = (5i - 50, f(5i - 50)),  i = 0..20, placed on the 256 x 256 axial slice.

Step 2 - ray extraction
    Rays are LINES THROUGH THE FIXED CENTRES. Starting from c_i the ray is
    rotated by theta_i (0.5 deg for i = 0, 1, 18, 19; 1.5 deg for i = 10;
    0.6 deg otherwise). The final ray of c_i is the line through c_i and
    c_{i+1}. Then c_{i+1} becomes the new rotation centre.

Step 3 (point sampling) lives in ``render.py``.

Conventions used here
---------------------
Curve coordinates (x, y): x to the patient's left/right, y = f(x) towards the
front (anterior). A ray's direction is the angle ``phi`` of the line in curve
coordinates, measured from +x towards +y. Every ray travels towards +y
(source behind, detector in front), so phi lies in [phi_start, 180 - phi_start].

Slice coordinates (row, col) of the CBCT volume (D, H, W):
    col = col_offset + x          (default 128 -> curve centred left/right)
    row = row_offset - y          (default 200 -> apex (y=100) at row 100, ends (y=25) at row 175)
    anterior = decreasing row, i.e. the front teeth are near row 0.
These offsets match Fig. 6(b) of the paper and are the SAME for every patient.
Volumes are brought into this frame by ``cbct_prep.align_volume``.

Interpretation choices (the paper does not state them; documented, not hidden):
    * phi_start, the direction of the very first ray at c_0, is solved so that
      the total ray count is exactly 256 (the SimPX width in the paper). The
      ray layout is mirror-symmetric, so the last ray at c_20 is 180 - phi_start.
    * Between the previous chord and the current chord a centre emits rays at
      exact multiples of theta_i; a multiple that falls within theta_i / 2 of
      the chord is dropped so the chord ray is not duplicated.
    * The chord ray of c_{i-1} passes through c_i, so c_i does not repeat it.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "PAPER_N_CENTERS",
    "PAPER_N_RAYS",
    "paper_curve",
    "theta_schedule",
    "ray_angles",
    "solve_phi_start",
    "build_paper_geometry",
]

PAPER_N_CENTERS = 21
PAPER_N_RAYS = 256
GEOMETRY_VERSION = "nebla-paper-geometry/2"


def f_curve(x):
    x = np.asarray(x, dtype=np.float64)
    return np.where(x <= 0, 0.01 * (x + 100.0) ** 2, 0.01 * (x - 100.0) ** 2)


def paper_curve(n_centers: int = PAPER_N_CENTERS) -> np.ndarray:
    """(21, 2) array of c_i = (5i - 50, f(5i - 50))."""
    i = np.arange(n_centers, dtype=np.float64)
    x = 5.0 * i - 50.0
    return np.stack([x, f_curve(x)], axis=1)


def theta_schedule(n_centers: int = PAPER_N_CENTERS, theta_end: float = 0.5) -> np.ndarray:
    """theta_i in degrees: 0.5 at i = 0, 1, 18, 19; 1.5 at i = 10; 0.6 otherwise.

    The paper's rule rotates c_i's rays until they reach c_{i+1}, so theta is
    only defined for i = 0..19. The last centre c_20 still needs an end fan
    (it images the molars/ramus of the other side, like c_0's fan), but the
    paper gives no theta for it. ``theta_end`` (default 0.5, the same as c_0)
    is used; this gives 128 rays on each side of the midline (82 at c_0, 81 at
    c_20). The paper's literal list would give 0.6 here (89 vs 74 rays).
    """
    th = np.full(n_centers, 0.6)
    th[[0, 1, n_centers - 3, n_centers - 2]] = 0.5
    th[n_centers // 2] = 1.5
    th[n_centers - 1] = float(theta_end)
    return th


def _chord_angles(centers: np.ndarray) -> np.ndarray:
    """Line direction of c_i -> c_{i+1} as an angle in [0, 180) degrees."""
    d = np.diff(centers, axis=0)
    a = np.degrees(np.arctan2(d[:, 1], d[:, 0]))
    return np.where(a < 0, a + 180.0, a)


def ray_angles(phi_start: float, centers: np.ndarray | None = None,
               theta: np.ndarray | None = None):
    """Angles (deg) and pivot index of every ray, in detector-column order."""
    centers = paper_curve() if centers is None else np.asarray(centers, np.float64)
    theta = theta_schedule(len(centers)) if theta is None else np.asarray(theta, np.float64)
    n = len(centers)
    chords = _chord_angles(centers)
    ends = np.concatenate([chords, [180.0 - phi_start]])

    angles, pivots = [], []
    start = phi_start
    for i in range(n):
        end, th = float(ends[i]), float(theta[i])
        seq = []
        if i == 0:
            seq.append(start)                                                     
        k = 1
        while start + k * th < end - 0.5 * th + 1e-9:
            seq.append(start + k * th)
            k += 1
        seq.append(end)                                                           
        angles += seq
        pivots += [i] * len(seq)
        start = end
    return np.asarray(angles), np.asarray(pivots, dtype=np.int64)


def solve_phi_start(n_rays: int = PAPER_N_RAYS, centers=None, theta=None) -> float:
    """Smallest first-ray angle (deg) that gives exactly ``n_rays`` rays."""
    lo, hi = 0.0, 40.0
    count = lambda p: len(ray_angles(p, centers, theta)[0])
    if count(lo) < n_rays:
        raise ValueError(f"even phi_start = 0 gives only {count(lo)} rays (< {n_rays})")
                                                                            
    grid = np.arange(lo, hi, 0.001)
    for p in grid:
        if count(p) == n_rays:
            return float(round(p, 3))
    raise ValueError(f"no phi_start gives exactly {n_rays} rays")


def build_paper_geometry(volume_shape=(128, 256, 256), n_rays: int = PAPER_N_RAYS,
                         row_offset: float = 200.0, col_offset: float = 128.0,
                         phi_start: float | None = None, theta_end: float = 0.5,
                         low_col_side_first: bool = True) -> dict:
    """The fixed, patient-independent in-plane ray bundle.

    Returns pivots (n_rays, 2) as (row, col), unit directions (n_rays, 2) as
    (drow, dcol) pointing from source to detector, angles, pivot index and the
    21 centres in slice coordinates. The same bundle is used on every axial
    slice, so the SimPX image is (D, n_rays).

    Column order. Rays through c_0 (low column) image the HIGH-column side of
    the jaw. ``low_col_side_first=True`` reverses the ray order so SimPX column
    0 shows the volume's low-column side. With the v3 volume convention
    (column 0 = patient's right) that is the usual radiograph layout: the
    patient's right on the viewer's left. Only the order of the columns
    changes; the rays themselves do not.
    """
    D, H, W = volume_shape
    centers = paper_curve()
    theta = theta_schedule(len(centers), theta_end)
    if phi_start is None:
        phi_start = solve_phi_start(n_rays, centers, theta)
    ang, piv = ray_angles(phi_start, centers, theta)
    if len(ang) != n_rays:
        raise ValueError(f"geometry gives {len(ang)} rays, expected {n_rays}")
    if low_col_side_first:
        ang, piv = ang[::-1].copy(), piv[::-1].copy()

    c_rc = np.stack([row_offset - centers[:, 1], col_offset + centers[:, 0]], axis=1)
    rad = np.radians(ang)
    dirs = np.stack([-np.sin(rad), np.cos(rad)], axis=1)                         
    return {
        "version": GEOMETRY_VERSION,
        "volume_shape": (int(D), int(H), int(W)),
        "n_rays": int(n_rays),
        "phi_start_deg": float(phi_start),
        "row_offset": float(row_offset),
        "col_offset": float(col_offset),
        "centers_rc": c_rc,
        "theta_deg": theta,
        "theta_end_deg": float(theta_end),
        "low_col_side_first": bool(low_col_side_first),
        "angles_deg": ang,
        "pivot_index": piv,
        "pivots_rc": c_rc[piv],
        "directions_rc": dirs,
    }
