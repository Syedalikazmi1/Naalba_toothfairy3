

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

try:
    from cbct_prep import estimate_pose
except ImportError:                                       
    from .cbct_prep import estimate_pose

AXIS_MARGIN = 0.8                                                            
MAX_ROTATION_DEG = 45.0                                                            


def _hull_area(pts: np.ndarray) -> float:
    from scipy.spatial import ConvexHull
    if len(pts) < 10:
        return 1.0
    try:
        return float(ConvexHull(pts).volume)                                  
    except Exception:
        return 1.0


def orientation_report(volume: np.ndarray, quantile: float = 0.997) -> dict:
    """Axis and rotation checks for one volume. ``status`` is PASS, WARN or FAIL."""
    v = np.asarray(volume, np.float32)
    fg = v[v > 0.05]
    if fg.size < 1000:
        return {"status": "FAIL", "reasons": ["volume has almost no foreground"]}
    thr = float(np.quantile(fg, quantile))
    scores = []
    for k in range(3):
        vk = np.moveaxis(v, k, 0)
        fp = (vk >= thr).any(0)
        pts = np.argwhere(fp).astype(np.float64)
        fill = float(fp.sum() / max(_hull_area(pts), 1.0))
        try:
            p = estimate_pose(vk, quantile)
            rms_rel = float(np.sqrt(p["fit_residual"]) / max(p["width_robust"], 1.0))
        except ValueError:
            rms_rel = float("inf")
        scores.append({"axis": k, "hull_fill": fill, "rms_rel": rms_rel, "score": fill * rms_rel})
    order = sorted(scores, key=lambda d: d["score"])
    best, second = order[0], order[1]
    ratio = best["score"] / max(second["score"], 1e-12)
    confident = ratio < AXIS_MARGIN
    rot = float(estimate_pose(v, quantile)["angle_deg"])

    reasons, status = [], "PASS"
    if confident and best["axis"] != 0:
        status = "FAIL"
        reasons.append(f"axis 0 is not superior-inferior: the teeth form an arch when projected along "
                       f"axis {best['axis']} (score ratio {ratio:.2f}). The volume is not in the "
                       f"(Z, Y, X) contract -- remake it with cbct_preprocess_v3.py")
    if abs(rot) > MAX_ROTATION_DEG:
        status = "FAIL"
        reasons.append(f"in-plane rotation {rot:+.1f} deg: the front teeth do not face row 0. That is an "
                       f"axis swap or flip, not a rotated patient")
    if status == "PASS" and not confident:
        status = "WARN"
        reasons.append(f"axial axis inconclusive (score ratio {ratio:.2f}; sparse dentition?): check the "
                       f"figure by eye")
    return {"status": status, "reasons": reasons, "axial_axis": int(best["axis"]),
            "axis_ratio": float(ratio), "confident": bool(confident), "rotation_deg": rot,
            "axis_scores": scores, "threshold": thr}


def orientation_figure(path: str, volume: np.ndarray, report: dict, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    v = np.asarray(volume, np.float32)
    views = [(v.max(axis=0), "axial MIP (along axis 0)\nexpect: arch, front teeth at TOP"),
             (v.max(axis=1), "coronal MIP (along axis 1)\nexpect: teeth rows horizontal, upper jaw TOP"),
             (v.max(axis=2), "sagittal MIP (along axis 2)\nexpect: face LEFT, upper jaw TOP")]
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.6))
    for a, (im, t) in zip(ax, views):
        a.imshow(im, cmap="gray", aspect="auto")
        a.set_title(t, fontsize=9)
        a.axis("off")
    col = {"PASS": "green", "WARN": "darkorange", "FAIL": "red"}[report["status"]]
    fig.suptitle(f"{title}: {report['status']}  (arch along axis {report.get('axial_axis')}, "
                 f"ratio {report.get('axis_ratio', float('nan')):.2f}, rotation "
                 f"{report.get('rotation_deg', float('nan')):+.1f} deg)", color=col, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=70)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--volumes", required=True, help="folder of <case>.npy volumes")
    ap.add_argument("--out", default=None, help="folder for per-case figures and report.json")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.volumes, "*.npy")))
    if not files:
        raise SystemExit(f"no .npy volumes in {args.volumes}")
    if args.out:
        os.makedirs(args.out, exist_ok=True)
    reports, n_fail = {}, 0
    print(f"{'case':24s} {'status':6s} {'arch axis':>9s} {'ratio':>6s} {'rotation':>9s}")
    for f in files:
        case = os.path.basename(f)[:-4]
        v = np.load(f, mmap_mode="r")
        r = orientation_report(np.asarray(v))
        reports[case] = {k: val for k, val in r.items() if k != "threshold"}
        n_fail += r["status"] == "FAIL"
        print(f"{case:24s} {r['status']:6s} {r.get('axial_axis', '-'):>9} {r.get('axis_ratio', float('nan')):6.2f} "
              f"{r.get('rotation_deg', float('nan')):+8.1f}")
        for msg in r["reasons"]:
            print(f"    - {msg}")
        if args.out:
            orientation_figure(os.path.join(args.out, case + ".png"), np.asarray(v), r, case)
    if args.out:
        with open(os.path.join(args.out, "report.json"), "w") as fh:
            json.dump(reports, fh, indent=2)
    print(f"\n{len(files) - n_fail}/{len(files)} cases pass the axis contract")
    raise SystemExit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
