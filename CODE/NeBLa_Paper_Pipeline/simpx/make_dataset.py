

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_orientation import orientation_report              
from cbct_prep import (DEFAULT_INCISOR_ROW, PAPER_DENTITION_WIDTH_VOX,              
                       align_volume, check_volume, dentition_width, estimate_pose,
                       resample_spacing)
from paper_geometry import build_paper_geometry              
from render import (NORMALIZE_MODES, beer_lambert, normalize_simpx, render_tau,              
                    sample_grid, search_beta)
from voxel_index import build_voxel_index              


def sha(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def spacing_for(case, meta_dir):
    if not meta_dir:
        return None
    p = os.path.join(meta_dir, case + ".json")
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        sp = json.load(fh).get("output_spacing_zyx_mm")
    return [float(s) for s in sp] if sp else None


def real_px_mean(folder, size, per_image_minmax=False):
    """Average intensity of real PX images in [0, 1], each resized to (rows, cols) = size.
    With ``per_image_minmax`` each image is min-max normalised first (for --normalize minmax)."""
    from PIL import Image
    ext = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    files = sorted(f for f in os.listdir(folder) if f.lower().endswith(ext))
    if not files:
        raise SystemExit(f"no images in {folder}")
    means = []
    for f in files:
        im = Image.open(os.path.join(folder, f))
        a = np.asarray(im, np.float32)
        if im.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
            a = (a - a.min()) / max(float(a.max() - a.min()), 1e-6)
        else:
            a = np.asarray(im.convert("L"), np.float32) / 255.0
        a = np.asarray(Image.fromarray(a).resize((size[1], size[0]), Image.BILINEAR), np.float32)
        if per_image_minmax:
            a = (a - a.min()) / max(float(a.max() - a.min()), 1e-6)
        means.append(float(a.mean()))
    return float(np.mean(means))


def qc_figure(path, vol, geom, grid, img, case):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(16, 5), gridspec_kw={"width_ratios": [1, 2]})
    ax[0].imshow(vol.max(0), cmap="gray")
    c = geom["centers_rc"]
    ax[0].plot(c[:, 1], c[:, 0], "y.", ms=4)
    for k in range(0, geom["n_rays"], 8):
        p = geom["pivots_rc"][k] + geom["directions_rc"][k] * grid["t"][k][[0, -1]][:, None]
        ax[0].plot(p[:, 1], p[:, 0], "c-", lw=0.3)
    ax[0].set_xlim(0, vol.shape[2] - 1); ax[0].set_ylim(vol.shape[1] - 1, 0)
    ax[0].set_title(f"{case}: aligned axial MIP, 21 centres, every 8th ray", fontsize=9)
    ax[1].imshow(img, cmap="gray", vmin=0, vmax=1, aspect="auto")
    ax[1].set_title(f"SimPX {img.shape}  mean {img.mean():.3f}", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=80); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--volumes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cases", nargs="*", default=None)
    ap.add_argument("--n-points", type=int, default=200)
    ap.add_argument("--incisor-row", type=float, default=DEFAULT_INCISOR_ROW)
    ap.add_argument("--beta", type=float, default=None, help="fix beta instead of searching")
    ap.add_argument("--target-mean", type=float, default=0.357,
                    help="beta search target: mean intensity of real PX images. 0.357 is the value "
                         "from your earlier calibration file; its source is unconfirmed")
    ap.add_argument("--real-px-dir", default=None,
                    help="folder of real PX images: their average intensity (resized to the SimPX size) "
                         "becomes the beta search target, replacing --target-mean")
    ap.add_argument("--meta-dir", default=None,
                    help="v3 sidecar JSONs with output_spacing_zyx_mm (required unless --no-harmonise)")
    ap.add_argument("--target-mm", default="auto",
                    help="common isotropic voxel size in mm, or 'auto' (default): the size that puts the "
                         "median dentition width at the paper's ~100 voxels (Fig. 6(b))")
    ap.add_argument("--no-harmonise", action="store_true",
                    help="keep each case's own voxel size (NOT paper-like: the ray layout is in voxels)")
    ap.add_argument("--no-align", action="store_true", help="volumes are already in the standard pose")
    ap.add_argument("--theta-end", type=float, default=0.5,
                    help="angle step of the end fan at c_20 (not given in the paper; 0.5 = same as c_0)")
    ap.add_argument("--high-col-side-first", action="store_true",
                    help="keep the raw ray order (SimPX column 0 = volume's high-column side)")
    ap.add_argument("--normalize", choices=list(NORMALIZE_MODES), default="global",
                    help="SimPX normalisation after Eq. 4. global (default): one min-max map for all "
                         "cases; minmax: per image; none: Eq. 4 on the [0,1] volume, which differs "
                         "from the paper's gray-value formula by a constant factor on T (render.py)")
    ap.add_argument("--norm-range", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                    help="reuse a global map (norm_range from an existing dataset.json) for new cases")
    ap.add_argument("--skip-orientation-check", action="store_true",
                    help="build even if a volume fails the axis-contract check (check_orientation.py)")
    ap.add_argument("--max-cropped", type=float, default=0.01,
                    help="warn when alignment puts more than this fraction of the teeth (densest 0.3%%, "
                         "as in the pose estimate) outside the 128 x 256 x 256 block (geometric check)")
    args = ap.parse_args()

    t_start = time.time()
    cases = args.cases or sorted(f[:-4] for f in os.listdir(args.volumes) if f.endswith(".npy"))
    if not cases:
        raise SystemExit("no .npy volumes found")
    for sub in ("volumes", "tau", "simpx", "qc"):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)

    first = np.load(os.path.join(args.volumes, cases[0] + ".npy"), mmap_mode="r")
    shape = tuple(int(s) for s in first.shape)

                                                                                 
    orient, failed = {}, []
    for case in cases:
        r = orientation_report(np.asarray(np.load(os.path.join(args.volumes, case + ".npy"), mmap_mode="r")))
        orient[case] = {k: val for k, val in r.items() if k != "threshold"}
        if r["status"] != "PASS":
            print(f"  orientation {r['status']} {case}: " + "; ".join(r["reasons"]))
        if r["status"] == "FAIL":
            failed.append(case)
    if failed and not args.skip_orientation_check:
        raise SystemExit(
            f"{len(failed)} case(s) are not in the (Z, Y, X) axis contract: {', '.join(failed)}.\n"
            "Remake their volumes with cbct_preprocess_v3.py (it reorients by the NIfTI affine and "
            "transposes to Z, Y, X) -- do not resize nii.get_fdata() directly. Figures: "
            "python simpx/check_orientation.py --volumes <folder> --out <folder>")
    print(f"orientation: {len(cases) - len(failed)}/{len(cases)} cases pass the axis contract"
          + (f" ({sum(o['status'] == 'WARN' for o in orient.values())} inconclusive: check their figures)"
             if any(o["status"] == "WARN" for o in orient.values()) else ""))

                                                                            
    spacing, target_mm, widths_mm = {}, None, {}
    if not args.no_harmonise:
        if not args.meta_dir:
            raise SystemExit("give --meta-dir (v3 sidecars with the true spacing) so every case gets one "
                             "voxel size, or --no-harmonise to keep each case's own (not paper-like)")
        for case in cases:
            sp = spacing_for(case, args.meta_dir)
            if sp is None:
                raise SystemExit(f"{case}: no output_spacing_zyx_mm in {args.meta_dir}/{case}.json")
            spacing[case] = sp
        if str(args.target_mm).lower() == "auto":
            for case in cases:
                v = np.load(os.path.join(args.volumes, case + ".npy")).astype(np.float32)
                widths_mm[case] = estimate_pose(v)["width_robust"] * float(np.mean(spacing[case][1:]))
            med = float(np.median(list(widths_mm.values())))
            target_mm = round(med / PAPER_DENTITION_WIDTH_VOX, 4)
            print(f"voxel size: auto -> {target_mm} mm (median dentition width {med:.1f} mm over "
                  f"{len(cases)} cases = {PAPER_DENTITION_WIDTH_VOX:.0f} voxels, as in Fig. 6(b))")
        else:
            target_mm = float(args.target_mm)
            print(f"voxel size: {target_mm} mm for every case")
    geom = build_paper_geometry(shape, theta_end=args.theta_end,
                                low_col_side_first=not args.high_col_side_first)
    grid = sample_grid(geom, n_points=args.n_points)
    index = build_voxel_index(grid, shape)
    print(f"geometry: {geom['n_rays']} rays from 21 fixed centres, phi_start {geom['phi_start_deg']:.3f} deg; "
          f"{args.n_points} points at {grid['step']:.3f} voxels; samples inside {grid['inside'].mean() * 100:.1f}%; "
          f"voxels reached {np.mean(index['count'] > 0) * 100:.1f}%")

    taus, records, warnings_list = {}, {}, []
    for case in cases:
        v = np.load(os.path.join(args.volumes, case + ".npy")).astype(np.float32)
        if v.shape != shape:
            raise SystemExit(f"{case}: shape {v.shape} differs from {shape}")
        rec = {"input": check_volume(v, case), "orientation": orient[case]}
        if target_mm is not None:
            v = resample_spacing(v, spacing[case], target_mm)                   
            rec["spacing_zyx_mm"], rec["resampled_to_mm"] = spacing[case], target_mm
            if case in widths_mm:
                rec["dentition_width_mm"] = widths_mm[case]
        if args.no_align and v.shape != shape:
            raise SystemExit(f"{case}: --no-align with resampling gives shape {v.shape}, not {shape}")
        if not args.no_align:
            v_in = v
            v, info = align_volume(v, incisor_row=args.incisor_row, output_shape=shape)
                                                                                     
                                                                       
            info["teeth_kept_frac"] = float((v > 0.6).sum() / max(int((v_in > 0.6).sum()), 1))
            rec["align"] = info
            if info["dense_tissue_cropped_frac"] > args.max_cropped:
                warnings_list.append(case)
                print(f"  WARNING {case}: alignment put {info['dense_tissue_cropped_frac'] * 100:.1f}% of the "
                      f"teeth outside the block -- check qc/{case}.png")
        rec["dentition_width_vox"] = dentition_width(v)
                                                                                
                                                                                 
        teeth = (v >= float(np.quantile(v[v > 0.05], 0.997))).astype(np.float32)
        rec["teeth_columns_frac"] = float((render_tau(teeth, grid).max(axis=0) > 0).mean())
        if not (0.7 * PAPER_DENTITION_WIDTH_VOX <= rec["dentition_width_vox"] <= 1.3 * PAPER_DENTITION_WIDTH_VOX):
            warnings_list.append(case)
            print(f"  WARNING {case}: dense dental footprint {rec['dentition_width_vox']:.0f} voxels wide (paper "
                  f"~{PAPER_DENTITION_WIDTH_VOX:.0f}). Sparse dentitions read narrower; if this case has a "
                  f"full arch, check qc/{case}.png")
        np.save(os.path.join(args.out, "volumes", case + ".npy"), v)
        tau = render_tau(v, grid)
        np.save(os.path.join(args.out, "tau", case + ".npy"), tau.astype(np.float32))
        taus[case] = tau
        rec["aligned"] = check_volume(v, case)
        rec["sha_volume"] = sha(v)
        records[case] = rec
        print(f"  {case}: aligned" + (f" (dense tissue outside FOV {rec['align']['dense_tissue_lost_frac'] * 100:.1f}%)"
                                     if "align" in rec else "") + f", tau [{tau.min():.1f}, {tau.max():.1f}]")

    if args.real_px_dir:
        args.target_mean = real_px_mean(args.real_px_dir, (shape[0], geom["n_rays"]),
                                        per_image_minmax=args.normalize == "minmax")
        print(f"real PX reference: mean intensity {args.target_mean:.4f} from {args.real_px_dir}")
    if args.beta is None:
        bs = search_beta(list(taus.values()), target_mean=args.target_mean,
                         normalize=args.normalize, norm_range=args.norm_range)
        beta = bs["beta"]
        print(f"beta search ({args.normalize} normalisation): beta = {beta:.6f} gives mean "
              f"{bs['achieved_mean']:.4f} (target {args.target_mean})")
        if not bs["reachable"]:
            raise SystemExit(
                f"target mean {args.target_mean:.4f} is out of reach with {args.normalize} normalisation: any "
                f"beta gives a mean in [{bs['mean_range'][0]:.3f}, {bs['mean_range'][1]:.3f}]. After min-max "
                "normalisation beta only sets how much the bright end saturates. Pass --beta (e.g. 0.0075, then "
                "pick by validation loss: the paper's 'hyperparameter search'), or a reachable target.")
    else:
        beta, bs = float(args.beta), None

    names = list(taus)
    normed, norm_range = normalize_simpx([beer_lambert(taus[c], beta) for c in names],
                                         args.normalize, args.norm_range)
    if norm_range[0] is not None:
        print(f"global normalisation: SimPX = (1 - T - {norm_range[0]:.6f}) / {norm_range[1] - norm_range[0]:.6f}")
    for case, img in zip(names, normed):
        np.save(os.path.join(args.out, "simpx", case + ".npy"), img)
        try:
            from PIL import Image
            Image.fromarray((img * 255).round().astype(np.uint8)).save(os.path.join(args.out, "simpx", case + ".png"))
        except Exception:
            pass
        vol = np.load(os.path.join(args.out, "volumes", case + ".npy"), mmap_mode="r")
        qc_figure(os.path.join(args.out, "qc", case + ".png"), np.asarray(vol), geom, grid, img, case)
        records[case]["simpx"] = {"mean": float(img.mean()), "std": float(img.std()),
                                  "min": float(img.min()), "max": float(img.max()), "sha": sha(img)}

    np.savez_compressed(os.path.join(args.out, "geometry.npz"),
                        **{k: np.asarray(v) for k, v in geom.items() if k != "version"},
                        points_rc=grid["points_rc"], inside=grid["inside"], t=grid["t"])
    np.savez_compressed(os.path.join(args.out, "voxel_index.npz"),
                        **{k: np.asarray(v) for k, v in index.items() if k != "version"})
    meta = {
        "pipeline": "NeBLa paper-faithful SimPX", "geometry_version": geom["version"],
        "index_version": index["version"], "volume_shape": list(shape),
        "n_rays": geom["n_rays"], "n_points": grid["n_points"], "step_voxels": grid["step"],
        "phi_start_deg": geom["phi_start_deg"], "row_offset": geom["row_offset"],
        "col_offset": geom["col_offset"], "incisor_row": args.incisor_row,
        "beta": beta, "beta_search": None if bs is None else
        {"target_mean": bs["target_mean"], "achieved_mean": bs["achieved_mean"],
         "mean_range": list(bs["mean_range"])},
        "pixel": "1 - exp(-beta * sum(sigma_i * delta))",
        "normalize": args.normalize, "norm_range": list(norm_range) if norm_range[0] is not None else None,
        "voxel_mm": target_mm, "theta_end_deg": geom["theta_end_deg"],
        "low_col_side_first": geom["low_col_side_first"],
        "alignment_warnings": warnings_list,
        "sha_geometry": sha(geom["directions_rc"]) + sha(geom["pivots_rc"]),
        "sha_index": sha(index["plane_index"]) + sha(index["sample_weight"]),
        "cases": records, "seconds": round(time.time() - t_start, 1),
    }
    with open(os.path.join(args.out, "dataset.json"), "w") as fh:
        json.dump(meta, fh, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(f"written to {args.out} in {meta['seconds']} s")


if __name__ == "__main__":
    main()
