                     
"""Reconstruct 3D volumes with a trained generation module.

SimPX input (Synth. PX => CBCT evaluation):
    python infer.py --ckpt runs/generation/best.pt --data data/nebla_paper \
        --simpx data/nebla_paper/simpx/ToothFairy3F_004.npy --out preds \
        --target data/nebla_paper/volumes/ToothFairy3F_004.npy

Real PX input (Real PX => CBCT): the PX is resized to 128 x 256, translated to
SimPX style by the trained translation generator G, then reconstructed.
    python infer.py --ckpt runs/generation/best.pt --data data/nebla_paper \
        --px patient.png --translator runs/translation/last.pt --out preds

The geometry is the fixed paper geometry stored in ``<data>/voxel_index.npz``;
nothing about the patient's CBCT is needed.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os

import numpy as np
import torch

from nebla.metrics import MetricBundle, evaluate_volume
from nebla.models import GenerationConfig, GenerationModule
from nebla.translation import UNetGenerator
from nebla.translation.data import load_px


def load_model(ckpt_path, index_path, device, dataset_json=None):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    if dataset_json and "settings" in ck and "sha_index" in ck["settings"]:
        with open(dataset_json) as fh:
            meta = json.load(fh)
        for k in ("sha_index", "sha_geometry", "beta", "normalize", "norm_range", "voxel_mm"):
            if k not in ck["settings"]:                                                       
                continue
            if ck["settings"].get(k) != meta.get(k, "none" if k == "normalize" else None):
                raise SystemExit(f"checkpoint was trained with a different {k} than {dataset_json}; "
                                 "use the dataset folder the model was trained on")
    if "settings" in ck:
        known = {f.name for f in dataclasses.fields(GenerationConfig)}
        cfg = GenerationConfig(**{k: v for k, v in ck["settings"]["config"].items() if k in known})
    else:
        cfg = GenerationConfig()
    model = GenerationModule(index_path, cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model


def save_mips(path, vol):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for a, im, t in zip(ax, (vol.max(0), vol.max(1), vol.max(2)), ("axial", "coronal", "sagittal")):
        a.imshow(im, cmap="gray", vmin=0, vmax=1, aspect="auto"); a.set_title(t); a.axis("off")
    fig.tight_layout(); fig.savefig(path, dpi=80); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True, help="dataset folder holding voxel_index.npz")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--simpx", nargs="+")
    g.add_argument("--px", nargs="+")
    ap.add_argument("--translator", default=None, help="translation checkpoint (needed with --px)")
    ap.add_argument("--target", nargs="*", default=None, help="ground-truth volumes, same order, for metrics")
    ap.add_argument("--lpips", action="store_true")
    ap.add_argument("--out", default="preds")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)
    model = load_model(args.ckpt, os.path.join(args.data, "voxel_index.npz"), dev,
                       os.path.join(args.data, "dataset.json"))
    D, H, W = model.volume_shape

    inputs = args.simpx or args.px
    if args.target and len(args.target) != len(inputs):
        raise SystemExit("give one --target per input, in the same order")
    G, flip = None, False
    if args.px:
        if not args.translator:
            raise SystemExit("--px needs --translator (real PX must be translated to SimPX style first)")
        tk = torch.load(args.translator, map_location=dev, weights_only=False)
        G = UNetGenerator().to(dev)
        G.load_state_dict(tk["G"])
        G.eval()
        flip = bool(tk.get("flip_real", False))
    bundle = MetricBundle(device=dev) if args.lpips else None
    results = []
    for i, path in enumerate(inputs):
        img = load_px(path, (D, model.n_rays), flip_lr=flip) if args.px else np.load(path).astype(np.float32)
        if img.shape != (D, model.n_rays):
            raise SystemExit(f"{path}: shape {img.shape}, expected {(D, model.n_rays)}")
        x = torch.from_numpy(img)[None, None].to(dev)
        with torch.no_grad():
            if G is not None:
                x = (G(x * 2 - 1) + 1) * 0.5
            vol = model(x)
        name = os.path.splitext(os.path.basename(path))[0]
        v = vol[0, 0].float().cpu().numpy()
        np.save(os.path.join(args.out, name + "_pred.npy"), v)
        save_mips(os.path.join(args.out, name + "_mip.png"), v)
        row = {"input": path}
        if args.target:
            t = torch.from_numpy(np.load(args.target[i]).astype(np.float32))[None, None]
            if tuple(t.shape) != tuple(vol.shape):
                raise SystemExit(f"target {args.target[i]} has shape {tuple(t.shape[2:])}, "
                                 f"prediction {tuple(vol.shape[2:])}; use the ALIGNED volume from make_dataset")
            row.update(evaluate_volume(vol.float().cpu(), t, bundle=bundle,
                                       lpips_nets=("alex", "vgg", "squeeze")))
        results.append(row)
        print(row)
    with open(os.path.join(args.out, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2)


if __name__ == "__main__":
    main()
