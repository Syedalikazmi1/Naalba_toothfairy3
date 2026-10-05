

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch

from nebla.data import NeBLaDataset, check_split, read_meta
from nebla.losses import GenerationLoss, GenerationLossConfig
from nebla.metrics import evaluate_volume
from nebla.models import GenerationConfig, GenerationModule


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, crit, ds, device, amp, optimizer=None):
    train = optimizer is not None
    model.train(train)
    totals, n = {}, 0
    order = list(range(len(ds)))
    if train:
        random.shuffle(order)
    for i in order:
        b = ds[i]
        simpx = b["simpx"].unsqueeze(0).to(device)
        target = b["volume"].unsqueeze(0).to(device)
        with torch.set_grad_enabled(train):
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                sigma = model(simpx)
            out = crit(sigma.float(), target)
            if not math.isfinite(float(out["loss"])):
                raise FloatingPointError(f"non-finite loss on {b['case']} (weights not updated)")
            if train:
                optimizer.zero_grad(set_to_none=True)
                out["loss"].backward()
                optimizer.step()
        for k, v in out.items():
            totals[k] = totals.get(k, 0.0) + float(v)
        if not train:
            for k, v in evaluate_volume(sigma.float(), target).items():
                totals["metric_" + k] = totals.get("metric_" + k, 0.0) + float(v)
        n += 1
    return {k: v / max(n, 1) for k, v in totals.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--train-cases", nargs="+", required=True)
    ap.add_argument("--val-cases", nargs="+", required=True)
    ap.add_argument("--test-cases", nargs="*", default=[], help="only checked for overlap; never used here")
    ap.add_argument("--out", default="runs/generation")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=30, help="early stopping: epochs without val improvement")
    ap.add_argument("--lambda-proj", type=float, default=10.0)
    ap.add_argument("--lambda-perc", type=float, default=1.0)
    ap.add_argument("--refine-f-maps", type=int, nargs="+", default=[64, 128, 256, 512])
    ap.add_argument("--refine-checkpoint", action="store_true")
    ap.add_argument("--chunk-pixels", type=int, default=2048)
    ap.add_argument("--amp", action="store_true", help="bfloat16 autocast for the networks (loss in fp32)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    seed_all(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)
    meta = read_meta(args.data)
    check_split(meta, args.train_cases, args.val_cases, args.test_cases)
    train_ds = NeBLaDataset(args.data, args.train_cases)
    val_ds = NeBLaDataset(args.data, args.val_cases)

    cfg = GenerationConfig(refine_f_maps=tuple(args.refine_f_maps), refine_checkpoint=args.refine_checkpoint,
                           chunk_pixels=args.chunk_pixels)
    model = GenerationModule(os.path.join(args.data, "voxel_index.npz"), cfg).to(device)
    crit = GenerationLoss(GenerationLossConfig(lambda_proj=args.lambda_proj,
                                               lambda_perc=args.lambda_perc)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    start, best, stale, log = 0, float("inf"), 0, []
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); optimizer.load_state_dict(ck["optimizer"])
        start, best, stale, log = ck["epoch"] + 1, ck["best_val"], ck.get("stale", 0), ck.get("log", [])
        if "rng" in ck:
            random.setstate(ck["rng"]["python"]); np.random.set_state(ck["rng"]["numpy"])
            torch.set_rng_state(ck["rng"]["torch"].cpu())
        if stale >= args.patience:
            raise SystemExit(f"run already stopped early at epoch {ck['epoch']} (patience {args.patience})")
    print(f"[train] {len(train_ds)} train / {len(val_ds)} val cases on {device}; "
          f"params {model.parameter_groups()}")

    settings = {**vars(args), "beta": meta["beta"], "normalize": meta.get("normalize", "none"),
                "norm_range": meta.get("norm_range"), "voxel_mm": meta.get("voxel_mm"),
                "geometry_version": meta["geometry_version"],
                "index_version": meta["index_version"], "sha_index": meta["sha_index"],
                "sha_geometry": meta["sha_geometry"], "config": cfg.__dict__}
    for epoch in range(start, args.epochs):
        t0 = time.time()
        tr = run_epoch(model, crit, train_ds, device, args.amp, optimizer)
        with torch.no_grad():
            va = run_epoch(model, crit, val_ds, device, args.amp)
        row = {"epoch": epoch, "seconds": round(time.time() - t0, 1),
               **{f"train_{k}": v for k, v in tr.items()}, **{f"val_{k}": v for k, v in va.items()}}
        log.append(row)
        improved = va["loss"] < best
        if improved:
            best, stale = va["loss"], 0
        else:
            stale += 1
        ck = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
              "best_val": best, "stale": stale, "log": log, "settings": settings,
              "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                      "torch": torch.get_rng_state()}}
        torch.save(ck, os.path.join(args.out, "last.pt"))
        if improved:
            torch.save(ck, os.path.join(args.out, "best.pt"))
        with open(os.path.join(args.out, "log.json"), "w") as fh:
            json.dump(log, fh, indent=2, default=str)
        print(f"[epoch {epoch:3d}] train {tr['loss']:.5f}  val {va['loss']:.5f}  "
              f"PSNR {va.get('metric_psnr', float('nan')):.2f}  Dice {va.get('metric_dice', float('nan')):.1f}  "
              f"{'*' if improved else ''} ({row['seconds']} s)")
        if stale >= args.patience:
            print(f"[train] early stop: no validation improvement for {args.patience} epochs")
            break
    print(f"[train] best validation loss {best:.5f} -> {os.path.join(args.out, 'best.pt')}")


if __name__ == "__main__":
    main()
