                     
"""Train the teeth segmentation model S used by the translation module's L_seg.

The paper trains a UNet segmentation model on a public PX dataset
(Abdi, Kasaei and Mehdizadeh 2015). Point --images / --masks at such a set
(file names must match; masks are teeth = white).

    python train_teeth_seg.py --images px_images --masks px_teeth_masks --out runs/teeth_seg
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from nebla.translation import TeethSegmenter
from nebla.translation.data import SegPairs


def dice_loss(p, t, eps=1.0):
    inter = (p * t).sum((1, 2, 3))
    return 1 - ((2 * inter + eps) / (p.sum((1, 2, 3)) + t.sum((1, 2, 3)) + eps)).mean()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--out", default="runs/teeth_seg")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = torch.device(args.device)
    ds = SegPairs(args.images, args.masks, augment=True)
    ds_val = SegPairs(args.images, args.masks, augment=False)
    n_val = max(1, int(round(len(ds) * args.val_frac)))
    perm = torch.randperm(len(ds), generator=torch.Generator().manual_seed(0)).tolist()
    tr, va = Subset(ds, perm[n_val:]), Subset(ds_val, perm[:n_val])
    dl = DataLoader(tr, batch_size=args.batch, shuffle=True, drop_last=len(tr) > args.batch)
    dv = DataLoader(va, batch_size=args.batch)
    model = TeethSegmenter().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    best = float("inf")
    for ep in range(args.epochs):
        model.train()
        for b in dl:
            x, m = b["image"].to(dev), b["mask"].to(dev)
            p = model(x)
            loss = F.binary_cross_entropy(p, m) + dice_loss(p, m)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = []
            for b in dv:
                p, m = model(b["image"].to(dev)), b["mask"].to(dev)
                vl.append(float(F.binary_cross_entropy(p, m) + dice_loss(p, m)))
        v = sum(vl) / len(vl)
        if v < best:
            best = v
            torch.save({"model": model.state_dict(), "epoch": ep, "val_loss": v}, os.path.join(args.out, "best.pt"))
        print(f"[seg {ep:3d}] val loss {v:.4f}{' *' if v == best else ''}")


if __name__ == "__main__":
    main()
