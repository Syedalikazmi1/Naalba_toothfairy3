

from __future__ import annotations

import argparse
import itertools
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from nebla.translation import ImagePool, PatchDiscriminator, TeethSegmenter, UNetGenerator, init_weights
from nebla.translation.data import UnpairedPX


def gan_loss(pred, real: bool):
    return F.mse_loss(pred, torch.ones_like(pred) if real else torch.zeros_like(pred))


def set_requires_grad(nets, flag: bool):
    for n in nets:
        for p in n.parameters():
            p.requires_grad_(flag)


def to01(x):
    return (x + 1.0) * 0.5


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real", required=True, help="folder of real PX images")
    ap.add_argument("--simpx", required=True, help="folder of SimPX .npy files")
    ap.add_argument("--seg", required=True, help="checkpoint from train_teeth_seg.py")
    ap.add_argument("--out", default="runs/translation")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lambda-seg", type=float, default=10.0)
    ap.add_argument("--lambda-cyc", type=float, default=10.0)
    ap.add_argument("--lambda-id", type=float, default=0.5)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--flip-real", action="store_true")
    ap.add_argument("--simpx-cases", nargs="*", default=None,
                    help="use only these SimPX (the generation module's training cases)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = torch.device(args.device)

    dl = DataLoader(UnpairedPX(args.real, args.simpx, flip_real=args.flip_real,
                               simpx_cases=args.simpx_cases or None),
                    batch_size=args.batch, shuffle=True, drop_last=True)
    G, H = init_weights(UNetGenerator()).to(dev), init_weights(UNetGenerator()).to(dev)                 
    DX, DY = init_weights(PatchDiscriminator()).to(dev), init_weights(PatchDiscriminator()).to(dev)
    S = TeethSegmenter().to(dev)
    S.load_state_dict(torch.load(args.seg, map_location=dev, weights_only=False)["model"])
    S.eval()
    for p in S.parameters():
        p.requires_grad_(False)

    opt_g = torch.optim.Adam(itertools.chain(G.parameters(), H.parameters()), lr=args.lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(itertools.chain(DX.parameters(), DY.parameters()), lr=args.lr, betas=(0.5, 0.999))
    half = args.epochs // 2
    rule = lambda e: 1.0 if e < half else max(0.0, 1.0 - (e - half) / max(args.epochs - half, 1))              
    sch_g = torch.optim.lr_scheduler.LambdaLR(opt_g, rule)
    sch_d = torch.optim.lr_scheduler.LambdaLR(opt_d, rule)
    pool_x, pool_y = ImagePool(50), ImagePool(50)

    for ep in range(args.epochs):
        G.train(); H.train(); DX.train(); DY.train()
        tot = {}
        for b in dl:
            x, y = b["x"].to(dev), b["y"].to(dev)
                                                                                      
            set_requires_grad((DX, DY), False)
            fake_y, fake_x = G(x), H(y)
            rec_x, rec_y = H(fake_y), G(fake_x)
            l_gan = gan_loss(DY(fake_y), True) + gan_loss(DX(fake_x), True)
            l_cyc = F.l1_loss(rec_x, x) + F.l1_loss(rec_y, y)
            l_id = F.l1_loss(G(y), y) + F.l1_loss(H(x), x)
            with torch.no_grad():
                seg_x = S(to01(x))                                                                
            l_seg = F.mse_loss(S(to01(fake_y)), seg_x)                                                
            loss_g = (l_gan + args.lambda_cyc * l_cyc + args.lambda_id * args.lambda_cyc * l_id
                      + args.lambda_seg * l_seg)
            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            opt_g.step()
            set_requires_grad((DX, DY), True)
                                 
            fy, fx = pool_y.query(fake_y), pool_x.query(fake_x)
            loss_d = 0.5 * (gan_loss(DY(y), True) + gan_loss(DY(fy), False)) \
                + 0.5 * (gan_loss(DX(x), True) + gan_loss(DX(fx), False))
            opt_d.zero_grad(set_to_none=True)
            loss_d.backward()
            opt_d.step()
            for k, v in (("G", loss_g), ("D", loss_d), ("cyc", l_cyc), ("seg", l_seg), ("gan", l_gan)):
                tot[k] = tot.get(k, 0.0) + float(v)
        sch_g.step(); sch_d.step()
        n = max(len(dl), 1)
        print(f"[trans {ep:3d}] " + "  ".join(f"{k} {v / n:.4f}" for k, v in tot.items()))
        torch.save({"G": G.state_dict(), "H": H.state_dict(), "DX": DX.state_dict(), "DY": DY.state_dict(),
                    "epoch": ep, "flip_real": args.flip_real}, os.path.join(args.out, "last.pt"))


if __name__ == "__main__":
    main()
