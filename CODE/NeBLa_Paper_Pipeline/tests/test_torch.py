"""PyTorch tests for the generation and translation modules.

Run from the pipeline root (needs torch, torchvision; a GPU only for --full):

    python tests/test_torch.py --data data/nebla_paper                # unit tests, CPU is fine
    python tests/test_torch.py --data data/nebla_paper --full --steps 200
                                                                      # + full-size run and 1-case overfit

The Eq. 7 scatter in PyTorch is compared against the NumPy reference that
tests/test_simpx.py checks against a brute-force loop.
"""

from __future__ import annotations

import os
import sys
import time
import traceback

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [ROOT, os.path.join(ROOT, "simpx")]

from nebla.data import NeBLaDataset, check_split, read_meta              
from nebla.losses import GenerationLoss, GenerationLossConfig, mip_views              
from nebla.models import GenerationConfig, GenerationModule, UNet3D              
from nebla.models.mlp import NeRF              
from nebla.models.point_embedder import get_embedder              
from nebla.translation import ImagePool, PatchDiscriminator, TeethSegmenter, UNetGenerator              
from paper_geometry import build_paper_geometry              
from render import sample_grid              
from voxel_index import aggregate_reference, build_voxel_index              

DATA = os.environ.get("NEBLA_DATA")                                        
FULL = os.environ.get("NEBLA_FULL", "0") == "1"
STEPS = int(os.environ.get("NEBLA_STEPS", "200"))
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

_SMALL_D = 16
_GEOM = build_paper_geometry((_SMALL_D, 256, 256))
_SMALL_IDX = build_voxel_index(sample_grid(_GEOM), (_SMALL_D, 256, 256))
_TINY_IDX = build_voxel_index(sample_grid(build_paper_geometry((4, 256, 256))), (4, 256, 256))
_SMALL_CFG = GenerationConfig(encoder_features=(8, 16, 32, 64), refine_f_maps=(8, 16, 32, 64),
                              chunk_pixels=512)


try:                                                                               
    import pytest
    Skip = pytest.skip.Exception
except ImportError:
    class Skip(Exception):
        pass


def small_model(index=None, **kw):
    cfg = GenerationConfig(**{**_SMALL_CFG.__dict__, **kw})
    return GenerationModule(_SMALL_IDX if index is None else index, cfg).to(DEV)


                                                                                             
def test_positional_encoding():
    embed, ch = get_embedder(7)
    assert ch == 42
    x = torch.rand(10, 3) * 2 - 1
    ref = torch.cat([fn(x * 2.0 ** k) for k in range(7) for fn in (torch.sin, torch.cos)], -1)
    assert torch.allclose(embed(x), ref, atol=1e-6)


def test_mlp_structure_and_range():
    m = NeRF(D=8, W=128, input_ch=42, output_ch=1, skips=[4])
    assert 1 + len(m.pts_linears) == 8
    assert [l.in_features for l in m.pts_linears] == [128, 128, 128, 128, 256, 128, 128]
    out = m(torch.randn(3, 5, 42), torch.randn(3, 1, 128))
    assert out.shape == (3, 5, 1) and bool((out > 0).all() and (out < 1).all()), "sigmoid output"


def test_encoder_shapes():
    m = small_model()
    for D in (16, 128):
        f = m.encoder(torch.rand(1, 1, D, 256, device=DEV))
        assert f.shape == (1, 128, D, 256), f.shape


def test_unet3d_structure_and_checkpoint():
    net = UNet3D(1, 1, f_maps=(64, 128, 256, 512)).to(DEV)
    assert len(net.encoders) == 4 and len(net.decoders) == 3
    first_conv = [m for m in net.encoders[0].modules() if isinstance(m, torch.nn.Conv3d)][0]
    assert first_conv.out_channels == 32, "pytorch-3dunet: encoder conv1 = max(out // 2, in)"
    small = UNet3D(1, 1, f_maps=(8, 16, 32, 64)).to(DEV)
    x = torch.rand(1, 1, 16, 32, 32, device=DEV, requires_grad=True)
    y = small(x)
    assert y.shape == x.shape and bool((y > 0).all() and (y < 1).all())
    g1 = torch.autograd.grad(y.sum(), x)[0]
    small.use_checkpoint = True
    small.train()
    y2 = small(x)
    g2 = torch.autograd.grad(y2.sum(), x)[0]
    assert torch.allclose(y, y2, atol=1e-5) and torch.allclose(g1, g2, atol=1e-5)


                                                                                     
def _all_F(model, feat):
    D, R = feat.shape[2], feat.shape[3]
    pairs = torch.arange(D * R, device=DEV)
    out = []
    with torch.no_grad():                                                            
        for s in range(0, D * R, 512):
            p = pairs[s:s + 512]
            out.append(model._density_chunk(feat[0], p // R, p % R))
    return torch.cat(out, 0).view(D, R, -1).cpu().double().numpy()


def test_aggregate_matches_numpy_reference():
    m = small_model()
    m.eval()
    feat = torch.randn(1, 128, _SMALL_D, 256, device=DEV)
    with torch.no_grad():
        rho = m.aggregate(feat)[0, 0].cpu().double().numpy()
    ref = aggregate_reference(_all_F(m, feat), _SMALL_IDX)
    assert np.abs(rho - ref).max() < 1e-5, np.abs(rho - ref).max()


def test_rho_zero_where_no_ray_and_chunk_invariance():
    m = small_model()
    m.eval()
    feat = torch.randn(1, 128, _SMALL_D, 256, device=DEV)
    with torch.no_grad():
        a = m.aggregate(feat)
        m.cfg.chunk_pixels = 1024
        b = m.aggregate(feat)
    unc = m.count.view(256, 256) == 0
    assert bool(a[0, 0][:, unc].abs().max() == 0) and bool(a[0, 0][:, ~unc].abs().min() > 0)
    assert torch.allclose(a, b, atol=1e-6)


def test_mlp_checkpoint_gives_same_gradients():
    m = small_model(index=_TINY_IDX)                                                                        
    m.train()
    feat = torch.randn(1, 128, 4, 256, device=DEV, requires_grad=True)
    w = torch.randn(1, 1, 4, 256, 256, device=DEV)
    m.cfg.mlp_checkpoint = True
    g1 = torch.autograd.grad((m.aggregate(feat) * w).sum(), feat)[0]
    m.cfg.mlp_checkpoint = False
    g2 = torch.autograd.grad((m.aggregate(feat) * w).sum(), feat)[0]
    assert torch.allclose(g1, g2, rtol=1e-4, atol=1e-6)


def test_forward_backward_reaches_every_network():
    m = small_model()
    m.train()
    simpx = torch.rand(1, 1, _SMALL_D, 256, device=DEV)
    sigma, rho = m(simpx, return_rho=True)
    assert sigma.shape == (1, 1, _SMALL_D, 256, 256) and rho.shape == sigma.shape
    assert bool((sigma > 0).all() and (sigma < 1).all())
    target = torch.rand_like(sigma)
    torch.nn.functional.mse_loss(sigma, target).backward()
    for name, mod in (("encoder", m.encoder), ("mlp", m.mlp), ("refine", m.refine)):
        g = [p.grad.abs().sum().item() for p in mod.parameters() if p.grad is not None]
        assert g and max(g) > 0, f"no gradient in {name}"


def test_real_index_loads():
    if not DATA:
        raise Skip("needs --data")
    idx_path = os.path.join(DATA, "voxel_index.npz")
    m = GenerationModule(idx_path, GenerationConfig(refine_f_maps=(8, 16, 32, 64))).to(DEV)
    assert m.volume_shape == (128, 256, 256) and m.n_rays == 256 and m.n_points == 200
    assert int(m.sample_mask.sum()) == 256 * 200


                                                                                         
def test_loss_weights_and_mips():
    v = torch.rand(1, 1, 8, 16, 24)
    shapes = [tuple(t.shape) for t in mip_views(v)]
    assert shapes == [(1, 1, 16, 24), (1, 1, 8, 24), (1, 1, 8, 16)]
    crit = GenerationLoss(GenerationLossConfig(lambda_perc=0.0))
    p, t = torch.rand(1, 1, 8, 16, 24, requires_grad=True), torch.rand(1, 1, 8, 16, 24)
    o = crit(p, t)
    assert torch.allclose(o["loss"], o["mse"] + 10 * o["proj"], rtol=1e-6)
    o["loss"].backward()
    assert torch.isfinite(p.grad).all()


def test_perceptual_is_pretrained():
    try:
        crit = GenerationLoss(GenerationLossConfig())
    except RuntimeError as e:                                                       
        raise Skip(f"pretrained VGG-16 could not be loaded ({e}); training needs it") from e
    assert crit.perceptual.is_pretrained
    p = torch.rand(1, 1, 16, 32, 32, requires_grad=True)
    o = crit(p, torch.rand(1, 1, 16, 32, 32))
    o["loss"].backward()
    assert torch.isfinite(p.grad).all() and float(o["perc"]) > 0


def test_split_checks():
    if not DATA:
        raise Skip("needs --data")
    meta = read_meta(DATA)
    cases = list(meta["cases"])
    for bad in ((cases[:1], []), (cases[:1], cases[:1]), (cases[:1], ["nope"])):
        try:
            check_split(meta, *bad)
        except ValueError:
            continue
        raise AssertionError(f"split {bad} should be refused")
    ds = NeBLaDataset(DATA, cases[:1])
    b = ds[0]
    assert b["simpx"].shape == (1, 128, 256) and b["volume"].shape == (1, 128, 256, 256)
    assert 0 <= float(b["simpx"].min()) and float(b["simpx"].max()) <= 1


                                                                                         
def test_translation_networks_and_seg_gradient():
    G, D, S = UNetGenerator().to(DEV), PatchDiscriminator().to(DEV), TeethSegmenter().to(DEV)
    x = torch.rand(2, 1, 128, 256, device=DEV) * 2 - 1
    y = G(x)
    assert y.shape == x.shape
    assert not any(isinstance(m, torch.nn.BatchNorm2d) for m in G.modules()), "generators use InstanceNorm"
    assert D(y).dim() == 4 and D(y).shape[1] == 1
    S.eval()
    for p in S.parameters():
        p.requires_grad_(False)
    l_seg = torch.nn.functional.mse_loss(S((y + 1) / 2), S((x + 1) / 2))
    l_seg.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in G.parameters()), "L_seg must train G"
    pool = ImagePool(2)
    for _ in range(4):
        assert pool.query(torch.rand(1, 1, 4, 4)).shape == (1, 1, 4, 4)


def test_checkpoint_roundtrip():
    import tempfile
    sys.path.insert(0, ROOT)
    from infer import load_model
    m = small_model()
    ck = {"model": m.state_dict(), "settings": {"config": m.cfg.__dict__}}
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "ck.pt")
        torch.save(ck, p)
        np.savez(os.path.join(d, "idx.npz"), **_SMALL_IDX)
        m2 = load_model(p, os.path.join(d, "idx.npz"), DEV)
    x = torch.rand(1, 1, _SMALL_D, 256, device=DEV)
    m.eval()
    with torch.no_grad():
        assert torch.allclose(m(x), m2(x), atol=1e-6)


                                                                                             
def test_full_size_and_overfit():
    if not (FULL and DATA):
        raise Skip("needs --full and --data")
    meta = read_meta(DATA)
    case = list(meta["cases"])[0]
    ds = NeBLaDataset(DATA, [case])
    b = ds[0]
    x, t = b["simpx"][None].to(DEV), b["volume"][None].to(DEV)
    m = GenerationModule(os.path.join(DATA, "voxel_index.npz"),
                         GenerationConfig(refine_checkpoint=True)).to(DEV)
    crit = GenerationLoss(GenerationLossConfig()).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=1e-4)
    if DEV.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    first = None
    for it in range(STEPS):
        t0 = time.time()
        m.train()
        o = crit(m(x), t)
        opt.zero_grad(set_to_none=True)
        o["loss"].backward()
        opt.step()
        first = float(o["loss"]) if first is None else first
        if it == 0:
            mem = torch.cuda.max_memory_allocated() / 2 ** 30 if DEV.type == "cuda" else float("nan")
            print(f"      full-size step: {time.time() - t0:.1f} s, peak GPU {mem:.1f} GiB")
        if it % 25 == 0 or it == STEPS - 1:
            with torch.no_grad():
                m.eval()
                mse = float(torch.mean((m(x) - t) ** 2))
            print(f"      step {it:4d} loss {float(o['loss']):.4f}  PSNR {10 * np.log10(1 / mse):.2f} dB")
    assert float(o["loss"]) < 0.7 * first, f"loss did not fall enough on one case ({first:.4f} -> {float(o['loss']):.4f})"


if __name__ == "__main__":
    a = sys.argv[1:]
    if "--data" in a:
        DATA = a[a.index("--data") + 1]
    FULL = "--full" in a
    if "--steps" in a:
        STEPS = int(a[a.index("--steps") + 1])
    print(f"torch {torch.__version__} on {DEV}")
    tests = [(n, f) for n, f in list(globals().items()) if n.startswith("test_") and callable(f)]
    failed = skipped = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Skip as e:
            skipped += 1
            print(f"SKIP  {name} ({e})")
        except Exception:
            failed += 1
            print(f"FAIL  {name}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed - skipped} passed, {failed} failed, {skipped} skipped")
    sys.exit(1 if failed else 0)
