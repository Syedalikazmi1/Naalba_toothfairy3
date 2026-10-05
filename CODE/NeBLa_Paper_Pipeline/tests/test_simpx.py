

from __future__ import annotations

import json
import os
import sys
import traceback

import numpy as np
from scipy.ndimage import map_coordinates, rotate

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "simpx"))

from cbct_prep import (align_volume, dentition_width, estimate_pose, harmonise_spacing,              
                       resample_spacing)
from check_orientation import orientation_report              
from paper_geometry import (build_paper_geometry, f_curve, paper_curve,              
                            theta_schedule)
from render import beer_lambert, normalize_simpx, render_tau, sample_grid, search_beta              
from voxel_index import aggregate_reference, build_voxel_index              

DATA = os.environ.get("NEBLA_DATA")
RAW = os.environ.get("NEBLA_RAW")
SHAPE = (128, 256, 256)
_G = build_paper_geometry(SHAPE)
_GRID = sample_grid(_G)
_IDX = build_voxel_index(_GRID, SHAPE)


                                                                                      
def test_curve_matches_appendix():
    c = paper_curve()
    i = np.arange(21)
    assert np.allclose(c[:, 0], 5 * i - 50)
    assert np.allclose(c[:, 1], np.where(c[:, 0] <= 0, 0.01 * (c[:, 0] + 100) ** 2, 0.01 * (c[:, 0] - 100) ** 2))
    assert np.isclose(f_curve(0.0), 100.0) and np.isclose(f_curve(-50.0), 25.0) and np.isclose(f_curve(50.0), 25.0)


def test_theta_schedule():
    th = theta_schedule()
    assert len(th) == 21
    for i in range(21):
        want = 0.5 if i in (0, 1, 18, 19, 20) else 1.5 if i == 10 else 0.6                             
        assert th[i] == want, (i, th[i])


def test_ray_count_order_and_pivots():
    g = _G
    assert g["n_rays"] == 256 and len(g["angles_deg"]) == 256
    a = g["angles_deg"]
    assert np.all(np.diff(a) < 0), "beam must sweep monotonically (column 0 = low-column side)"
    assert np.isclose(a[-1], g["phi_start_deg"]) and np.isclose(a[0], 180 - g["phi_start_deg"])
    assert set(np.unique(g["pivot_index"])) == set(range(21)), "every centre emits rays"
    assert np.all(np.diff(g["pivot_index"]) <= 0), "rays grouped by centre in order"
    assert (a < 90).sum() == (a > 90).sum() == 128, "128 rays on each side of the midline"
                                                       
    assert np.allclose(g["pivots_rc"], g["centers_rc"][g["pivot_index"]])
    assert len(np.unique(g["pivots_rc"], axis=0)) == 21


def test_chord_rule_and_theta_steps():
    g = _G
    c, d, piv, a = g["centers_rc"], g["directions_rc"], g["pivot_index"], g["angles_deg"]
    th = theta_schedule()
    for i in range(20):
        k = np.where(piv == i)[0]
        v = c[i + 1] - c[i]
        v /= np.linalg.norm(v)
        k = k[np.argsort(a[k])]                                                         
        last = d[k[-1]]
        assert abs(v[0] * last[1] - v[1] * last[0]) < 1e-9, f"last ray of c_{i} must hit c_{i+1}"
        steps = np.diff(a[k])
        if len(steps) > 1:                                                             
            assert np.allclose(steps[:-1], th[i]), (i, steps)
            assert steps[-1] <= 1.5 * th[i] + 1e-9 and steps[-1] >= 0.5 * th[i] - 1e-9


def test_column_order_patient_right_first():
    """A tooth on the low-column side lands in the left half of the SimPX, the incisors in the middle."""
    def cols(r, c):
        v = np.zeros((1, 256, 256), np.float32)
        v[0, r - 2:r + 3, c - 2:c + 3] = 1.0
        hit = np.nonzero(render_tau(v, _GRID)[0] > 0)[0]
        return hit.min(), hit.max()
    assert cols(110, 60)[1] < 100, cols(110, 60)
    assert cols(110, 196)[0] > 156, cols(110, 196)
    lo, hi = cols(56, 128)
    assert 120 <= lo and hi <= 136, (lo, hi)


def test_directions_unit_and_anterior():
    d = _G["directions_rc"]
    assert np.allclose(np.linalg.norm(d, axis=1), 1.0)
    assert np.all(-d[:, 0] >= -1e-12), "every beam travels towards the front (decreasing row)"


def test_geometry_is_patient_independent():
    g2 = build_paper_geometry(SHAPE)
    for k in ("pivots_rc", "directions_rc", "angles_deg"):
        assert np.array_equal(g2[k], _G[k])


                                                                                      
def test_sampling_200_uniform_inside():
    gr = _GRID
    assert gr["points_rc"].shape == (256, 200, 2)
    assert np.isclose(gr["step"], 256 / 200)
    assert np.allclose(np.diff(gr["t"], axis=1), gr["step"]), "one shared interval"
    assert gr["inside"].all(), "all 200 samples must lie inside the slice"
                                                                   
    first = gr["points_rc"][:, 0]
    on_edge = (np.isclose(first[:, 0], 0) | np.isclose(first[:, 0], 255)
               | np.isclose(first[:, 1], 0) | np.isclose(first[:, 1], 255))
    assert on_edge.all()
    d = _G["directions_rc"]
    later = gr["points_rc"][:, 1]
    assert np.allclose(later - first, d * gr["step"]), "samples advance from source towards detector"


def test_beer_lambert_constant_volume():
    v = np.full((4, 256, 256), 0.5, np.float32)
    tau = render_tau(v, _GRID)
    assert tau.shape == (4, 256)
    assert np.allclose(tau, 0.5 * 200 * _GRID["step"], rtol=1e-5)
    img = beer_lambert(tau, 0.01)
    assert np.allclose(img, 1 - np.exp(-0.01 * tau), atol=1e-6)


def test_render_matches_independent_interpolation():
    rng = np.random.default_rng(0)
    v = rng.random((3, 256, 256)).astype(np.float32)
    tau = render_tau(v, _GRID)
    pts = _GRID["points_rc"]
    for z in range(3):
        for r in rng.integers(0, 256, 20):
            s = map_coordinates(v[z], [pts[r, :, 0], pts[r, :, 1]], order=1, mode="nearest")
            assert abs(s.sum() * _GRID["step"] - tau[z, r]) < 1e-3, (z, r)


def test_single_voxel_reaches_only_its_rays():
    v = np.zeros((2, 256, 256), np.float32)
    v[1, 60, 128] = 1.0                                                                       
    tau = render_tau(v, _GRID)
    assert tau[0].max() == 0.0, "other slices untouched"
    hit = np.where(tau[1] > 0)[0]
    assert len(hit) > 0
                                                   
    p, d = _G["pivots_rc"][hit], _G["directions_rc"][hit]
    w = np.array([60.0, 128.0]) - p
    dist = np.abs(w[:, 0] * d[:, 1] - w[:, 1] * d[:, 0])
    assert dist.max() < 1.5


def test_beta_search():
    rng = np.random.default_rng(1)
    taus = [rng.uniform(20, 80, (8, 256)) for _ in range(3)]
    r = search_beta(taus, target_mean=0.357)
    assert abs(r["achieved_mean"] - 0.357) < 0.01
    m = r["table"][:, 1]
    assert np.all(np.diff(m) >= -1e-12), "mean intensity rises with beta"


                                                                                         
def test_index_counts_distinct_rays():
    idx = _IDX
    H, W = 256, 256
    ref = np.zeros(H * W)
    for r in range(256):
        ref[np.unique(idx["plane_index"][r][idx["mask"][r]])] += 1
    assert np.array_equal(ref, idx["count"])
                                                                           
    for r in range(0, 256, 17):
        m = idx["mask"][r]
        vox = idx["plane_index"][r][m]
        sums = np.bincount(vox, weights=idx["sample_weight"][r][m], minlength=H * W)
        assert np.allclose(sums[np.unique(vox)], 1.0)


def test_aggregate_reference_vs_brute_force():
    idx = _IDX
    rng = np.random.default_rng(2)
    D = 2
    F = rng.random((D, 256, 200))
    rho = aggregate_reference(F, {**idx, "volume_shape": np.array([D, 256, 256])})
                                                                                         
    acc = {}
    for z in range(D):
        for r in range(256):
            for s in range(200):
                if not idx["mask"][r, s]:
                    continue
                key = (z, int(idx["plane_index"][r, s]))
                acc.setdefault(key, {}).setdefault(r, []).append(F[z, r, s])
    ref = np.zeros((D, 256 * 256))
    for (z, q), rays in acc.items():
        ref[z, q] = np.mean([np.mean(vals) for vals in rays.values()])
    assert np.allclose(rho.reshape(D, -1), ref, atol=1e-10)


def test_aggregate_constant_and_uncovered():
    idx = _IDX
    D = 2
    rho = aggregate_reference(np.full((D, 256, 200), 0.7), {**idx, "volume_shape": np.array([D, 256, 256])})
    cov = idx["count"].reshape(256, 256) > 0
    assert np.allclose(rho[:, cov], 0.7) and np.all(rho[:, ~cov] == 0)


def test_index_coordinates():
    idx = _IDX
    vn = idx["voxel_norm"]
    assert vn.min() >= -1 and vn.max() <= 1
    rows = (vn[..., 0] + 1) / 2 * 255
    cols = (vn[..., 1] + 1) / 2 * 255
    q = np.rint(rows).astype(int) * 256 + np.rint(cols).astype(int)
    assert np.array_equal(q[idx["mask"]], idx["plane_index"][idx["mask"]]), "F is evaluated at the voxel it is averaged into"
    assert np.isclose(idx["depth_norm"][0], -1) and np.isclose(idx["depth_norm"][-1], 1)


                                                                                            
def _phantom_arch(angle_deg=0.0, shift=(0, 0)):
    v = np.zeros((24, 256, 256), np.float32)
    rr, cc = np.mgrid[0:256, 0:256]
    v[:, (rr - 140) ** 2 + (cc - 128) ** 2 < 110 ** 2] = 0.25           
    s = np.linspace(-1, 1, 400)
    for x, y in zip(128 + 80 * s, 50 + 110 * s ** 2):                                               
        v[6:18, int(y) - 4:int(y) + 5, int(x) - 4:int(x) + 5] = 0.95
    if angle_deg:
        v = rotate(v, angle_deg, axes=(1, 2), reshape=False, order=1)
    if shift != (0, 0):
        v = np.roll(v, shift, axis=(1, 2))
    return np.clip(v, 0, 1)


def test_alignment_recovers_standard_pose():
    for ang, sh in ((0.0, (0, 0)), (12.0, (15, -10)), (-8.0, (-12, 18))):
        v = _phantom_arch(ang, sh)
        out, info = align_volume(v, incisor_row=28)
        assert info["det"] > 0, "no mirroring"
        p2 = estimate_pose(out)
        ang_err = np.degrees(np.arccos(np.clip(p2["u_ap"] @ np.array([-1.0, 0.0]), -1, 1)))
        assert ang_err < 3.0, (ang, sh, ang_err)
        assert abs(p2["vertex"][1] - 128) < 3 and abs(p2["vertex"][0] - 28) < 4, (ang, sh, p2["vertex"])


def test_harmonise_spacing_scale():
    v = np.zeros((40, 80, 80), np.float32)
    v[18:22, 30:50, 30:50] = 1.0                                                          
    out = harmonise_spacing(v, (0.6, 0.6, 0.6), 0.48, out_shape=(50, 100, 100))
    w = (out[25] > 0.5).any(0).sum()
    assert abs(w - 25) <= 1, w                                                       


def test_resample_then_align_keeps_offcentre_teeth():
    """Resampling first and cropping in the alignment never cuts an off-centre jaw."""
    v = _phantom_arch(0.0, (0, 40))                                                                      
    r = resample_spacing(v, (0.6, 0.6, 0.6), 0.48)                                         
    assert r.shape == (30, 320, 320)
    teeth = np.argwhere(r > 0.8).astype(np.float64)
    out, info = align_volume(r, incisor_row=28, output_shape=(24, 256, 256))
    assert out.shape == (24, 256, 256)
    m, off = np.array(info["matrix"]), np.array(info["offset"])
    o = (teeth - off) @ np.linalg.inv(m).T
    inside = np.all((o >= 0) & (o <= np.array([23, 255, 255])), axis=1)
    assert inside.mean() > 0.999, inside.mean()
    zc = np.nonzero(out > 0.8)[0].mean()
    assert abs(zc - 11.5) < 1.5, zc                                                   
    h = harmonise_spacing(v, (0.6, 0.6, 0.6), 0.48, out_shape=(24, 256, 256))
    assert (h > 0.8).sum() < 0.9 * (r > 0.8).sum(), "the old centre crop does cut this jaw"


def test_orientation_check_axis_contract():
    """Axis 0 must be superior-inferior: wrong axis orders and swapped in-plane axes fail."""
    v = _phantom_arch()
    r = orientation_report(v)
    assert r["status"] == "PASS" and r["axial_axis"] == 0, r
    for perm in ((1, 0, 2), (2, 1, 0), (1, 2, 0), (2, 0, 1)):                                             
        r = orientation_report(np.ascontiguousarray(np.transpose(v, perm)))
        assert r["status"] == "FAIL", (perm, r["status"], r["axial_axis"], r["axis_ratio"], r["rotation_deg"])
    r = orientation_report(np.ascontiguousarray(np.rot90(v, 1, axes=(1, 2))))                     
    assert r["status"] == "FAIL" and abs(r["rotation_deg"]) > 45, r
    r = orientation_report(_phantom_arch(15.0))                                                             
    assert r["status"] == "PASS", r


def test_dentition_width_in_voxels():
    w = dentition_width(_phantom_arch())                                                        
    assert 150 <= w <= 170, w


def test_normalisation_removes_sigma_convention():
    """Paper Eq. 3-4 uses CBCT gray values (air ~ -1000); the pipeline uses [0, 1]. Same after normalising."""
    rng = np.random.default_rng(0)
    vols = [np.clip(_phantom_arch(a, (0, s)) + rng.normal(0, .01, (24, 256, 256)), 0, 1)[:4].astype(np.float32)
            for a, s in ((0, 0), (8, 10), (-6, -12))]
    b = 0.0075
    pipe = [beer_lambert(render_tau(v, _GRID), b) for v in vols]
    paper = [beer_lambert(render_tau(4000 * v - 1000, _GRID), b / 4000) for v in vols]
    assert max(np.abs(a - c).max() for a, c in zip(pipe, paper)) > 0.1, "unnormalised they differ"
    for mode in ("global", "minmax"):
        pn, _ = normalize_simpx(pipe, mode)
        qn, _ = normalize_simpx(paper, mode)
        assert max(np.abs(a - c).max() for a, c in zip(pn, qn)) < 1e-5, mode


def test_global_norm_range_reuse():
    imgs = [np.linspace(0.2, 0.5, 100).reshape(10, 10), np.linspace(0.3, 0.6, 100).reshape(10, 10)]
    out, (lo, hi) = normalize_simpx(imgs, "global")
    assert np.isclose(lo, 0.2) and np.isclose(hi, 0.6) and np.isclose(out[0].min(), 0) and np.isclose(out[1].max(), 1)
    again, _ = normalize_simpx([imgs[1]], "global", (lo, hi))
    assert np.allclose(again[0], out[1], atol=1e-6), "a stored map reproduces the same image"


                                                                                           
def _need_data():
    if not DATA:
        print("      (skipped: set --data)")
        return False
    return True


def test_dataset_files_and_formula():
    if not _need_data():
        return
    meta = json.load(open(os.path.join(DATA, "dataset.json")))
    assert meta["n_rays"] == 256 and meta["n_points"] == 200 and np.isclose(meta["step_voxels"], 1.28)
    gz = np.load(os.path.join(DATA, "geometry.npz"))
    assert np.array_equal(gz["directions_rc"], _G["directions_rc"]) and np.array_equal(gz["pivots_rc"], _G["pivots_rc"])
    iz = np.load(os.path.join(DATA, "voxel_index.npz"))
    for k in ("plane_index", "mask", "sample_weight", "count"):
        assert np.array_equal(iz[k], _IDX[k]), k
    for case in meta["cases"]:
        v = np.load(os.path.join(DATA, "volumes", case + ".npy"))
        s = np.load(os.path.join(DATA, "simpx", case + ".npy"))
        tau = np.load(os.path.join(DATA, "tau", case + ".npy"))
        assert v.shape == SHAPE and v.dtype == np.float32 and v.min() >= 0 and v.max() <= 1
        assert s.shape == (128, 256) and s.dtype == np.float32 and 0 <= s.min() and s.max() <= 1
        assert np.allclose(render_tau(v, _GRID), tau, rtol=1e-5, atol=1e-3), "tau reproducible from the aligned volume"
        img = beer_lambert(tau, meta["beta"])
        mode = meta.get("normalize", "none")
        if mode == "minmax":
            img = (img - img.min()) / (img.max() - img.min())
        elif mode == "global":
            lo, hi = meta["norm_range"]
            img = np.clip((img - lo) / (hi - lo), 0, 1)
        assert np.allclose(img, s, atol=2e-6), "SimPX = 1 - exp(-beta tau) (+ the recorded normalisation)"


def test_dataset_alignment_on_real_cases():
    if not _need_data():
        return
    meta = json.load(open(os.path.join(DATA, "dataset.json")))
    for case in meta["cases"]:
        v = np.load(os.path.join(DATA, "volumes", case + ".npy"))
        p = estimate_pose(v)
        ang = np.degrees(np.arccos(np.clip(p["u_ap"] @ np.array([-1.0, 0.0]), -1, 1)))
        assert ang < 3.0, (case, ang)
        assert abs(p["vertex"][1] - 128) < 4 and abs(p["vertex"][0] - meta["incisor_row"]) < 4, (case, p["vertex"])
        if RAW:
            raw = np.load(os.path.join(RAW, case + ".npy"))
            al = meta["cases"][case]["align"]
                                                                                          
                                                                                        
                                                                                           
            thr = float(estimate_pose(raw)["threshold"])
            pts = np.argwhere(raw >= thr).astype(np.float64)
            n_raw, n_res = np.array(raw.shape, np.float64), np.array(al["input_shape"], np.float64)
            pts = pts * (n_res - 1) / np.maximum(n_raw - 1, 1)                                    
            o = (pts - np.array(al["offset"])) @ np.linalg.inv(np.array(al["matrix"])).T
            inside = np.all((o >= -0.5) & (o <= np.array(SHAPE) - 0.5), axis=1).mean()
            assert inside > 0.98, f"teeth cropped by alignment ({case}: {inside * 100:.1f}% inside)"
                                                                                       
                                                                                     
                                                                          


                                                                                    
if __name__ == "__main__":
    args = sys.argv[1:]
    if "--data" in args:
        DATA = args[args.index("--data") + 1]
    if "--raw" in args:
        RAW = args[args.index("--raw") + 1]
    tests = [(n, f) for n, f in list(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception:
            failed += 1
            print(f"FAIL  {name}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
