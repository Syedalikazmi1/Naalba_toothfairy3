# NeBLa – paper-faithful pipeline

A rebuild of your NeBLa_Final_Pipeline that follows Park et al., *NeBLa*
(AAAI 2024, arXiv 2304.04027v6) and the official code
(github.com/sihwa-park/nebla). Your original files are not changed.

## What changed, and why

| Stage | Your v10 pipeline | This pipeline (paper) |
| --- | --- | --- |
| CBCT pose | arch measured per case, rays moved to fit it | each CBCT rotated/shifted once into one standard pose (`simpx/cbct_prep.py`) |
| Rotation centres | per-case position and scale | 21 fixed centres c_i = (5i−50, f(5i−50)), same for every case |
| Ray rule | 256 sliding origins, normal-based directions, 92.5° sweep | rays pivot on the fixed centres, turned by θ_i (0.5 / 0.6 / 1.5°); the last ray of c_i passes through c_{i+1} |
| Points per ray | 120 at 2.13 voxels | 200 at 1.28 voxels, one interval for all rays, all inside the slice |
| Pixel value | focal-trough weights (floor 0), display curve | Eq. 4: 1 − exp(−β Σ σ_i δ), every sample weighted equally |
| β | fitted with γ to mean/std | one β for the whole dataset, by search |
| Eq. 7 | samples counted | distinct rays counted (a ray is one member of B(x)); F evaluated at the voxel |
| MLP output | softplus | sigmoid (official code) |
| 3D UNet | own design + extra bottleneck, 32–256 | pytorch-3dunet design, 4 levels 64–512 |
| Validation | fell back to a training case; best by training loss | separate cases required; best by validation loss; early stopping |
| Translation module | missing | CycleGAN + teeth-segmentation loss (λ = 10) |
| Inference | needed the patient's CBCT | `infer.py`, works from SimPX or real PX |

Fixed from the official code: `point_embedder.get_embedder` in the official
repository stops without a `return`; the two missing lines are restored from
nerf-pytorch. `image_encoder.py` uses `DoubleConv` without defining it; the
standard block is added.

## Axis contract (check this first)

The paper's rays lie in **axial** slices: each SimPX row is one axial slice, so the teeth appear side by side. Every input volume must therefore be in the v3 contract: shape (D, H, W) = (Z, Y, X), axis 0 superior → inferior, axis 1 anterior → posterior (row 0 = front), axis 2 patient right → left. Make the volumes with `cbct_preprocess_v3.py`, which reorients by the NIfTI affine and transposes. **Do not resize `nii.get_fdata()` directly**: that array is (X, Y, Z), so axis 0 becomes left–right. Every SimPX row is then a sagittal slice, the alignment rotates that plane by ~90°, the central rays run up–down through the jaw, and the "SimPX" shows the dental arch from above like an occlusal view, with tissue pushed out of the field of view. Nothing later in the pipeline can permute the axes back.

`simpx/check_orientation.py` checks every volume, and `make_dataset.py` runs the same check and stops if any case fails (`--skip-orientation-check` to override):

- **Axial axis.** Only the projection along superior–inferior shows the teeth as a hollow arch (small fill of its convex hull, good parabola fit). FAIL when one axis wins clearly (score ratio < 0.8) and it is not axis 0. Very sparse dentitions (a few implants) form no arch in any projection: WARN, check the figure by eye.
- **In-plane rotation.** FAIL beyond ±45°: the front teeth must face row 0. That size of rotation means swapped or flipped axes, not a rotated patient.
- **Signs** (up/down, left/right) cannot be told reliably from the image and come from the NIfTI affine. The per-case figure shows three labelled projections to confirm them by eye: axial (arch, front teeth at the top), coronal (teeth rows horizontal, upper jaw at the top), sagittal (face to the left, upper jaw at the top).

After alignment, `dataset.json` also records `teeth_columns_frac`: the share of the 256 image columns whose ray crosses the teeth in some slice (100% for a full dentition when the rays pass through the jaw in the axial plane).

## Choices the paper leaves open (documented, not hidden)

- **First ray angle.** Solved so the ray count is exactly 256: φ_start = 5.648°. The ray layout runs from φ_start to 180° − φ_start.
- **End fan at c_20.** The paper's rule (rotate until the ray reaches c_{i+1}) defines θ only for c_0–c_19, but c_20 needs an end fan for the molars and ramus of the other side. It uses 0.5°, like c_0: 82 rays at c_0, 81 at c_20, 128 on each side of the midline. `--theta-end 0.6` gives the literal list (89 vs 74).
- **Column order.** Rays through c_0 image the opposite side of the jaw, so the ray order is reversed: SimPX column 0 shows the volume's low-column side. With your v3 convention (column 0 = patient's right) that is the usual radiograph layout, patient's right on the viewer's left. The v3 left/right convention itself could not be checked without the DICOM headers. `--high-col-side-first` keeps the raw order.
- **SimPX normalisation — default `global`.** In the paper, σ in Eq. 3–4 is the CBCT *gray value* (−1000 to +3000, water ≈ 0, air ≈ −1000): the offset b of μ ≈ aσ + b goes into A = exp(−bl), and AI₀ = 1 "after normalizing the PX images". This pipeline renders from the [0, 1] volume. Because every ray has the same length, the two differ by one constant factor on T (0.62 at β = 0.0075): unnormalised they differ by up to 0.43, and the paper's own formula gives negative pixels where rays cross much air. After any affine normalisation they agree to 10⁻⁶ (test `test_normalisation_removes_sigma_convention`). So `global` (one min-max map for all cases, stored as `norm_range` in `dataset.json`) gives the paper's image whatever convention the authors used, and keeps intensity differences between patients like the single β. `minmax` normalises per image; `none` is Eq. 4 on the [0, 1] volume and is **not** the paper's image. Contrast: std about 0.03 with `none`, about 0.11 with `global`.
- **Curve position.** col = 128 + x, row = 200 − f(x), matching Fig. 6(b). The volumes are aligned so the front of the incisors sits on row 56 and the arch is centred on column 128. Both numbers were measured on Fig. 6(b): there the apex of the curve is 44 voxels behind the incisors and the curve ends are 119 voxels behind them.
- **Sampling start.** The sampling grid starts where each ray enters the slice; the paper anchors it at the source, which differs by less than one interval.
- **β search criterion.** The paper gives none ("hyperparameter search"). After min-max normalisation β's scale cancels: for small β the image tends to τ/τ_max, so β only sets how much the bright end saturates, and a mean target is often out of reach (the build stops and prints the reachable range instead of returning a grid bound). Recommended: build datasets with a few `--beta` values (e.g. 0.0025, 0.005, 0.0075, 0.01, 0.02), train briefly on each, keep the one with the best validation loss. `--real-px-dir` / `--target-mean` still work when the target is reachable. The source of the earlier 0.357 is not confirmed.
- **Voxel size — one for every case (required).** The ray layout is fixed in voxels, so a jaw stored at 0.48 mm is 25% larger on the curve than one at 0.6 mm (your 002/004 vs 003). `make_dataset.py` needs `--meta-dir` (the v3 sidecars, true spacing) and resamples every case to one isotropic size; `--no-harmonise` opts out and is not paper-like. `--target-mm auto` (default) uses the size that puts the median dentition width at ~100 voxels, the width measured on Fig. 6(b) (bright teeth across columns 79–178); pass a number to fix it. Each case's width is stored as `dentition_width_vox`; sparse dentitions read narrower.
- **Resample, then crop in the alignment.** Resampling no longer crops. The alignment cuts the final 128 × 256 × 256 block around the jaw and centres the slices on the teeth. The old order (resample, centre-crop, align) lost 23% of the teeth of a jaw 70 voxels off-centre; the new order lost none. At ~0.56 mm the block is 71 mm tall, so in tall scans part of the mandible's inferior border can fall outside; teeth do not (checked geometrically). Cases whose scan is smaller than the block get empty (black) borders: that is the scan's real extent, and the target volume has zeros there too.
- **Alignment check.** `dense_tissue_cropped_frac` is geometric: the share of the teeth (the pose estimator's densest 0.3% of tissue) that fall outside the block. The build warns above 1%. In sparse jaws a looser definition (densest 0.5%) reaches cortical bone, which the 71 mm block may legitimately cut above or below. Voxel counts above 0.6 (`teeth_kept_frac`) are reported but are not a crop test: interpolation softens thin enamel and cortex.
- **Loss reductions.** All loss terms are means, not sums; the three MIP terms are averaged; the perceptual loss is computed on MIPs. See `nebla/losses/generation.py`.
- **Dice threshold.** 0.2, the paper's threshold for visual results.
- **CycleGAN details the paper doesn't give.** Defaults are used: LSGAN loss, cycle weight 10, identity weight 0.5, InstanceNorm, image pool of 50, linear learning-rate decay.

## Run

```bash
# 0. axis contract: every volume must have axis 0 = superior-inferior (~1.3 s per case)
python simpx/check_orientation.py --volumes data/cbct_v3/volumes --out data/orientation_check

# 1. SimPX dataset (NumPy only, ~5 s per case): one voxel size, global normalisation
python simpx/make_dataset.py --volumes data/cbct_v3/volumes --meta-dir data/cbct_v3/meta \
  --out data/nebla_paper --beta 0.0075
#    beta search: repeat with --beta 0.0025 0.005 0.01 0.02 into separate --out folders,
#    train briefly on each, keep the best validation loss
#    new cases later: add --target-mm <voxel_mm> --norm-range <lo hi> from dataset.json

# 2. tests
python tests/test_simpx.py --data data/nebla_paper --raw data/cbct_v3/volumes
python tests/test_torch.py --data data/nebla_paper                 # CPU or GPU
python tests/test_torch.py --data data/nebla_paper --full --steps 200   # GPU: full size + 1-case overfit

# 3. generation module
python train_generation.py --data data/nebla_paper \
  --train-cases ToothFairy3F_002 ToothFairy3F_003 --val-cases ToothFairy3F_004 \
  --out runs/generation --refine-checkpoint      # add --amp on A100/H100 if memory is short

# 4. reconstruct
python infer.py --ckpt runs/generation/best.pt --data data/nebla_paper \
  --simpx data/nebla_paper/simpx/ToothFairy3F_004.npy \
  --target data/nebla_paper/volumes/ToothFairy3F_004.npy --out preds

# 5. real PX (needs real PX images + a teeth-mask dataset)
python train_teeth_seg.py --images px_images --masks px_masks --out runs/teeth_seg
python train_translation.py --real real_px --simpx data/nebla_paper/simpx \
  --simpx-cases ToothFairy3F_002 ToothFairy3F_003 --seg runs/teeth_seg/best.pt --out runs/translation
python infer.py --ckpt runs/generation/best.pt --data data/nebla_paper \
  --px patient.png --translator runs/translation/last.pt --out preds
```

Run all commands from this folder. Check one real PX against a SimPX by eye
first; if the left/right sides are reversed, add `--flip-real`.

## Before a paper-style run

- **Cases.** The paper uses 90 CBCT scans: 55 train, 4 validation, 31 test. Three cases can only check that training works.
- **One voxel size.** Required: `--meta-dir` with the v3 sidecars. Check every `qc/<case>.png`: the curve should sit behind the incisors as in Fig. 6(b).
- **Alignment.** `make_dataset.py` prints a WARNING (and lists the case in `dataset.json` → `alignment_warnings`) when more than 1% of the teeth fall outside the block, or the dense footprint is far from ~100 voxels wide. Look at those cases' QC images.
- **Same dataset folder for training and inference.** Checkpoints record β, the normalisation map and the voxel size; `infer.py` refuses a dataset built differently.
- **Repetitions.** Table 1 reports mean ± std over 10 runs: train with `--seed 0` … `--seed 9`.
- **Real PX.** Only the "Synth. PX ⇒ CBCT" half of Table 1 can be reproduced with ToothFairy3 alone. "Real PX ⇒ CBCT" needs paired real PX. The translation module needs unpaired real PX, and the teeth segmenter needs teeth masks (check they are teeth, not mandible, masks).

## Memory

At the paper's widths (3D UNet 64–512 on 128 × 256 × 256) one training step needs roughly 35–40 GB with `--refine-checkpoint` in fp32: an 80 GB A100 is fine, a 40 GB A100 needs `--refine-checkpoint --amp`. On smaller GPUs use `--refine-f-maps 32 64 128 256` and report it as a deviation.

The 2D encoder keeps the official BatchNorm. With batch size 1, training normalises each image by its own statistics and validation/inference use running averages. If validation loss is far above training loss on the same case, this is the first thing to check.

## Test status

- **SimPX, alignment, orientation, geometry and Eq. 7 index:** 25/25 NumPy tests pass on synthetic data and on built test datasets (mixed 0.6 / 0.48 mm phantoms; full, sparse and implants-only dentitions). The orientation check was validated on phantoms in all axis orders. Rerun on your cases: `python tests/test_simpx.py --data data/nebla_paper --raw data/cbct_v3/volumes`.
- **PyTorch code:** not run here, because PyTorch cannot be installed in the build environment. It was linted (no undefined names) and reviewed line by line by a second reviewer who traced every tensor shape; no crash or wrong-value bugs were found, and the small issues found were fixed. Run `python tests/test_torch.py --data data/nebla_paper` before training (CPU, a few minutes, ~2.5 GB RAM), then `--full --steps 200` on the GPU.
- **Untested end to end:** the translation module and real-PX inference, because no real PX or teeth masks were available.
