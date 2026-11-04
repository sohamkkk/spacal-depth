# Gate 0: literature, baselines and data (25 Oct 2026)

Decision: **go.** No paper found that conformalizes depth completion or monocular depth by splitting one image's sparse measurements into a fitting set and a calibration set. The underlying statistics are known, so the claim must be the depth-sensor design and its evidence, not new theory. The search was web-based and not exhaustive; repeat it on arXiv and Google Scholar before submission `[VERIFY]`.

## Proposed method (dual-anchor)

- Structured *prediction anchors* (any layout: scan lines, clusters, a LiDAR pattern) build the centre and scale: A-dist on a frozen monocular model, or a completion model.
- Independent uniformly random *calibration probes* (not used by the predictor or scale) give a within-image split-conformal threshold.
- Guarantee: conditional on the image and the prediction anchors, coverage over a uniformly random pixel is at least 1 − α (T8 argument), whatever the prediction-anchor layout. No calibration images are needed for validity. It holds in expectation over probe draws, not conditional on them.
- Needs at least ⌈(1 − α)(m + 1)⌉ ≤ m probes, so m ≥ 9 at α = 0.1. Stratified probes give per-stratum guarantees only.

## Closest prior work

| Work | Relation | What separates us |
|---|---|---|
| Gui, Barber & Ma, *Conformalized matrix completion*, NeurIPS 2023 | Splits observed entries into fitting and calibration sets; weighted version for non-uniform sampling | Closest in spirit. Ours: one image at a time, a frozen monocular prior plus a structured fitting set, and a probe budget trade-off. Cite prominently |
| Mao et al., *Valid model-free spatial prediction* (GSCP/LSCP) `[VERIFY venue]` | Random spatial locations ⇒ split conformal valid; local variant | We already use LSCP as a baseline; the dual-role layout is not studied there |
| Marsal et al. 2024/25 `[VERIFY]` | Test-time metric scale from sparse points (RANSAC), no guarantee | We add a coverage guarantee |
| Kandinsky CP (CVPR 2024); CONSIGN `[VERIFY]` | Conditional / within-image conformal for dense outputs | Calibrate across images, not from one image's own probes |
| QUTCC (arXiv 2507.14760) | Pixel-marginal conformal for imaging inverse problems, calibrated across images | Same contrast |
| Conformalized multi-view reconstruction (DTU/ETH3D, scene-level exchangeability) `[VERIFY venue]` | Split conformal on depth residuals with group-wise calibration | Calibrate across scenes; no sparse test-time measurements |
| Taguchi et al., WACV 2023 `[VERIFY]` | Uncertainty-aware interactive LiDAR sampling | Placement for accuracy, not for calibration |
| Any2Full (2026), LDCM (ICLR 2026) `[VERIFY]` | Recent completion models | Possible extra centres; not needed this week |

## Completion baseline: OMNI-DC (chosen)

- ICCV 2025, BSD licence, `princeton-vl/OMNI-DC`, weights `zuoym15/OMNI-DC` on Hugging Face.
- Trained only on synthetic data, evaluated zero-shot on NYU, KITTI, ARKitScenes, ETH3D and DIODE, so no train/test leakage on any of our datasets.
- Takes RGB plus a sparse depth map at full resolution (0 = unknown), so arbitrary layouts work. v1.1 also uses a Depth Anything V2 prior.
- PromptDA rejected as the main baseline: trained on ARKitScenes and ScanNet++ (not zero-shot there); its prompt is a dense 192×256 LiDAR map, not a scattered set; PromptDA++ not released.
- Cost: one forward pass per prediction-anchor set (cannot cache per image).

## Independent-sensor data

**ARKitScenes depth upsampling (usable now, no application).**
- 287 validation videos, 5,598 frames, about 49 MB per video (≈14 GB total).
- Per frame: `wide` RGB 1920×1440, `highres_depth` 1920×1440 (rendered from the Faro laser scan; ground truth), `lowres_depth` 256×192 (iPad LiDAR/ARKit; the sensor), `confidence` 0/1/2. Depths are uint16 millimetres.
- One video (20 frames, 948K matched pixels): iPad vs laser median |log error| 0.017, p90 0.047, mean log −0.013 (iPad reads ≈1.3 % closer), 2.3 % of pixels off by more than 10 %. Confidence 2 (96.8 % of pixels): 1.9 % off by more than 10 %; confidence 0–1: 13–14 %.
- This is the real, independent sensor test KITTI could not give (KITTI raw LiDAR ≈ its GT). Check which frozen monocular models trained on ARKitScenes before calling results zero-shot `[VERIFY]`.
- Download (from a clone of `apple/ARKitScenes`; `--split` is ignored when a CSV is given, so filter the CSV first):
  `awk -F, 'NR==1 || $3=="Validation"' depth_upsampling/upsampling_train_val_splits.csv > val.csv && python3 download_data.py upsampling --video_id_csv val.csv --download_dir <dir>`

**ScanNet++.** Needs an application form with a handwritten PI/supervisor signature; default download 1.5 TB. Not feasible by 31 Oct.

## Kill criteria (unchanged)

- Method gate: at least 10 % narrower than A-dist-both on 4 of 6 model–dataset pairs while keeping per-image validity.
- Completion gate: at least 15 % lower valid interval score than ordinary conformalized completion, or distance-wise coverage spread reduced from over 5 points to under 1.5.

## Deadlines

CVPR 2027: registration 10 Nov 2026 AoE, paper 16 Nov AoE, supplementary 23 Nov.
