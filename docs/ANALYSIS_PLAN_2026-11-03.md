# Confirmatory analysis plan (frozen 3 Nov 2026, before any confirmatory run)

Everything below was fixed before the confirmatory runs were submitted. The exploratory runs (E13–E19 as of 3 Nov) chose γ, the probe budget and the designs; no number from them is reported as a confirmatory result. Changes after this date go in the amendment log at the end, with the date and whether any confirmatory output had been seen.

## Fixed settings

- α = 0.1 (target coverage 0.9), β = 0.1, n = 20 random probes per image, γ = 0.5 (A-dual scale shrinkage).
- A-dual per-image threshold: rank r = ⌈(n + 1)(1 − α)⌉ = 19 of the 20 probe scores; nominal coverage r/(n + 1) = 0.905.
- A-dual-rmax arm: r = n = 20 (largest probe score); nominal 20/21 = 0.952.
- Sensor-error envelope: α_s = α_e = 0.05; conformal risk control per confidence class on the two one-sided pixel fractions (Supp. D). Superseded before any envelope run by the 4 Nov amendment (one two-sided CRC over class-adaptive bands; see the amendment log). Any adaptive split must be a rule computed on head-train scenes only and written here before its run.
- Splits: scene/visit-level head/calibration/test split (`experiments.common.split_tables`, fractions 0.3/0.35/0.35). A split is the unit of replication; draws are averaged within a split.

## Designs

| Experiment | Data | Splits × draws | Arms |
|---|---|---|---|
| E17c | NYU, KITTI × {DA-V2, UniDepthV2, Depth Pro}; layouts uniform, rows, cluster, edges, near; k ∈ {50, 200} | 20 × 10 | A-local, A-local-split, A-dist-split, A-dist-both, A-dual, A-dual-both, A-dual-rmax |
| E18c | ARKitScenes × {UniDepthV2, DA-V2}; uniform, cluster; k = 200; anchor source × probe source ∈ {laser, iPad}² | 20 × 10 | A-dual, A-dual-both, A-dist-split, A-dist-both, A-local-split |
| E19c | ARKitScenes, UniDepthV2 frozen centre vs OMNI-DC v1.1 completed from laser anchors; uniform, cluster; k = 200; probe source ∈ {laser, iPad} | 10 × 1 (split j uses its own anchor/probe mask, seed j) | split-CP, A-dist-split, A-dual, A-dual-both on each centre |

Depth Pro is excluded from ARKitScenes: its ARKitScenes pipeline is mis-scaled and its training data include ARKitScenes. OMNI-DC completed from iPad anchors stays exploratory (3 splits) and appears only in the supplement.

Commands: `scripts/run_confirm.sh` via `slurm/confirm.sbatch` (E17c, E18c; seed shards of 2), and `experiments.e19_omnidc --omni-sources clean --seeds j` for j = 0…9 (E19c).

## Metrics

Per run (split × draw), images weighted equally:
- **Coverage**: image-averaged coverage of the laser/ground-truth depth.
- **p10**: 10th percentile of per-image coverage.
- **Short share (single draw)**: share of test images whose coverage on one draw is < 1 − α − 0.05 = 0.85 (`frac_img_short`); theory at n = 20: 17.6 %.
- **Relative interval score (primary)**: per image, the mean over pixels of the interval score divided by the true depth; then the median over images (`rel_interval_score_imgmed`). It keeps the miss penalty inside each image and is robust to a few extreme images.
- **Secondary, always reported next to the primary**:
  - `rel_interval_score_robust`: per image, the median over pixels; then the mean over images. It mostly drops the miss penalty at 90 % coverage, because the median pixel is usually covered.
  - `frac_uninformative`: share of pixels whose interval is wider than 10× the true depth (infinite included). This exposes the failures the median hides.
  - `rel_interval_score`: the mean, for continuity with the exploratory tables.
  - `median_rel_width`, and `median_width_over_gt` (E19).

## Decision rules

1. **Validity.** A method is valid in a cell if its mean coverage over splits is ≥ 1 − α − 2 SE (SE = SD over splits / √splits). For A-dual with probes that measure the target, the theory check is |mean − 0.905| ≤ 2 SE (`exact`).
2. **Comparison.** "A beats B on metric M" requires both:
   - the 95 % percentile bootstrap CI (10,000 resamples of splits) of the mean per-split difference A − B excludes 0 in A's favour;
   - A is valid in that cell.

   Relative differences are point estimates. Ranges over pairs and layouts are reported as min–max over cells. There is no multiplicity correction: only the comparisons listed below are headline claims, and every other number is descriptive.
3. **Truth-coverage claims with iPad probes.** These are dead in any cell where A-dual's mean truth coverage fails rule 1.
4. **Centre comparison (E19c).** The completion centre "wins" a layout if its best valid method beats UniDepthV2 + A-dual by ≥ 15 % on the primary metric under rule 2. The frozen centre "wins" if the reverse holds.
5. **Envelope gate.** The envelope goes in the main paper only if, with iPad probes, its truth coverage is valid by rule 1 at 0.9 **and** it beats matched A-dist-both on the primary metric by ≥ 10 % under rule 2. Otherwise it moves to the supplement (`\envelopemainfalse`) with a limitation sentence. The gate is evaluated per layout. Placement after a pass in one layout only is an author decision, recorded here before the envelope run.

## Pre-specified comparisons

- **E17c**, per pair × layout × k:
  - A-dual and A-dual-both vs A-dist-both (primary metric, median relative width);
  - A-dual-rmax vs A-dual (width cost, short share; this is the design-rule box);
  - A-local coverage (descriptive: the failure result).
- **E18c**, per model × layout × source cell:
  - validity of A-dual, A-dual-both and A-dist-split against the truth;
  - A-dual vs A-dist-both (primary metric).
- **E19c**, laser anchors, per layout and probe source:
  - OMNI-DC {split-CP, A-dist-split, A-dual} vs UniDepthV2 A-dual;
  - OMNI-DC A-dual vs OMNI-DC split-CP;
  - UniDepthV2 {A-dual, A-dist-split} vs OMNI-DC {A-dual, split-CP} (cluster).

Analysis: `python -m experiments.confirm_analysis` (validity.csv, compare.csv); the paper numbers come from those files via `paper/tools/gen_numbers.py`.

## Amendment log

- 4 Nov 2026, before any envelope run (E18c seeds 0–3 had finished; their outputs were not analysed): the sensor-error envelope's calibration changed from per-class one-sided conformal risk control at levels π_c·α_e/2 to one two-sided conformal risk control at level α_e = 0.05 over a class-adaptive one-parameter family. Reason: with U = 37 calibration scenes per split, conformal risk control cannot certify levels below 1/(U+1) = 0.026; the per-class targets are 0.0003, 0.001 and 0.024 (class shares 1.1 %, 4.3 %, 94.6 %), so every threshold would be infinite. Rule (all fitted per split, before any test data):
  - bias β̂ = median over head-train pixels of e = log(iPad reading) − log(laser depth);
  - band of confidence class c at parameter λ: [Q_c(λ/2), Q_c(1 − λ/2)] of the corrected head-train error e − β̂, clipped to contain 0 (a_c = −Q_c(λ/2), b_c = Q_c(1 − λ/2));
  - λ̂ = the largest value on a fixed grid (400 log-spaced values in [10⁻⁶, 0.5]) with U/(U+1)·L̂(λ) + 1/(U+1) ≤ α_e, where L̂ is the mean over calibration scenes of the fraction of the scene's pixels outside their class band; no feasible λ gives infinite bands;
  - A-dual runs with bias-corrected iPad probes at level 1 − α_s = 0.95 (arm `A-dual-s`); the envelope widens it to [ℓe^{−b_c}, ue^{a_c}] (arm `A-dual-env`). α_s = α_e = 0.05 and the gate (rule 5) are unchanged.
  - Run (E20): `experiments.e18_sensor_dual --envelope --cells clean/lidar lidar/lidar --exp-id e20_env`, ARKitScenes × {UniDepthV2, DA-V2} × {uniform, cluster}, k = 200, 20 splits × 10 draws, via `scripts/confirm_pool.py --parts e20`. Gate comparison: A-dual-env vs A-dist-both in the same cell (primary metric), per layout and probe cell.
- 3 Nov 2026, before any confirmatory output existed: the primary interval score changed from the per-image pixel median (`rel_interval_score_robust`) to the median over images of the per-image mean (`rel_interval_score_imgmed`). Reason: on a 400-image smoke test the pixel median roughly halved A-dual's score (0.047 vs 0.088 vs 0.103 for the plain mean), because it ignores most misses.
