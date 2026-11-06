# SpaCal-Depth

Post-hoc conformal calibration of **frozen** monocular metric depth models (Depth Anything V2 Metric, UniDepthV2, Depth Pro, Metric3D v2): per-pixel depth intervals with distribution-free, finite-sample coverage guarantees, evaluated on NYU Depth v2, KITTI, DIODE, ETH3D and ARKitScenes. The exchangeable unit is the **image or scene**, not the pixel. The main method, **A-dual**, splits a sparse depth sensor's budget into structured anchors that build the prediction and a few uniformly random probes that calibrate it inside each image, which gives a per-image guarantee for any anchor layout.

The course report is [`report_neurips/main.pdf`](report_neurips/main.pdf) (sources in `report_neurips/`). The full write-up of the broad study (E0–E16), with the theory behind each guarantee, is [`docs/PROJECT_REPORT.md`](docs/PROJECT_REPORT.md); the analysis plan of the confirmatory runs (E17c–E20), fixed before they ran, is [`docs/ANALYSIS_PLAN_2026-11-03.md`](docs/ANALYSIS_PLAN_2026-11-03.md).

## Main results

Broad study (E0–E16):

- **The image (or scene) is the unit.** Run-to-run variability of coverage follows the number of calibration images, about 100× above a pixel-count prediction. Scene-level calibrators fix the small shortfall of image-level ones at a 3–12 % width cost.
- **Coverage is even on average, not per image.** Per-image coverage ranges from about 40 % to 100 %; one global scale error per image explains 63–79 % of that variation.
- **A few true depths per image ("anchors") help most.** About 20 random anchors match an oracle that knows each image's scale: intervals 1.1–3.7× narrower at the same 90 % average coverage. A-both cuts images below 0.85 coverage from 16 % to 0.2 % (NYU, DA-V2).
- **Structured anchors break within-image calibration, silently.** Calibrating inside the image on clustered anchors covers only 0.35–0.41 at a 0.9 target, with *narrower* intervals; cross-image guarantees survive.
- **Distance-aware anchors (A-dist).** Coverage stays flat across distance to the nearest anchor under all five layouts, and intervals are up to 67 % narrower than A-hybrid.
- **Safety queries.** A guarantee protects only the quantity it was calibrated on: calibrating the nearest depth in a region directly is both valid and informative; per-pixel guarantees are not.

Main method (confirmatory runs E17c–E20):

- **A-dual restores the per-image guarantee.** It covers 0.904–0.905 in every layout, anchor count and model–dataset pair (theory r/(n+1) = 0.905 with n = 20 probes), is 16–69 % narrower than the hull baseline A-dist-both, and has the better interval score in 46 of 60 cells. Per-image coverage follows the predicted Beta law (10th percentile 0.819).
- **Which centre (E19c, ARKitScenes).** Under uniform anchors the depth-completion network OMNI-DC is the better centre (16–20 % lower interval score); under one clustered patch the frozen UniDepthV2 with an anchor-built centre is better (41 % lower). A-dual runs unchanged on either.
- **Probes certify what they measure (E18c).** With iPad LiDAR anchors and laser probes A-dual covers the laser truth at 0.905; with iPad probes it certifies the iPad reading and covers the laser truth at only 0.57–0.63 (uniform anchors).
- **Sensor-error envelope (E20).** Widening by a calibrated iPad-error band restores truth coverage (0.993–0.996) but costs 2–78 % more interval score than A-dist-both, so it fails the gate fixed in advance.

![Coverage by distance to the nearest anchor and A-dist width](results/figures/e14_distance__S2-log.png)

## Repository layout

```
src/
  config.py          paths and device via SPACAL_* environment variables
  data/              NYU, KITTI, DIODE, ETH3D, ARKitScenes loaders, eval protocol, PixelTable storage, scene splits
  models/            DA-V2, UniDepthV2, Depth Pro and Metric3D v2 wrappers (GPU only)
  scores/            scale maps S0–S4c, learned S5/S6, groups, raw baselines
  calibrators/       exact CRC, K1/K1-scene, RCPS, split CP, Mondrian, registry,
                     anchors.py (A-rescale, A-hybrid, A-local, A-both, layouts), anchor_dist.py (A-dist),
                     anchor_noise.py (noisy anchors), dual.py (A-dual, A-dual-both), lscp.py (LSCP baseline)
  metrics/           depth metrics, interval metrics, safety queries
experiments/         prepare_tables, e0–e19, run_calibration, confirm_analysis, summarize, make_figures,
                     qualitative, qual_dual (+ _fig: the qualitative ARKitScenes figures)
scripts/             data download, prediction caching, OMNI-DC forward pass, confirmatory task pool, run_*.sh
slurm/               sbatch files for a Slurm cluster
colab/               notebooks for GPU caching and the OMNI-DC run on Colab
tests/               85 tests, incl. Monte-Carlo checks of each guarantee
paper/               macros.tex, numbers.tex (every number in the report), refs.bib, figures used by the report;
                     tools/gen_numbers.py regenerates numbers.tex from the results
report_neurips/      the course report (NeurIPS 2025 format)
results/             figures*/ (all figures) and, per experiment, summary.md/csv, spec.json and _confirm/ tables
results_omnidc/      E19 (OMNI-DC) per-split results and confirmatory tables
docs/                project report of the broad study and the confirmatory analysis plan
```

Raw per-image and per-run CSVs (about 0.5 GB), prediction caches and prepared tables are not in the repository; they are regenerated by the pipeline below. Every result directory has a `spec.json` with its exact settings.

## Install

Python 3.12.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt     # CPU only: every calibration experiment, tests, figures
```

Model inference (caching predictions) needs a GPU and the full stack: `bash scripts/setup_remote.sh` installs torch (CUDA 12.8 wheels), `requirements.txt` and UniDepth. Depth Pro needs `transformers >= 4.49`. Metric3D v2 comes through torch.hub and needs `mmengine` (or `mmcv`); fetch it on a node with internet with `TORCH_HOME=$PWD/.torch_cache python -m scripts.prefetch_models --only-metric3d`. On Colab, use the notebooks in `colab/` with `requirements-colab.txt`.

## Pipeline

```
GPU  ── scripts/cache_predictions.py ──►  cache/<dataset>/{gt,<model>}/<sample>.npz   (once)
CPU  ── experiments/prepare_tables.py ──►  prepared/<dataset>__<model>.npz           (valid pixels on a lattice)
CPU  ── experiments/run_calibration.py, e*.py ──► results/<exp>/*.csv ──► summarize.py, confirm_analysis.py, make_figures.py
CPU  ── paper/tools/gen_numbers.py ──► paper/numbers.tex ──► report_neurips/main.tex
```

Everything after caching is numpy / scipy / scikit-learn and runs on any CPU. Paths and device come from `SPACAL_DATA_ROOT`, `SPACAL_CACHE_ROOT`, `SPACAL_PREPARED_ROOT`, `SPACAL_RESULTS_ROOT`, `SPACAL_DEVICE` and `SPACAL_SEED` (`src/config.py`).

## Reproducing the results

Broad study (E0–E16):

```bash
bash scripts/download_data.sh                 # NYU labelled + splits, KITTI Eigen test, DIODE val (~8 GB)
GPUS="0" bash scripts/smoke_remote.sh         # 3 images per pair, E0 table, tests
GPUS="0 1 2 3" bash scripts/run_all_gpu.sh    # cache predictions, one dataset per GPU (resumable)
bash scripts/run_all_cpu.sh                   # tables, main study, E0, E1, E5, E8, summaries, figures
bash scripts/run_e11.sh                       # anchors (E11)
bash scripts/run_new.sh                       # E12, E13 and the Depth Pro pairs
PARTS="e14 e15" bash scripts/run_new.sh       # A-dist, calibration vs test layout, LSCP
python -m scripts.add_kitti_lidar --zip data/kitti/data_depth_velodyne.zip   # raw-scan anchors for E16
PARTS=e16 bash scripts/run_new.sh             # noisy and real-LiDAR anchors
PARTS="metric3d eth3d" bash scripts/run_new.sh   # fourth model, fifth dataset (after caching them)
python -m experiments.make_figures
```

Main method, confirmatory runs (E17c on NYU/KITTI, E18c and E20 on ARKitScenes; 20 splits × 10 probe draws):

```bash
bash scripts/download_data.sh arkitscenes     # depth-upsampling validation split (~14 GB), then cache and prepare it
python -m scripts.confirm_pool --parts e17 e18 e20 --seeds 0-19 --draws 10   # or: sbatch slurm/confirm_pool.sbatch
python -m experiments.confirm_analysis --runs 'results/e18_conf/*/runs.csv' \
    --by source pred_source probe_source --arm-cols method \
    --compare A-dual:A-dist-both A-dual-both:A-dist-both --out results/e18_conf/_confirm
```

E19c (OMNI-DC as the centre, ten visit splits) needs a GPU for the network: export the anchor masks with `scripts/export_dual_anchor_masks.py`, run `scripts/omnidc_forward.py` (see `colab/e19c_omnidc.ipynb`), merge with `scripts/omnidc_merge.py`, then run `experiments/e19_omnidc.py` and `experiments/confirm_analysis.py`; each script's docstring has the exact command. The fairness ablation (`--simple-only`: arms `A-dual-const` and `A-dual-med`, which give OMNI-DC a fit-free centre instead of the distance model) runs on CPU from the merged outputs with `colab/e19s_simple_scale.ipynb`. The qualitative ARKitScenes figures come from `experiments/qual_dual.py` (compute) and `experiments/qual_dual_fig.py` (render).

On a Slurm cluster the same steps are in `slurm/` (`smoke`, `cache`, `cpu`, `e11`, `new`, `confirm_pool`, `omnidc`); `slurm/env.sh` holds the shared job environment. Pass partition and GPU type on the command line, e.g. `sbatch -p <partition> --gres=gpu:<type>:1 slurm/cache.sbatch`. The full CPU pipeline of the broad study takes roughly 8–10 hours on 32 cores.

To rebuild the report after new results:

```bash
python paper/tools/gen_numbers.py
cd report_neurips && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Methods

Scores (`src/scores/`): S0 constant, S1 relative, S2 log, S3 flip, S3e ensemble, S4 gradient, S4c UniDepth confidence, S5 learned quantile head, S6 factorised (image difficulty × within-image shape). Every scale model is fit on a separate head-train split.

Calibrators without anchors (`src/calibrators/registry.py`):

| id | guarantee |
|---|---|
| K0 | none (pixels treated as exchangeable; naive baseline) |
| K1 / K1-scene | E[uncovered fraction of a test image] ≤ α, over images / over scenes |
| RCPS | the same with probability ≥ 1 − δ |
| K2 / K2-scene | P(a fixed patch fully covered) ≥ 1 − α |
| K3-depth / K3-edge | K1 within each depth bin / edge group (Mondrian) |
| V4 / V4-scene | P(image coverage ≥ 1 − β) ≥ 1 − α |

Calibrators with k sparse measurements per test image (`src/calibrators/anchors.py`, `anchor_dist.py`, `dual.py`):

| id | guarantee |
|---|---|
| A-rescale, A-hybrid | average over scenes (anchor shift; A-hybrid also normalises by the anchors' spread) |
| A-local | per image, in expectation over random anchor draws (needs k ≥ 18 at α = 0.1); fails for structured anchors |
| A-both | hull of A-hybrid and A-local: both |
| A-dist(-auto), A-dist-split | average over scenes within each distance-to-nearest-anchor bin |
| A-dist-both | k − 20 anchors by layout + 20 at random; hull with A-local on the random 20: both |
| **A-dual** | k − n structured anchors build centre and scale, n uniform probes set one threshold per image: per image, any layout, any probe-independent centre, coverage law Beta(r, n + 1 − r) |
| A-dual-both | hull of A-dual and A-dist-split: per image and per distance bin |
| A-dual+env | A-dual on sensor-valued probes widened by a calibrated sensor-error band: truth coverage on average over scenes |
| LSCP (baseline) | Jiang & Xie's localized spatial conformal prediction within each image: finite-sample only under spatial exchangeability, which structured anchors break |

## Experiments

| exp | script | question |
|---|---|---|
| E0 | `experiments/e0_base_metrics.py` | base-model accuracy |
| E1 | `experiments/e1_synthetic.py` | does coverage spread follow images or pixels (synthetic) |
| E2–E4, E6, E9 | `experiments/run_calibration.py` | main study: scores × calibrators, 20 seeds |
| E5 | `experiments/e5_shift.py` | domain shift (NYU → DIODE indoor, KITTI → DIODE outdoor) |
| E8 | `experiments/e8_calsize.py` | calibration-set size |
| E10 | `experiments/e10_cost.py` | runtime cost |
| E11 | `experiments/e11_anchors.py` | guarantees vs number of anchors |
| E12 | `experiments/e12_safety.py` | nearest depth in a forward region and per lane |
| E13 | `experiments/e11_anchors.py --exp-id e13 --pattern {rows,cluster,edges,near}` | structured anchor layouts |
| E14 | `experiments/e11_anchors.py --exp-id e14 --pattern ... --dist-kinds median knn kernel bilateral` | distance-aware anchors and the budget split |
| E15 | `experiments/e15_layouts.py --pattern ...` | calibration anchors under a different layout than test anchors; LSCP baseline |
| E16 | `experiments/e16_noisy.py` | noisy, outlier, biased and real-LiDAR anchors |
| E17 / E17c | `experiments/e11_anchors.py --exp-id e17_...` (`scripts/run_confirm.sh`, `scripts/confirm_pool.py`) | A-dual on NYU/KITTI, five layouts, k = 50/200, n = 20 probes |
| E18 / E18c | `experiments/e18_sensor_dual.py` | ARKitScenes: laser vs iPad LiDAR as anchor and probe values |
| E19c | `experiments/e19_omnidc.py` | depth-completion network (OMNI-DC) vs frozen model as the centre; `--simple-only` for the fit-free OMNI-DC arms |
| E20 | `experiments/e18_sensor_dual.py --envelope` | sensor-error envelope for iPad probes |

## Tests

```bash
python -m pytest
```

Covers the exact CRC solver, infeasible-α handling, each guarantee under within-image correlation (Monte-Carlo checks), A-local's exact coverage, A-dist's per-bin coverage and bin merging, A-dual's per-image validity under clustered anchors (probes never enter the centre or scale), PixelTable layout, score/interval duality and an end-to-end run on a synthetic cache. A laptop dry run without data or GPU:

```bash
export SPACAL_CACHE_ROOT=$PWD/.scratch/synthetic/cache SPACAL_PREPARED_ROOT=$PWD/.scratch/synthetic/prepared SPACAL_RESULTS_ROOT=$PWD/.scratch/synthetic/results
python -m experiments.make_synthetic_cache
SEEDS=3 E8_SEEDS=4 E5_SEEDS=2 E1_REPS=10 bash scripts/run_all_cpu.sh
```

## License

MIT, see [`LICENSE`](LICENSE). The datasets and pretrained models keep their own licenses.
