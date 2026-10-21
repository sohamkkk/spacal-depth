# SpaCal-Depth: Research Plan, Theory Primer and Build Guide

Computer Vision course research project · Plan v1 · 8 October 2026 · living document

## 0. How to use this document

Read sections 2–5 first (theory, about 2–3 days with the reading list). Sections 6–15 are the build plan. Two markers are used throughout:

- **\[VERIFY\]**: I could not confirm this from a source while writing. Check it before you rely on it or put it in the report.
- **\[HYPOTHESIS\]**: a claim your experiments must test, not something to assert.

The project deadline was not given, so the timeline in section 11 assumes 8 weeks and shows what to cut if you have 4. Send me the real date and I will re-cut it.

## 1. Executive summary and honest novelty positioning

**One-paragraph project.** Take a frozen, pretrained monocular depth model. Wrap it with conformal calibration so that every pixel gets a depth interval with a finite-sample, distribution-free guarantee. Study how spatial correlation between pixels changes what "valid" even means, show which calibration schemes are correct and which only look correct, and benchmark against the standard depth-uncertainty baselines on coverage, width and efficiency.

**Correction to what I told you earlier.** I said nobody had adapted conformal prediction to dense depth regression. A deeper search says that is too strong. What exists:

- Per-pixel distribution-free intervals for image-to-image regression, using risk-controlling prediction sets (Angelopoulos et al., ICML 2022), with public code (im2im-uq).
- Spatially-aware conformal methods for discrete outputs: CONSIGN for segmentation (arXiv 2505.14113) and spatial-aware conformal prediction for hyperspectral classification (arXiv 2409.01236).
- Spatially-correlated uncertainty for image restoration (arXiv 2305.10124).
- An online risk-control paper (arXiv 2205.09095) whose references cite depth-estimation models; it may contain a depth experiment **\[VERIFY: read it and cite it as closest prior art if so\]**.

I did not find a dedicated study of spatially-aware conformal calibration for modern depth foundation models, but a search is not proof of absence. So do **not** write "first". The defensible claim is:

> A systematic, correctly-formulated study of spatially-aware conformal calibration for monocular depth foundation models, including the exchangeability unit, three validity notions, geometry-aware scores and groups, and a reusable benchmark protocol.

**Contributions to aim for**

- **C1.** Formulate image-level conformal risk control for depth, with pixels as sub-units and images as the exchangeable unit (section 5).
- **C2.** Region and group calibration (patch max-score, Mondrian groups defined from predicted geometry) with stated guarantees.
- **C3.** Geometry-aware score functions (relative, log-depth, learned scale map) compared at equal coverage.
- **C4.** A conditional-coverage study across depth range, edges, sky, reflective surfaces and distribution shift.
- **C5.** A benchmark protocol and baseline comparison. No official leaderboard for calibrated depth intervals exists, so defining one is itself a contribution.

**What "better" means.** Conformal methods hit nominal coverage by construction, so "better" cannot mean "more covered". It means tighter intervals at the same coverage, more uniform coverage across hard cases, and lower cost than ensembles. Section 10 fixes the decision rule.

## 2. Monocular depth estimation (MDE): theory

### 2.1 Geometry and why the problem is ill-posed

A pinhole camera with intrinsics K (focal lengths f\_x, f\_y, principal point c\_x, c\_y) maps a 3D point (X, Y, Z) to pixel (u, v):

```
u = f_x · X / Z + c_x
v = f_y · Y / Z + c_y
(X, Y, Z) = Z · K⁻¹ · [u, v, 1]ᵀ        (back-projection given depth Z)
```

For any s > 0, the point s·(X, Y, Z) lands on the same pixel. One image therefore fixes geometry only up to a global scale. A network resolves this using learned priors (typical object sizes, room layout, camera field of view). This is why monocular depth is called ill-posed, and why **metric** depth (real metres) is much harder than **relative** depth (ordering and shape).

Contrast with stereo, where Z = f·B / d (B baseline, d disparity), so a disparity error δd gives a depth error of roughly δZ ≈ Z² · δd / (f·B). Errors grow quadratically with distance. Learned monocular models show a similar growth of absolute error with depth (empirically, roughly proportional to depth). That is the reason to use relative or log-space scores later **\[HYPOTHESIS: check on your data\]**.

### 2.2 Learning formulation

Learn f\_θ : ℝ^{H×W×3} → ℝ₊^{H×W} from image–depth pairs. Common losses, with d\_i = log ŷ\_i − log y\_i over n valid pixels:

```
L1:      (1/n) Σ |ŷ_i − y_i|
SILog:   sqrt( (1/n) Σ d_i² − λ · ((1/n) Σ d_i)² ),   λ = 0.85 typical (Eigen et al. 2014)
```

With λ = 1 the loss is fully scale-invariant. Ground truth is never dense: Kinect has holes on shiny or distant surfaces, KITTI LiDAR is sparse. Every loss and metric is computed only on a validity mask M.

### 2.3 Relative vs metric vs affine-invariant

- **Metric models** output depth in metres. They depend on camera intrinsics and training domains (Metric3D v2, UniDepth, Depth Pro, ZoeDepth, Depth Anything V2 metric variants).
- **Affine-invariant models** (MiDaS, DPT, Depth Anything V2 relative) predict disparity up to an unknown per-image scale and shift. They are evaluated after fitting (a, b) on the ground truth: 1/y ≈ a · ŝ + b, least squares per image. That fit uses ground truth, so these outputs **cannot give metric intervals at test time** without extra information. This matters for your method: your primary base models should be metric.

### 2.4 Standard metrics

```
AbsRel = (1/n) Σ |ŷ_i − y_i| / y_i
RMSE   = sqrt( (1/n) Σ (ŷ_i − y_i)² )
δ_k    = fraction of pixels with max(ŷ_i / y_i, y_i / ŷ_i) < 1.25^k,   k = 1, 2, 3
```

Always report these for your base models. They prove you reproduced the base model correctly before you add uncertainty on top.

### 2.5 Where the field is, and its drawbacks

Rough lineage: Eigen et al. (2014, multi-scale CNN, SILog) → DORN, AdaBins, BinsFormer (depth as classification over adaptive bins) → MiDaS and DPT (mixed-dataset training, affine-invariant) → ZoeDepth (metric heads on relative backbones) → Depth Anything V1/V2 (large-scale pseudo-labelling, strong zero-shot) → Metric3D v2, UniDepth (camera-agnostic metric), Depth Pro (sharp boundaries), Marigold (diffusion prior). Newer releases may exist after my knowledge cutoff **\[VERIFY: check for newer Depth Anything or UniDepth versions before choosing models\]**.

Drawbacks that motivate your project:

- Point estimates only. No model reports a calibrated, guaranteed error bar.
- Metric scale depends on training-domain priors and camera intrinsics; it degrades under shift.
- Systematic failure regions: depth discontinuities and thin structures, sky, reflective and transparent surfaces, very far range.
- Evaluation culture is dominated by point metrics (AbsRel, δ1), which hide uncertainty quality.

### 2.6 Error sources (useful for your analysis section)

- **Aleatoric**: sensor noise, ground-truth holes and noise, boundary mixing, reflective or transparent surfaces. Does not shrink with more data.
- **Epistemic**: out-of-distribution scenes, unfamiliar camera, unseen layout. Shrinks with more or better data, and spikes under shift.

## 3. Uncertainty and calibration: theory

### 3.1 Predictive distributions and calibration

A model with predictive distribution p(y | x) and CDF F\_x is **quantile-calibrated** (Kuleshov et al. 2018) if for every level q in (0, 1):

```
P( Y ≤ F_X⁻¹(q) ) = q
```

For intervals, the practical target is **coverage**: P(Y ∈ \[l(X), u(X)\]) = 1 − α. Two things to keep apart:

- **Calibration** (coverage is right).
- **Sharpness** (intervals are narrow). A trivially huge interval is perfectly calibrated and useless. Always report both.

### 3.2 Metrics you will use

```
PICP  = fraction of valid pixels with y in [l, u]                    (coverage)
MPIW  = mean of (u − l) over valid pixels                            (width)
Interval score (Winkler; Gneiting & Raftery 2007), proper score:
  IS_α(l, u; y) = (u − l) + (2/α)(l − y)·1{y < l} + (2/α)(y − u)·1{y > u}
Gaussian NLL: ½ log(2π σ²) + (y − μ)² / (2 σ²)
ECE_reg = Σ_j w_j · | q_j − q̂_j |   (observed vs nominal quantile frequencies over bins j)
Sparsification: remove the most uncertain pixels first, track the error of the rest;
  AUSE = area between that curve and the oracle curve (ranks uncertainty, ignores coverage)
```

Interval score is the single best summary because it trades width against miscoverage in one proper number.

### 3.3 Existing depth-uncertainty methods and their limits

| Method | Idea | Limitation |
| --- | --- | --- |
| MC-Dropout (Gal & Ghahramani 2016) | Sample T dropout passes, use variance | T× inference cost; no coverage guarantee; often under-dispersed |
| Deep ensembles (Lakshminarayanan 2017) | M independently trained models | M× training and inference cost; no guarantee |
| Heteroscedastic NLL (Kendall & Gal 2017) | Network predicts mean and variance | Needs retraining; Gaussian assumption; miscalibrates under shift |
| Poggi et al. (CVPR 2020, arXiv 2005.06209) | Empirical, snapshot, bootstrapped and learned variants for self-supervised depth | Judged by sparsification (ranking), not by coverage |
| Gradient-based uncertainty (Hornauer & Belagiannis, ECCV 2022 and arXiv 2502.05964) | Post-hoc, from gradients, no retraining | Good ranking; no coverage guarantee **\[VERIFY details\]** |
| MonoProb (arXiv 2311.06137) | Probabilistic self-supervised depth with interpretable uncertainty | Training-time method; no guarantee |

Common gap: **ranking quality is measured; guaranteed coverage is not.** Conformal prediction fills exactly that gap.

## 4. Conformal prediction: the mathematics

### 4.1 Split conformal prediction and its guarantee

Setup. A fixed pretrained model (frozen) gives a point prediction ŷ(x). Hold out n calibration pairs (X\_i, Y\_i). Assume the n calibration pairs and the test pair are **exchangeable** (their joint distribution is invariant to reordering; i.i.d. is the common special case).

```
Score:      S_i = |Y_i − ŷ(X_i)|
Threshold:  q̂ = the ⌈(n+1)(1−α)⌉-th smallest of S_1..S_n   (q̂ = ∞ if that index exceeds n)
Interval:   C(x) = [ ŷ(x) − q̂ , ŷ(x) + q̂ ]
Guarantee:  1 − α  ≤  P( Y_{n+1} ∈ C(X_{n+1}) )  ≤  1 − α + 1/(n+1)
```

The upper bound holds when scores are almost surely distinct.

**Proof sketch.** By exchangeability the rank of S\_{n+1} among the n+1 scores is uniform on {1, …, n+1}. The test point is covered iff its rank is at most ⌈(n+1)(1−α)⌉. That event has probability ⌈(n+1)(1−α)⌉/(n+1) ≥ 1−α. No assumption on the model or the data distribution was used.

**How noisy is one calibration run?** Conditional on the calibration set, the realised coverage follows a Beta(n+1−l, l) law with l = ⌊(n+1)α⌋, with standard deviation about sqrt(α(1−α)/(n+2)). For n = 300 and α = 0.1 that is about 0.017, i.e. ±1.7 points. Use this to size your calibration sets and to judge whether your observed coverage is plausible.

### 4.2 Scores for regression

- **Absolute residual** (above): constant-width bands. Simple but ignores that errors grow with depth.
- **Normalised (locally adaptive; Lei et al. 2018):** S = |Y − ŷ| / σ̂(X) gives C(x) = ŷ ± q̂ · σ̂(x). Any heuristic uncertainty map can serve as σ̂: ensemble std, TTA std, gradient-based map, a learned head. **Conformal turns any heuristic into a guaranteed interval**; the heuristic only affects width and uniformity, never validity.
- **Conformalised quantile regression (CQR; Romano et al., NeurIPS 2019, arXiv 1905.03222):** train quantile heads q̂\_lo, q̂\_hi, then S = max(q̂\_lo(x) − y, y − q̂\_hi(x)) and C(x) = \[q̂\_lo − q̂, q̂\_hi + q̂\]. Intervals can be asymmetric, which suits depth (far errors skew).
- **Log-space score:** S = |log Y − log ŷ| gives multiplicative intervals \[ŷ·e^{−q̂}, ŷ·e^{q̂}\], a natural fit for depth.

### 4.3 What marginal coverage does not give you

The guarantee is **marginal**: averaged over the test input and over the calibration draw. It does not say coverage is 1−α for far pixels, for edges, or for any particular image. Distribution-free **conditional** coverage is impossible in general (Vovk 2012; Barber, Candès, Ramdas, Tibshirani 2021). Practical approximations:

- **Normalised scores / CQR** (width adapts to difficulty).
- **Mondrian (group-conditional) conformal:** split the data into groups defined from the input and calibrate a separate threshold per group. Coverage then holds within each group. The group must be a function of x (or other quantities known at test time), never of the true y.

This is the core tool for your hard-case analysis: groups such as predicted-depth bins, edge versus non-edge, sky, and uncertainty deciles.

### 4.4 Conformal risk control and risk-controlling prediction sets

Coverage is one loss; you can control others. **Conformal risk control** (CRC; Angelopoulos, Bates, Fisch, Lei, Schuster, ICLR 2024) generalises split conformal to any bounded loss that is non-increasing in a size parameter λ.

```
Loss L_i(λ) ∈ [0, B], non-increasing in λ, for calibration item i = 1..n
λ̂ = inf { λ :  ( L_1(λ) + … + L_n(λ) + B ) / (n+1)  ≤  α }
Guarantee:  E[ L_{n+1}(λ̂) ] ≤ α   (and close to α from below for well-behaved losses)
```

**Risk-controlling prediction sets** (RCPS; Bates et al., JACM) give the high-probability version: P( E\[L(λ̂)\] > α ) ≤ δ, using a concentration bound over the calibration items. Im2Im-UQ uses this for image-to-image regression.

For depth, take λ as the interval half-width multiplier and define the loss as the **fraction of valid pixels not covered**. That is the heart of section 5.

## 5. Spatial correlation: where SpaCal-Depth lives

### 5.1 The exchangeability unit is the image, not the pixel

Standard split conformal needs exchangeable units. Pixels from one image are strongly dependent (errors are smooth in space and share causes such as a wrong scale or a mirror). So the exchangeable unit is the **image** (and, for video datasets, arguably the **scene or sequence**). Pooling millions of pixel scores from n images into one big calibration set and calling the (1−α) quantile "valid" treats them as if they were millions of independent samples. They are not.

**Effective sample size.** Take an image with m valid pixels whose miscoverage indicators have average pairwise correlation ρ̄. Then:

```
Var( per-image miscoverage fraction ) = p(1−p)/m · ( 1 + (m−1) ρ̄ )
m_eff = m / ( 1 + (m−1) ρ̄ )     →   Var → p(1−p) ρ̄  as m → ∞
```

Even a megapixel image does not average its own error away. Illustration (equicorrelation is an idealisation): with α = 0.1 and ρ̄ = 0.2, the per-image coverage has standard deviation about sqrt(0.09 × 0.2) ≈ 0.13. The average can be nominal while single images are far off.

**\[HYPOTHESIS H1\]** Naive pooled-pixel split conformal gets close to nominal average coverage but shows larger per-image dispersion and worse conditional coverage than image-level calibration. I expect this but have not proven it; section 7 includes a synthetic experiment that checks it cleanly.

### 5.2 Three validity notions

Pick the notion first; it determines the algorithm and the width you pay.

| ID | Guarantee | Calibration unit | Algorithm |
| --- | --- | --- | --- |
| **V1** Image-averaged pixel coverage | E\[fraction of valid pixels uncovered in a new image\] ≤ α | one loss per image | CRC with L\_i(λ) = fraction of uncovered valid pixels, B = 1 |
| **V2** Region-simultaneous coverage | P(every valid pixel in region R covered) ≥ 1−α | one max-score per image per region | Split conformal on S\_i = max over R of \|y − ŷ\| / s; R fixed from x (e.g. patch index) |
| **V3** Group-conditional coverage | V1 or V2 holds within each group g | one loss per image per group | Mondrian CRC; images with no valid pixels in g are excluded (the filter is applied identically to every item, so exchangeability is preserved among the rest) |

Trade-offs. V2 is far more conservative than V1 (the max over many pixels is large) but gives a stronger statement per region. V3 costs width but protects hard cases. Report all three; the **width-versus-validity curve** across V1/V2/V3 is a result in itself.

### 5.3 What exists and where the gaps are

| Work | What it does | Gap relative to your project |
| --- | --- | --- |
| Im2Im-UQ / RCPS (Angelopoulos et al. 2022) | Per-pixel intervals with risk control for image-to-image regression | Not studied with modern depth foundation models, geometry-aware groups or hard-case breakdowns **\[VERIFY against paper tasks\]** |
| CONSIGN (2025), SACP (2024) | Spatially-grouped conformal sets for segmentation and classification | Discrete outputs; interval width and scale behaviour for continuous depth untested |
| Principal UQ with spatial correlation (2023) | Correlated uncertainty directions for restoration | Different task; not distribution-free coverage per pixel |
| CQR and normalised conformal (2018–2019) | Adaptive intervals for tabular regression | Ignore spatial dependence and the image exchangeability unit |

So the gap you target is narrow but real: correct unit of exchangeability, spatially-aware validity notions, geometry-aware scores and groups, applied to depth foundation models with a proper benchmark.

## 6. SpaCal-Depth method specification

### 6.1 Pipeline

```
image x ──► frozen depth model f ──► ŷ (H×W metric depth)
   └──► scale map s(x) > 0  (one of S0..S5 below)

calibration only:  r = |y − ŷ| / s   on valid pixels M
calibrator (K0..K4) ──► λ̂  (scalar, per-patch, or per-group)
output:  [ ŷ − λ̂ · s ,  ŷ + λ̂ · s ]  per pixel
```

The base model stays frozen. Everything below is post-hoc, which is why the method is cheap and model-agnostic.

### 6.2 Scale maps (what sets interval shape)

| ID | Scale map s(x) | Needs training? | Role |
| --- | --- | --- | --- |
| S0 | constant (1) | no | Vanilla split conformal baseline |
| S1 | s = ŷ (relative error) | no | Tests "errors grow with depth" |
| S2 | log-space residual (multiplicative interval ŷ·e^{±λ}) | no | Natural geometry for depth |
| S3 | TTA std (flip, multi-scale) or ensemble / MC-dropout std where available | no | Cheap heuristic uncertainty |
| S4 | Gradient-based uncertainty (Hornauer & Belagiannis) | no | Strongest post-hoc depth baseline |
| S5 | Learned head on frozen features: quantile (pinball loss) or σ (Gaussian NLL) | small head only | Adaptive width; CQR variant gives asymmetric intervals |

Rule: train S5 on a **head-train** split that is disjoint from calibration and test.

### 6.3 Calibrators (what sets validity)

| ID | Method | Validity notion | Note |
| --- | --- | --- | --- |
| K0 | Pooled-pixel split conformal | none rigorous at image level | **Deliberately naive baseline**; used to show what goes wrong |
| K1 | Image-level CRC, loss = uncovered pixel fraction | V1 | Core method |
| K2 | Patch max-score (fixed patch grid, one score per image per patch) | V2 | Stronger per-region statement, wider |
| K3 | Mondrian CRC over groups | V3 | Groups from predicted depth bins, edge vs non-edge (predicted depth gradient), sky or invalid region, uncertainty deciles |
| K4 | Two-parameter scale-aware calibration (global scale error plus local residual) | V1 on a combined loss | **Stretch.** Only for affine-invariant models; needs care (Learn-Then-Test or Bonferroni split) |

### 6.4 Core algorithm (V1), ready to adapt

```python
import numpy as np

def per_image_losses(resid_norm_list, lambdas):
    """resid_norm_list[i]: 1-D array of |y - yhat| / s over VALID pixels of image i.
    Returns L[i, j] = fraction of valid pixels with normalised residual > lambdas[j]."""
    L = np.empty((len(resid_norm_list), len(lambdas)))
    for i, r in enumerate(resid_norm_list):
        r = np.sort(r)
        L[i] = 1.0 - np.searchsorted(r, lambdas, side="right") / len(r)
    return L

def crc_lambda(L, lambdas, alpha, B=1.0):
    """lambdas ascending. Smallest lambda with (sum_i L_i(lambda) + B)/(n+1) <= alpha."""
    n = L.shape[0]
    risk = (L.sum(axis=0) + B) / (n + 1)
    ok = np.where(risk <= alpha)[0]
    return lambdas[ok[0]] if len(ok) else lambdas[-1]   # flag the fallback: intervals are huge
```

Use a dense, log-spaced λ grid that reaches well past the largest plausible normalised residual. Each image costs one sort, so calibration over thousands of images takes seconds on CPU. For K3 run the same code per group on the images that contain valid pixels of that group.

### 6.5 Mapping your four novelties to implementation and acceptance tests

| Novelty (from your proposal) | What to implement | Acceptance test (done means) |
| --- | --- | --- |
| **N1.** Adapt spatially-aware conformal prediction to dense depth regression | K1, K2, K3 with continuous-valued scores S0–S5 | Mean coverage within ±1 point of nominal over at least 20 scene-level resplits on NYU and KITTI, at α in {0.05, 0.1, 0.2} |
| **N2.** Region/patch-grouped score aggregation with guarantees | K2 and K3 plus a short written proof sketch (section 5) | Synthetic correlated-error experiment (E1) matches the theory; real-data coverage per group within sampling noise from section 4.1 |
| **N3.** Systematic calibration study | Conditional-coverage tables by depth range, scene type, edges, sky, reflective surfaces, shift | Every hard-case group has coverage, width and n reported with confidence intervals |
| **N4.** Benchmark against MC-dropout, ensembles, gradient-based | Baselines in raw and conformalised form (section 10) | Results table with paired bootstrap intervals; efficiency table |

## 7. Hypotheses and experiment matrix

| ID | Experiment | Tests | Success criterion |
| --- | --- | --- | --- |
| E0 | Reproduce base depth metrics (AbsRel, RMSE, δ1) on NYU and KITTI | Pipeline correctness | Close to published numbers for the same model, or a documented reason |
| E1 | **Synthetic validity check**: truth depth map plus errors from a Gaussian random field (white noise through scipy.ndimage.gaussian\_filter, amplitude proportional to depth, variable correlation length ℓ). Compare K0 vs K1 across ℓ, n, image size | H1 and the section 5.1 variance formula | K1 hits the CRC guarantee; K0 shows larger per-image dispersion growing with ℓ. Runs on CPU in about a day |
| E2 | K0 vs K1 vs K2 vs K3 on real data | H1, N1, N2 | Coverage and per-image dispersion tables |
| E3 | Score ablation S0–S5 at fixed calibrator | **H2:** geometry-aware scores give narrower intervals at equal coverage | Lower interval score, paired CI excludes 0 |
| E4 | Group-conditional coverage with and without K3 | **H3:** Mondrian groups shrink the worst-group coverage gap at modest width cost | Worst-group gap drops clearly; width cost reported |
| E5 | Shift: calibrate on NYU, test on DIODE-indoor and SUN RGB-D; same for KITTI to DIODE-outdoor | **H4:** coverage degrades under shift; measure by how much and why | Coverage-vs-shift table; optional recovery with a small target-domain calibration set |
| E6 | Baselines raw vs conformalised (section 10) | N4 | Main results table |
| E7 | Scale-aware calibration on a relative model | **H5 (stretch)** | Only if time allows |
| E8 | Calibration-set size sweep (n = 50, 100, 200, 500, 1000+) | Section 4.1 noise prediction | Observed spread matches Beta prediction |
| E9 | α sweep {0.05, 0.1, 0.2} | Robustness | Coverage tracks nominal across α |
| E10 | Cost: inference passes, wall-clock, memory per method | "More efficient" claim | Wrapper cost vs MC-dropout (T passes), ensembles (M models), TTA (2+ passes) |

## 8. Datasets and splits

Sizes below are from memory. **\[VERIFY all counts on the official pages before the report.\]**

| Dataset | Content | Use |
| --- | --- | --- |
| NYU Depth v2 | Indoor, Kinect, depth up to about 10 m, 640×480. Official depth test set 654 images (Eigen split); 1,449 labelled images with semantic labels; large raw training pool (commonly about 24K–36K frames) | Primary indoor; semantic labels give window and mirror masks for hard cases |
| KITTI (Eigen split) | Outdoor driving, sparse LiDAR ground truth, depth cap 80 m. Test set 697 images (652 with improved ground truth); use the standard Garg or Eigen crop | Primary outdoor |
| DIODE | Dense indoor and outdoor, validation set about 771 images | Distribution shift |
| SUN RGB-D | Indoor, multiple sensors, about 10K images | Sensor shift |
| iBims-1 | About 100 high-quality indoor images with edge and plane annotations | Boundary and discontinuity analysis |
| ETH3D / ScanNet / Sintel | Optional extra domains | Only if time allows |
| Hypersim / Virtual KITTI 2 | Synthetic | Only to understand the base model; not needed for calibration |

### 8.1 Split design (read this twice)

Three disjoint pools: **head-train** (only for S5), **calibration**, **test**.

- **Split by scene or sequence, never by frame.** NYU: split by scene name. KITTI: split by drive sequence. Adjacent frames are near-duplicates and silently break exchangeability.
- **Never calibrate on images the base model trained on.** Models fine-tuned on NYU or KITTI train will be overconfident on those images.
- **Recommended primary setting:** use a metric model that never saw real NYU or KITTI data. The Depth Anything V2 metric indoor model was fine-tuned on Hypersim and the outdoor one on Virtual KITTI 2 **\[VERIFY on the model cards\]**. Then all of NYU and KITTI is unseen, so you can draw thousands of calibration images and your calibration/test splits stay large. Expect larger errors (domain gap) and therefore wider intervals; that is an honest and interesting result.
- If you use models trained on the official train splits, calibrate on held-out test-split scenes only (small). Compensate with many repeated random scene-level resplits.
- Apply the validity mask M and the depth cap in **every** loss, score and metric. Evaluate at the same resolution and crop as the model's published protocol.
- Cache predictions once (float16 files, about 0.6 MB per 480×640 image) so all calibration experiments run on CPU in seconds.

## 9. Models, libraries and compute

### 9.1 Base depth models (use two at minimum)

**\[VERIFY every identifier and check for newer releases.\]**

| Model | Type | How to get it | Role |
| --- | --- | --- | --- |
| Depth Anything V2 (S/B/L) metric indoor and outdoor | Metric | Hugging Face transformers (look for the Depth-Anything-V2-Metric-Indoor/Outdoor-Large-hf checkpoints) | **Primary** |
| UniDepth (V1/V2) | Metric, camera-agnostic; may expose a confidence output | Official GitHub repo | Second model; confidence as extra baseline |
| Metric3D v2 | Metric | Official GitHub repo | Alternative second model |
| ZoeDepth | Metric | torch hub / GitHub | Older SOTA reference |
| Depth Pro | Metric, sharp edges | Apple GitHub repo | Optional (boundary behaviour) |
| Marigold | Diffusion; sample variance gives uncertainty | GitHub / diffusers | Optional expensive baseline |
| Depth Anything V2 relative | Affine-invariant | transformers | Only for stretch E7 |

### 9.2 Libraries

| Purpose | Library |
| --- | --- |
| Deep learning | PyTorch, torchvision, timm, Hugging Face transformers |
| Conformal tooling | MAPIE (conformal regression, CQR), TorchCP, crepes (all optional); the im2im-uq repository (reference RCPS and quantile-head code for image-to-image regression); your own 30-line CRC from section 6.4 |
| Numerics and stats | numpy, scipy, scikit-learn, pandas, statsmodels (bootstrap) |
| Images and IO | opencv-python, h5py, Pillow |
| Experiment hygiene | Hydra or OmegaConf for configs, Weights & Biases or TensorBoard, pytest |
| Plots | matplotlib, seaborn |

### 9.3 Compute

One 16–24 GB GPU (or Colab/Kaggle) is enough, because the backbone is frozen and only a small head is trained. Caching predictions makes everything after the forward pass CPU-only. Budget most GPU time for feature extraction and the optional TTA / ensemble baselines.

## 10. Benchmarking against the state of the art

### 10.1 What "SOTA" means for this project

- **Base depth accuracy is not your contribution.** You only show it is reproduced correctly (E0), so reviewers trust the rest.
- **There is no official leaderboard for calibrated depth intervals.** The comparison targets are (a) depth-uncertainty methods, (b) conformal and risk-control methods for regression, and (c) your protocol itself. State this plainly in the paper.

### 10.2 Baselines to run

| Family | Methods | Modes |
| --- | --- | --- |
| Depth uncertainty | TTA / MC-dropout std, deep ensemble (M = 3–5, e.g. different checkpoints or sizes if you cannot retrain), Gaussian-NLL head (Kendall & Gal), gradient-based (Hornauer & Belagiannis), Marigold sample std, UniDepth confidence | **Raw** (interval = ±z·σ under a Gaussian assumption) and **conformalised** (their σ used as scale map s) |
| Conformal regression | Constant-width split CP, normalised CP, CQR, Im2Im-UQ-style RCPS (reimplemented or via its repo) | Native |
| Yours | K1, K2, K3 with the best S-scores | Native |

The raw-versus-conformalised pairing is the sharpest single figure of the project: it shows that guarantees come from the calibrator while the scale map only changes width.

### 10.3 Metrics to report

- Coverage: pixel-averaged, **per-image dispersion** (standard deviation across images), group-conditional (depth bins, edges, sky, window and mirror, far range), worst-group coverage gap.
- Sharpness: MPIW, relative width (width divided by depth), interval score.
- Ranking quality: sparsification curves with AUSE and AURG (for continuity with prior depth-uncertainty papers).
- Calibration of raw σ methods: ECE\_reg, NLL.
- Efficiency: forward passes, wall-clock, memory.

### 10.4 Statistical rigor and the decision rule

- Repeat calibration/test splits at least 20 times (by scene), fixed seeds, report mean and standard deviation, and paired bootstrap 95% confidence intervals for differences.
- α in {0.05, 0.1, 0.2}; at least two datasets and two base models for any headline claim.
- **Decision rule.** Claim that method A beats B only if, at coverage within ±1 point of nominal, A has lower mean interval score and a smaller worst-group coverage gap, with paired bootstrap intervals excluding zero on at least two datasets or two base models. Otherwise report the result as mixed or negative. A clean negative result (for example "pooling pixels is fine here") is still a valid and useful finding.

## 11. Timeline (8 weeks, assumed start 12 October 2026)

Shift the dates to your real deadline. Tiers: **A** = needed for a complete project, **B** = strengthens the claims, **C** = stretch.

| Week | Dates | Work | Tier | Exit check |
| --- | --- | --- | --- | --- |
| 1 | 12–18 Oct | Read theory (section 14, items 1–6); set up repo and env; load Depth Anything V2 metric; build data loaders with masks and scene-level splits; reproduce base metrics (E0); start prediction caching | A | E0 numbers sane |
| 2 | 19–25 Oct | K0 and K1 on cached predictions with S0–S2; **synthetic validity experiment E1 (CPU only)** | A | First coverage tables; E1 figure |
| 3 | 26 Oct–1 Nov | Lighter week: add S3 (TTA std) and the S5 quantile head; unit tests for calibrators | A | S3, S5 produce intervals |
| 4 | 2–8 Nov | K2, K3; full runs on NYU and KITTI (E2, E3, E4, E9) | A | **Go/no-go checkpoint**: do results support H1–H3? If not, reframe as a negative-results study now |
| 5 | 9–15 Nov | Baselines raw and conformalised (E6), S4 gradient-based, second base model | B | Main results table drafted |
| 6 | 16–22 Nov | Shift experiments E5, size sweep E8, efficiency table E10; hard-case analysis (edges, far range, window/mirror) | B | Conditional-coverage figures |
| 7 | 23–29 Nov | Stretch E7 only if on schedule; otherwise rerun and polish; write report; make interval-heatmap demo notebook | B/C | Full draft |
| 8 | 30 Nov–4 Dec | Buffer, proofread, reproducibility script, slides | A | Submission-ready |

**If you only have 4 weeks:** drop the second base model, S4, K2, E5, E7 and E8. Keep E0, E1, E2 (K0 vs K1 vs K3), E3, E4, E6 with two baselines, E9 and the report. That is still a complete, defensible project.

## 12. Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| Zero-shot metric errors are large, so intervals are wide and unimpressive | Report relative width; add a small learned head (S5); frame wide intervals as an honest measure of the domain gap |
| Calibration set too small (official test splits only) | Use the zero-shot setting with the large unseen train pools; otherwise repeated scene-level resplits and Beta-based expected noise |
| Leakage between frames or from base-model training data | Scene/sequence splits; check each model card; document the split in the report |
| KITTI sparse ground truth distorts per-image loss | Always normalise by valid-pixel count; report the number of valid pixels; exclude images with too few |
| Naive pooling (K0) turns out to be fine | Report it; the contribution becomes "when and why spatial correlation matters", backed by E1 |
| Prior art appears mid-project (for example a depth experiment in the online risk-control paper) | Cite it, narrow the claim to the systematic study and benchmark, keep the framing from section 1 |
| Re-implementing gradient-based uncertainty eats time | Use the authors' code if available **\[VERIFY\]**; otherwise drop S4 (Tier B) |
| Scope creep into training new depth models | Frozen base models only; no backbone fine-tuning |

## 13. Deliverables and report outline

**Deliverables:** code repository with one-command reproduction, cached-prediction scripts, results tables and figures, report (6–8 pages, conference style), slides, and a demo notebook that overlays interval-width heatmaps and coverage maps on example images.

**Report outline:** Abstract; Introduction (problem, why guarantees matter, contributions C1–C5); Related work (section 5.3 table); Background (MDE, conformal, CRC); Method (K1–K3, scale maps, proof sketch); Experimental protocol (datasets, splits, baselines, metrics, decision rule); Results (E1 synthetic, E2–E4 main, E5 shift, E6 baselines, E10 cost); Analysis and failure cases; Limitations (marginal only, exchangeability under shift, sparse ground truth); Conclusion; Appendix with derivations and extra tables.

**Key figures to plan for:** E1 per-image coverage dispersion versus correlation length (K0 vs K1); coverage versus depth bin for K1 vs K3; width-versus-validity curve across V1, V2, V3; raw-versus-conformalised baseline bars; interval-width heatmap on example images; coverage under shift.

## 14. Reading list (in order)

**Conformal prediction (start here)**

1. Angelopoulos and Bates, *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification* (tutorial; read first).
2. Romano, Patterson, Candès, *Conformalized Quantile Regression* (NeurIPS 2019; arXiv 1905.03222).
3. Angelopoulos, Bates, Fisch, Lei, Schuster, *Conformal Risk Control* (ICLR 2024).
4. Angelopoulos et al., *Image-to-Image Regression with Distribution-Free Uncertainty Quantification and Applications in Imaging* (ICML 2022; code: aangelopoulos/im2im-uq). Closest prior art.
5. Bates, Angelopoulos, Lei, Malik, Jordan, *Distribution-Free, Risk-Controlling Prediction Sets* (JACM).
6. Lei et al., *Distribution-Free Predictive Inference for Regression* (JASA 2018).
7. Vovk, *Conditional Validity of Inductive Conformal Predictors* (2012) and Barber, Candès, Ramdas, Tibshirani, *The Limits of Distribution-Free Conditional Predictive Inference* (2021).
8. Gibbs and Candès, *Adaptive Conformal Inference Under Distribution Shift* (2021); Barber et al., *Conformal Prediction Beyond Exchangeability* (2023); Tibshirani et al., *Conformal Prediction Under Covariate Shift* (2019). For the shift experiments.

**Spatial and vision conformal work**

9. CONSIGN (arXiv 2505.14113); Spatial-Aware Conformal Prediction for Hyperspectral Classification (arXiv 2409.01236); Principal Uncertainty Quantification with Spatial Correlation for Image Restoration (arXiv 2305.10124); Achieving Risk Control in Online Learning Settings (arXiv 2205.09095; check for a depth experiment); Mossina et al., conformal semantic segmentation (2024) **\[VERIFY citation\]**.

**Depth estimation**

10. Eigen, Puhrsch, Fergus (NeurIPS 2014); Ranftl et al., MiDaS (TPAMI 2020) and DPT (ICCV 2021); Bhat et al., ZoeDepth (2023).
11. Yang et al., Depth Anything (CVPR 2024) and Depth Anything V2 (NeurIPS 2024); Piccinelli et al., UniDepth (CVPR 2024); Yin et al., Metric3D v2; Bochkovskii et al., Depth Pro (2024); Ke et al., Marigold (CVPR 2024).

**Uncertainty in depth and calibration**

12. Poggi et al., *On the Uncertainty of Self-Supervised Monocular Depth Estimation* (CVPR 2020; arXiv 2005.06209).
13. Hornauer and Belagiannis, *Gradient-Based Uncertainty for Monocular Depth Estimation* (ECCV 2022; arXiv 2208.02005) and the 2025 revisit (arXiv 2502.05964); MonoProb (arXiv 2311.06137).
14. Kendall and Gal (NeurIPS 2017); Gal and Ghahramani (ICML 2016); Lakshminarayanan et al. (NeurIPS 2017); Kuleshov et al. (ICML 2018); Gneiting and Raftery (JASA 2007); Ilg et al. (ECCV 2018, sparsification plots).

## 15. First 48 hours checklist

1. Create the repo with the layout `data/`, `src/{models,scores,calibrators,metrics,viz}/`, `configs/`, `scripts/`, `tests/`, `notebooks/`.
2. Set up the environment; confirm GPU access; load Depth Anything V2 metric on one NYU image and overlay the depth.
3. Download NYU Depth v2 (official test split plus a train subset) and write the loader with the validity mask, depth cap and scene-name parsing.
4. Write and unit-test the three metrics (AbsRel, δ1, RMSE) and the CRC functions from section 6.4 on synthetic arrays.
5. Read reading-list items 1–3 and write a half-page in your own words on why the image, not the pixel, is the exchangeable unit.
6. Read the online risk-control paper (arXiv 2205.09095) to settle whether it contains a depth experiment; note the answer in section 1.
7. Send me your real deadline so I can re-cut section 11, and tell me whether you want the experiment code scaffolded next.
