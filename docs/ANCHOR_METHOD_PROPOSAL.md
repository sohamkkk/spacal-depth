# Proposal: anchor-geometry–aware conformal calibration (A-dist)

Status: design, not implemented. Claims about prior work are marked `[VERIFY]` until the literature search is repeated. Nothing here claims to be "first".

## 0. Summary

E11 and E13 show that sparse metric anchors (a few pixels with measured depth at test time) can shrink depth intervals 1.3–3.7×. They also show that our two families of anchor methods each fail in one direction:

- **A-local** (within-image split conformal) has a per-image guarantee, but only when anchors are a uniform random sample of the image. Under realistic anchor layouts it collapses, for example to 26–47 % coverage for clustered anchors. It also becomes *narrower* while failing, so the failure is silent.
- **A-rescale / A-hybrid** (one shift and one scale per image, calibrated across images) keep their average guarantee under every layout. But they ignore where the anchors are, so coverage drops with distance from the nearest anchor.

We propose **A-dist**, which adds three things:

1. A **local shift**: anchor errors interpolated to every pixel, instead of one median per image.
2. A **distance-aware scale**: the scale grows with distance to the nearest anchor.
3. **Per-distance-bin calibration**: conformal risk control is run across images, separately in each bin of distance to the nearest anchor (the existing K3 Mondrian machinery).

The target guarantee is **coverage ≥ 1 − α within every distance-to-anchor bin**, for any anchor layout, as long as calibration simulates the same layout rule. A companion variant, **A-dist-both**, adds a per-image guarantee whenever a small part of the anchor budget can be placed at random.

The goal is a method that is (a) valid under the anchor layouts real sensors produce, (b) adaptive to anchor geometry, and (c) no wider than A-hybrid where anchors are informative.

## 1. Motivation: what the data already show

All numbers below use the plain log score at α = 0.1, from `results/e11` and `results/e13`. "Pairs" means dataset/model combinations.

| Observation | Evidence |
|---|---|
| A-local is exact under uniform anchors | 0.908–0.910 / 0.922–0.925 / 0.900–0.901 at k = 20 / 50 / 200; theory 0.909 / 0.923 / 0.901 |
| A-local breaks under structured layouts | clustered anchors: 0.26–0.36 at k = 20 and 0.37–0.47 at k = 200; near-biased: 0.71–0.79; rows: 0.87–0.90 |
| …and narrows while breaking | median width with clustered anchors 0.02–0.07, against 0.15–0.45 with uniform anchors |
| Even uniform anchors give distance-dependent coverage | A-local at k = 20: 0.93–0.95 within 0.02 of the diagonal of an anchor, 0.61–0.84 beyond 0.2 |
| Cross-image anchored methods keep their average | A-rescale / A-hybrid at 0.89–0.94 under all five layouts (within 0.01 of their uniform value) |
| …but pay for structured layouts in width | A-rescale median width +30–45 % under clustered anchors |
| A single image-wide shift wastes information | A-hybrid at k = 200 still leaves 9.6–11.9 % of NYU images short; the oracle-shift width equals A-hybrid's (E11, Finding 1), so a constant shift is saturated |

Real anchors are structured:

| Layout | Real source |
|---|---|
| Rows | few-beam LiDAR scan lines |
| Near | range-limited time-of-flight or stereo |
| Edges | SLAM / structure-from-motion feature points |
| Cluster | a user tap, or a single rangefinder spot |

A method whose guarantee assumes uniform anchors does not match deployment.

## 2. Setup and notation

- Image i has valid ground-truth pixels P_i with log residuals d_j = log y_j − log ŷ_j.
- An **anchor rule** R picks A_i ⊂ P_i, with |A_i| = k. R may use any image features and its own randomness. E13's patterns are examples.
- The evaluation pixels are E_i = P_i \ A_i.
- At test time, the anchors' residuals d_a (a ∈ A_i) are observed. They are test-time features, exactly as in E11 (T7 in `PROJECT_REPORT.md`).
- δ_i(x) is the distance from pixel x to its nearest anchor, divided by the image diagonal (already computed by `nearest_anchor_distance` in `src/calibrators/anchors.py`).
- Data are split, per seed, into head-train / calibration / test (`split_tables`). Calibration and head images get anchors from the **same rule R** as test images.

The key exchangeability fact: if images are i.i.d. and R is applied independently and identically to each one, then the tuples Z_i = (x_i, y_i, A_i, d_{A_i}) are i.i.d. Anything computed from Z_i alone (interpolated shifts, distances, bins, per-image losses) inherits this. This is why A-hybrid survived E13, and A-dist relies on the same fact.

## 3. Method

### 3.1 Local shift m_i(x)

The local shift estimates d at pixel x from the image's anchors. Candidates, in order of complexity:

| Id | Local shift | Notes |
|---|---|---|
| M0 | global median of d_A | = A-rescale / A-hybrid (regression baseline) |
| M1 | median of the K nearest anchors (image plane) | robust to outlier anchors |
| M2 | Gaussian-kernel weighted mean, bandwidth h (fraction of diagonal) | smooth; falls back to M0 when the total weight is below ε |
| M3 | bilateral weights: exp(−‖x − x_a‖² / 2h²) · exp(−(log ŷ(x) − log ŷ(x_a))² / 2h_d²) | anchors on the same predicted surface dominate (joint-bilateral upsampling idea) |

K, h, h_d and ε are chosen on the **head split** by log interval score, as `pick_tau` does for τ. They are never chosen on calibration data.

### 3.2 Scale s_i(x)

s_i(x) = b(x) · (σ_i(x) + τ) · g(δ_i(x))

- b(x) is the base score map: 1 for the plain log score, or the learned S5 map. It must not depend on anchor values when combined with A-local; see §5.
- σ_i(x) is the local anchor spread. It is the leave-one-out median absolute deviation of the K nearest anchors' residuals around their own local shift, the same construction as A-hybrid's σ but local.
- τ is a floor, chosen on the head split (existing `pick_tau`).
- g(δ) is a non-decreasing function fitted on the head split. It is the isotonic fit of the median |d − m| / ((σ + τ)·b) against δ. It encodes "farther from anchors means less certain".

### 3.3 Per-distance-bin calibration

The score is r_j = |d_j − m_i(x_j)| / s_i(x_j).

Bins are B = {<0.02, 0.02–0.05, 0.05–0.1, 0.1–0.2, ≥0.2} of the image diagonal (`DIST_LABELS` in `experiments/e11_anchors.py`).

For each bin b:

1. Take the calibration images with at least `min_pixels` evaluation pixels in bin b. This is the existing filter in `src/calibrators/mondrian.py`.
2. Each image's loss is L_i^b(λ) = the share of its bin-b evaluation pixels with r_j > λ.
3. Set λ_b = the CRC solution over those images. This is `mondrian_crc_lambdas` with groups = distance bins. A bin that is too small gives λ_b = ∞.
4. Merge bins that are too sparse **on the head split**, before seeing calibration data. For example, with uniform anchors at k = 200 almost no pixels are ≥ 0.1 from an anchor.

The test interval at pixel x in bin b is [ŷ · exp(m − λ_b s), ŷ · exp(m + λ_b s)]. Anchor pixels take their measured depth.

**A-dist-scene** is the same procedure with CRC over scenes (`crc_lambda_exact_units`), for clustered data.

### 3.4 Algorithm (per seed)

```
head, cal, test = split_tables(table, seed)
for T in (head, cal, test): A_T = R(T)                     # same rule everywhere
fit (K, h, h_d, tau, g, bin merges) on head                 # never on cal
for T in (cal, test): m_T, s_T, delta_T = features(T, A_T)  # uses anchors' d, never eval pixels' d
r_cal = |d_cal - m_cal| / s_cal
lambda_b = mondrian_crc_lambdas(r_cal, image_of, bin(delta_cal), n_bins, n_cal, alpha)
interval_test = pred * exp(m_test -/+ lambda_{bin(delta_test)} * s_test)
```

## 4. Guarantee

**Proposition 1 (per-bin marginal validity).** Assume:

- calibration and test images are i.i.d.;
- the anchor rule R is applied independently and identically to every image;
- the hyperparameters and bin definitions are fixed using head-split data only.

For each bin b, let λ̂_b be the CRC solution on the n_b calibration images that pass bin b's filter. Then, for a test image that passes the same filter,

  E[ L_test^b(λ̂_b) ] ≤ α,

that is, the expected share of its bin-b evaluation pixels that fall outside the interval is at most α.

*Proof sketch.*

1. With the hyperparameters fixed, L_i^b(·) is a function of Z_i alone. It is non-increasing in λ and bounded by 1.
2. The filter "has ≥ `min_pixels` evaluation pixels in bin b" is a function of Z_i (anchor positions and validity masks, not label values). So the selected images are again i.i.d. from the conditional distribution given the filter event. This is the same argument as K3 (`mondrian.py` docstring) and danger-CRC in E12.
3. Conditional on n_b, apply the CRC theorem (Angelopoulos et al., ICLR 2024).

The anchor *values* enter through m and s. That is allowed because they are part of Z_i, and the test image's anchor values are observed at test time. ∎

What it does **not** give:

1. **No per-image guarantee.** Coverage holds on average over images, not for each image. That is what A-local and A-dist-both (§5) add.
2. **No guarantee if the test anchor rule differs from the simulated one.** Calibrating with uniform anchors and deploying with clustered anchors is outside the theorem. We test this mismatch empirically (§7.3), and we expect the per-distance-bin design to degrade more gracefully than a single λ.
3. **No guarantee under domain shift.** As for every cross-image method, target-domain calibration data are needed. On DIODE (3–4 calibration scans) A-dist is infeasible, and A-local remains the only option (E11, Finding 4).

**Special cases (regression checks).** M0 with a single bin and g ≡ 1 gives A-hybrid exactly. M0 with σ ≡ 0 and τ = 1 gives A-rescale.

## 5. A per-image companion: A-dist-both

**Remark (no free lunch).** Suppose a pixel has zero probability of being chosen as an anchor (a pixel far from the cluster, a non-edge pixel under the edge rule). If the residual field is otherwise arbitrary, the anchors say nothing about its error. So no rule using only that image's anchors can guarantee per-image coverage there. Per-image guarantees require the evaluated pixels to be reachable by random anchor selection. This explains A-local's E13 failures. It is elementary, not a new result.

**Constructive consequence.** If the sensor or protocol allows it, split the budget as k = k_s + k_r:

- k_s structured anchors, chosen by the sensor;
- k_r anchors drawn uniformly at random from the remaining ground-truth pixels.

Then:

- A-local on the k_r random anchors keeps its exact per-image guarantee for a uniformly random evaluation pixel. It needs k_r ≥ 18 at α = 0.1 and k_r ≥ 38 at α = 0.05.
- A-dist uses all k anchors.
- **A-dist-both** is the hull of the two intervals. By the standard union argument it carries both guarantees, as A-both does in E11/T9.

This turns the E13 finding into a design rule: *spend about 20 anchors on random probes if you need per-image guarantees; place the rest wherever the sensor likes.*

## 6. Relation to prior work

Checked against a web search on 20 Oct 2026; still `[VERIFY]` against a full manual read of each paper before the related-work section is finalized, and against any paper published after this date.

| Work | Relation |
|---|---|
| Mondrian conformal prediction (Vovk et al.) | per-group validity; our groups are distance-to-anchor bins |
| Conformal risk control (Angelopoulos et al., ICLR 2024) | the calibration engine |
| Conformal prediction with conditional guarantees (Gibbs, Cherian & Candès, *JRSS-B* 2025, arXiv:2305.12616) | group-conditional coverage over a class of covariate-defined shifts; confirmed — our per-bin guarantee is a special case with groups defined by test-time anchor geometry, not a new statistical mechanism |
| Kandinsky conformal prediction (Bairaktari et al., ICML 2025, arXiv:2502.17264) | generalizes Mondrian/Gibbs-style group-conditional coverage to overlapping, fractional groups defined jointly on covariates and labels; our fixed, disjoint distance bins are a simple special case of this more general framework too |
| Localized conformal prediction (Guan, *Biometrika* 110(1), 2023) | localizes coverage in feature space by distance-weighting calibration points; confirmed as a direct mechanism match — we localize by anchor distance instead of a generic feature distance, with groups fixed across images rather than per-test-point weights |
| Spatial conformal inference through localized quantile regression / LSCP (Jiang & Xie, arXiv:2412.01098, Dec 2024 / ICML 2026) | **overlap is with the local-shift/scale part, not the cross-image guarantee.** LSCP calibrates *within a single spatial field*, using residuals near the new location — mechanically close to A-dist's M1-M3 local shift and distance-aware scale, but its finite-sample guarantee needs spatial exchangeability (stationarity and mixing for the asymptotic version), which is the same precondition A-local needs and the same one E13 breaks under structured layouts. So LSCP is closer to **A-local** than to A-dist's per-distance-bin, *cross-image* guarantee, and it would plausibly fail the same way under clustered/edge/near anchors. It makes a good **E14 baseline**: a localized method operating within one image, tested under the five E13 layouts (added to §7.1). |
| Conformalized matrix completion (Gui, Barber & Ma, NeurIPS 2023, arXiv:2305.10637) | weighted, within-instance conformal method for non-uniformly sampled entries — exactly the reweighted-A-local idea raised informally in §6's Tibshirani row. Also fails when some entries have zero probability of being sampled, matching §5's "no free lunch" remark directly. Already row 104 of `PROJECT_REPORT.md`'s prior-work table — this proposal should cite it in the same place. |
| Weighted conformal prediction under covariate shift (Tibshirani et al., NeurIPS 2019) | could reweight A-local for non-uniform anchor sampling, but needs positive selection probability everywhere; §5's remark says why this fails for clustered / edge / near layouts |
| Spatial conformal prediction / valid model-free spatial prediction (Mao, Martin & Reich, *JASA* 119(546), 2024) | within-field exchangeability, the basis of A-local |
| Distributionally robust conformal prediction (Cauchois, Gupta, Ali & Duchi, *JASA* 119(548), 2024, arXiv:2008.04267) | gives valid coverage for any test distribution within an f-divergence ball of the calibration distribution — the right tool if A-dist needs to be robust to an *unknown* anchor layout rather than one matching the simulated rule (see §12) |
| Not all distributional shifts are equal: fine-grained robust conformal inference (Ai & Ren, ICML 2024, PMLR 235, arXiv:2402.13042) | shows Cauchois et al.'s joint f-divergence approach can be overly conservative in practice, and reweights for an identifiable covariate shift while protecting against worst-case conditional shift only within an f-divergence ball — relevant if §12.2 is ever pursued |
| Adaptive conformal inference under distribution shift (Gibbs & Candès, NeurIPS 2021, arXiv:2106.00170) | anytime-valid long-run coverage under an unannounced, possibly drifting shift, by adapting a single parameter online — relevant if the anchor layout rule itself drifts over deployment time rather than being fixed (see §12) |
| Achieving Risk Control in Online Learning Settings (Feldman, Ringel, Bates & Romano, TMLR 2023, arXiv:2205.09095) | already confirmed in `PROJECT_REPORT.md` Appendix B, item 4 — the earlier "could not be re-confirmed" note was wrong; this is the correct citation. Uses ~200 ground-truth pixels per frame to fit scale/shift online, with a long-run average guarantee over frames, not per-distance-bin within a frame. |
| Uncertainty in sparse-to-dense depth completion | searched 20 Oct 2026; no conformal-prediction paper found in this area (closest are learned/Bayesian uncertainty methods: Eldesokey et al. CVPR 2020, Qu et al. ICCV 2021, Zhu et al. AAAI 2022). Treat as open, not as confirmed absence — a narrower, dedicated search pass is still owed before the paper claims no conformal depth-completion baseline exists. |
| Conformal depth completion | same search as above; no direct hit. Same caveat applies. |

**Revised candidate contribution**, now that the mechanism itself (distance/locality-based group-conditional conformal prediction) is confirmed non-novel — covered in general form by Guan (2023) and Gibbs et al. (2023/2025), in overlapping-group form by Kandinsky (2025), and in a spatial-statistics form close enough to be a real collision risk by Jiang & Xie (2024/2026):

- using **distance to a sparse, test-time-observed metric anchor** — not a generic spatial or feature-space covariate — as the localization variable, in the specific sensor-deployment setting (LiDAR scan lines, SLAM clusters, rangefinder spots) none of the above address;
- the evaluation under realistic, non-uniform anchor layouts (E13) — an empirical question, not a theorem, and one nobody in this table has asked;
- the budget-split design rule (§5), which is a design decision about sensor budget allocation, not a statistical result, and is the part most clearly free of prior art.

The statistical machinery (Mondrian CRC, kernel/localized interpolation, group-conditional coverage) is standard and increasingly well-covered by 2023-2025 work; the contribution is the application and the empirical finding, not a new guarantee mechanism. This framing should carry through to §1 of `CVPR_PLAN.md` if this experiment is folded into the paper, in the same spirit as the E12/E13 contribution statement's "the contribution is the analysis, not a new calibrator."

## 7. Experiments (proposed E14)

### 7.1 Main grid

| Factor | Levels |
|---|---|
| Pairs | KITTI and NYU × DA-V2, UniDepthV2, Depth Pro (6) |
| Anchor layout | uniform, rows, cluster, edges, near (E13 rules) |
| k | 5, 10, 20, 50, 200 (A-dist, unlike A-local, has no minimum k) |
| Base score | plain log, S5-shape |
| α | 0.05, 0.1 |
| Seeds × draws | 5 × 3 (as E13) |
| Methods | A-rescale, A-hybrid, A-local, A-both, **A-dist (M1/M2/M3)**, **A-dist-scene**, **A-dist-both** (k_r = 20 when k ≥ 50), **LSCP** (Jiang & Xie-style within-image localized quantile regression, as a within-image baseline, §6; k ≥ 50 only, since the anchors are the image's only labelled points and must cover both fitting and calibration; new code, unlike the other baselines) |

**Metrics:**
- the E13 set: image-averaged coverage, per-image SD, short-image share, median and mean width, interval score, over-estimate rate;
- coverage per distance bin, and the **worst-bin gap** (target minus the minimum bin coverage);
- the share of infinite bins.

### 7.2 Ablations (one pair per dataset, k = 20 and 200)

1. Local shift: M0 vs M1 vs M2 vs M3.
2. Scale: with and without g(δ); with and without local σ.
3. Calibration: 1 bin (marginal) vs 5 bins (per-bin).
4. Bandwidth sensitivity: h ∈ {0.02, 0.05, 0.1, 0.2}.

These isolate how much comes from the better shift (completion) and how much from the per-bin calibration (the guarantee).

### 7.3 Layout mismatch

Calibrate under layout P and test under layout Q, for all 5 × 5 pairs at k = 50, on one pair per dataset. This measures how A-dist, A-hybrid and A-local degrade outside the theorem. It is the experiment that tells a practitioner what happens with a wrong simulator.

Add two more configurations (§12.1), same grid: **mixture calibration** (pool calibration images equally across all 5 layouts, report per-layout miscoverage — the mixture averages to α, so it is expected to exceed α on the hardest layouts, likely cluster and near, and to over-cover the easiest ones; the `α / w_l` bound only caps the worst case) and **max-λ calibration** (one score function fitted on the pooled head split, λ calibrated separately per layout and per bin, the largest deployed; report per-layout miscoverage — expected to hold ≤ α everywhere at the cost of width). Report both against matched calibration (Proposition 1) as the baseline, since matched calibration is the right default whenever the sensor type is known in advance.

### 7.4 Completion baseline

"Interpolate anchor residuals (M2), then plain K1" is a non-learned depth-completion baseline. It is A-dist with one bin and g ≡ 1, so it falls out of the ablation. A learned completion network would need a GPU and is optional; we note it as a threat to validity if skipped.

### 7.5 Safety query (E12)

Add the A-dist lower bound as an E12 method. The question is whether a per-distance guarantee lowers the danger miss rate near anchors at an acceptable false-alarm cost.

### 7.6 Success and kill criteria (fixed before running)

- **S1 (validity).** Per-bin coverage ≥ 1 − α − 2 SE in every bin with ≥ 20 test images, for every layout and pair, under a matched rule.
- **S2 (efficiency).** Median width ≤ A-hybrid's at the same k on at least 4 of 6 pairs for the uniform and rows layouts.
- **S3 (evenness).** Short-image share below A-hybrid's.
- **Kill / negative result.** If S1 holds but the width is more than 10 % above A-hybrid everywhere, report it as "the cost of a per-distance guarantee" rather than as an improved method.

## 8. Implementation plan

| File | Change |
|---|---|
| `src/calibrators/anchors.py` | `local_shift(d, row, col, pred, offsets, draw, kind, K, h, h_d)` (per-image cKDTree, already used by `nearest_anchor_distance`); `local_spread(...)`; `fit_distance_scale(head, ...)` (isotonic) |
| `src/calibrators/mondrian.py` | reused unchanged (groups = distance bins) |
| `experiments/e11_anchors.py` | methods `A-dist`, `A-dist-scene`, `A-dist-both`; `--k-random` for the budget split; `--cal-pattern` for the mismatch study; head-split hyperparameter search alongside `pick_tau` |
| `tests/test_anchors_dist.py` | (1) with M0, one bin and g ≡ 1, A-dist equals A-hybrid exactly; (2) synthetic smooth residual field with clustered anchors: per-bin coverage ≥ 1 − α − 3 SE over repetitions, while A-local fails; (3) A-dist-both covers ≥ both components pixel by pixel |
| `scripts/run_new.sh` | part `e14` |
| `experiments/make_figures.py` | coverage-vs-distance figure (methods × layouts); width vs k |

**Compute.** A kNN search over ≤ 200 anchors for about 13–16k pixels per image takes milliseconds. The head-split search adds roughly one E11 run's cost. Estimate: the main grid needs about 60 jobs of 30–40 min each, about 6–7 hours at 6 jobs in parallel, all on CPU.

## 9. Risks and fallbacks

| Risk | Fallback |
|---|---|
| Far bins have too few images (uniform, large k) | merge on the head split; report merged bins |
| Local shift overfits noisy anchors (KITTI LiDAR) | M1 (median) and a larger K; local σ widens automatically |
| No width gain over A-hybrid | still report: the per-bin guarantee and the mismatch study are contributions by themselves |
| Layout mismatch breaks validity badly | becomes a documented limitation, plus the recommendation to simulate the deployed layout |
| Prior conformal depth-completion work exists | reposition as an evaluation under structured anchors, cite it, and compare |

## 10. Timeline (about 2 weeks, CPU only)

| Day | Work |
|---|---|
| 1 | Finalise this design; write the tests from §8 (test-first) |
| 2–3 | Implement `local_shift`, `local_spread`, `fit_distance_scale`, the methods and the head-split search |
| 4–5 | LSCP baseline; mixture and max-λ calibration for the mismatch study |
| 6 | Local smoke run on the Depth Pro tables (`prepared_depthpro/`), all five layouts, 1 seed |
| 7 | Cluster run of the main grid plus ablations |
| 8 | Mismatch study (matched, mixture, max-λ) plus the E12 integration run |
| 9–10 | Analysis, figures, report section |

## 11. Open decisions

1. Include α = 0.05 in the main grid? It is cheap (calibration is a fraction of the run time), but far bins will be infeasible more often.
2. Is a learned completion baseline in scope (needs a GPU), or is the M2 baseline enough?
3. Name: "A-dist" or something descriptive for the paper ("distance-Mondrian anchored CRC")?

## 12. Can this be made to "work in any case and scenario"?

**No, not literally, and that is close to a theorem, not a gap in this proposal.** Literal "any layout" includes adversarial layouts, against which no finite calibration can certify coverage — that part is elementary, the same robustness fact as §5's remark, one level up. If the layout is instead treated as a covariate and the question is exact coverage conditional on every possible value of it, Barber, Candès, Ramdas & Tibshirani, "The limits of distribution-free conditional predictive inference" (arXiv:1903.04684; *Information and Inference*, 2021) additionally rules that out in finite samples without distributional assumptions. Proposition 1 already lives inside both limits — it only promises validity when the deployed anchor rule matches the one simulated at calibration time (§4, point 2).

What is achievable, in three different and non-equivalent senses of "any scenario":

**12.1 Calibration over the known layout family — cheap, but the mixture version is weaker than it sounds; the max version is the one with the guarantee wanted.** Pooling calibration images across a mixture of the five E13 layout rules (uniform, rows, cluster, edges, near) does **not** give validity for a deployment that uses mostly one layout — it only bounds the loss *averaged over that mixture*. If layout `l` has mixture weight `w_l`, the only bound for that single layout follows from the losses being non-negative: `Σ_l w_l R_l ≤ α` implies `miscoverage(l) = R_l ≤ α / w_l`, which is `0.5` at five equal weights and `α = 0.1` — useless. "Doesn't need to know the right simulator" was not true for a real sensor, which produces one layout, not a random mix of five.

The fix that gives the guarantee actually wanted: calibrate **separately for each layout** in the family, then take `λ = max` over layouts of each layout's own `λ_l`. Since each layout's loss is non-increasing in λ, using the max keeps every layout's own `E[loss] ≤ α` at that shared λ — so the result is valid for *every* layout in the known family *simultaneously*, at the cost of being as conservative as the hardest layout requires. The argument needs one shared score function: the local-shift variant, bandwidths, `g(δ)`, `τ` and bin merges must be fitted once on the head split pooled across layouts (the deployed layout is unknown, so per-layout settings could not be chosen anyway), and with per-distance-bin calibration the maximum is taken separately in each bin. Report both numbers: the mixture version, labeled honestly as "average over a layout mix" (not a per-layout guarantee), and the max version, labeled "worst case over the known family" (the one that actually answers "works across any layout we've characterized"). Worth noting explicitly: in practice the sensor type is usually known in advance, so matched calibration (Proposition 1, calibrating against the one true deployment layout) is often simply the right default, and the max-λ version is insurance for the case where it isn't.

**12.2 Distributionally robust calibration over an uncertainty set — stronger, costlier, a future-work item.** Cauchois, Gupta, Ali & Duchi, "Robust Validation: Confident Predictions Even When Distributions Shift" (*JASA* 119(548), 2024; arXiv:2008.04267) give coverage valid for any test distribution within an f-divergence ball of the calibration distribution. Applied here, this would cover layouts that are *unknown perturbations* of the simulated ones, not just members of a finite enumerated family — the strongest honest version of "any scenario." It is not free: the method is flagged in follow-up work (Ai & Ren) as conservative in practice, and choosing the right divergence-ball radius for "all layouts a real sensor might produce" is itself unresolved, not a solved input. Out of scope for this deadline; worth one sentence in a "future work" paragraph.

**12.3 Adaptive online calibration under drift — a different axis, not a stronger version of 12.1/12.2.** Gibbs & Candès, "Adaptive Conformal Inference Under Distribution Shift" (NeurIPS 2021; arXiv:2106.00170) give anytime-valid long-run coverage under an unannounced, possibly drifting shift, by nudging a single parameter online based on whether the last prediction missed. This answers "any scenario" in the sense of *drift over deployment time* (a sensor degrading, a robot switching modes) rather than *simultaneous validity across layouts at one instant*. It needs a sequential, online evaluation setting this project's current batch structure does not have — a separate experiment, not an A-dist variant.

**Bottom line for scheduling:** 12.1 (both the mixture and the max-λ configurations) is the only one of these that fits inside the 2-week estimate (§10); it is now a deliverable, added to §7.3's mismatch study, not a reason to delay. 12.2 and 12.3 are legitimate citable directions, not commitments — naming them in a future-work paragraph is honest and costs nothing; attempting either before the deadline risks the same scope creep already flagged for A-dist as a whole.
