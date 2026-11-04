# SpaCal-Depth: response to the review of 25 October and new results

**Review version 2, 2 November 2026.** This document accompanies `docs/PROJECT_REPORT.md` (review version of 25 October), which the reviewer has read. It answers that review point by point (§A–§C), describes the new method and its theory, and reports the new experiments. New sections carry the report's numbering so they slot into it: §5 (ARKitScenes), §6.11 (method), §7.6 (theory), §9.22–§9.23 (results), Limitations 14–19. Statements of the 25 October report that this document supersedes are listed in §E.

Every number comes from the result files of cluster jobs 44976 (E17 with the unshrunk scale), 45317 and 46165 (E17 with the final scale; 46165 adds A-dual-both and is the run reported), 46095 (probe-budget ablation), 46166 (E16 on ARKitScenes) and 46203 (E17 on ARKitScenes), with ARKitScenes predictions cached on a Colab A100 on 30–31 October. The analysis scripts are `.scratch/analyze_dualboth.py`, `.scratch/compare_gamma.py`, `.scratch/compare_probes.py`, `.scratch/analyze_arkit.py` and `.scratch/report_e17_numbers.py`. Statements that still need an external check are marked `[VERIFY]` and listed in §G.

---

## A. Summary

The review asked for three things: refocus on *structured measurements for prediction, randomized probes for per-image calibration*; compare against a modern depth-completion model; and validate with sensor ground truth independent of the anchor sensor. It also set four gates. Outcome:

| Gate / request | Outcome |
|---|---|
| Literature gate | **Passed** (web search, 25–26 Oct; `docs/GATE0_LITERATURE.md`). No paper found that conformalizes monocular depth or depth completion by splitting one image's sparse measurements into a fitting set and a calibration set. The statistics are known (conformalized matrix completion, Gui, Barber & Ma 2023; Mao, Martin & Reich's within-field argument), so no theoretical novelty is claimed. Must be repeated on arXiv and Google Scholar before submission `[VERIFY]`. |
| Dual-anchor method and CPU method gate | **Passed.** A-dual (§6.11) is 16–69 % narrower than A-dist-both (median over the six NYU/KITTI pairs) and at least 10 % narrower on 6 of 6 pairs in every layout and at both anchor counts. Its image-averaged coverage is 0.902–0.907 against the exact theoretical value 19/21 = 0.905, and its spread across images matches the theoretical Beta law to three decimals (§9.22). A-dual-both, which adds the guarantee by distance to the nearest anchor, passes in 8 of 10 layout × k cells; clustered anchors are the exception (2–3 % narrower). |
| Independent sensor ground truth | **Done on ARKitScenes** (5,594 frames, iPad depth as the sensor, Faro laser depth as ground truth; §5, §9.23). **ScanNet++ not done** (needs a signed application). |
| Depth-completion comparison and completion gate | **Not done.** OMNI-DC was chosen and vetted (zero-shot on all our datasets, accepts scattered sparse depth, BSD licence) but cut for time on 1 November. This is the largest open item (§C). |
| Paper gate | Plan with three contributions and the main-text / supplement split in §F. Paper draft not started. |

---

## B. Item-by-item response

| Review request | Status | Where |
|---|---|---|
| Split the sparse-depth budget: structured prediction anchors build the predictor, random calibration probes set a within-image threshold | Done: **A-dual** | §6.11, §7.6 (T13) |
| Probe values must not enter the predictor or the scale map | Done by construction: m(x) and s(x) use only the structured anchors; the shrinkage of s (γ) uses only s. A unit test checks that changing probe values leaves m and s unchanged; a Monte Carlo test checks per-image validity under clustered prediction anchors for γ ∈ {1, 0.5, 0} | `src/calibrators/dual.py`, `tests/test_dual.py` |
| One method rather than a calibrator zoo | A-dual for the per-image guarantee; **A-dual-both** (hull with A-dist-split) when the by-distance guarantee is also wanted. A-dual-both replaces A-dist-both | §6.11, §9.22 |
| A per-image guarantee without the hull penalty | A-dual has no hull and is 16–69 % narrower than A-dist-both. A-dual-both brings back a hull, but with A-dist-split instead of A-local, and is 2–60 % narrower than A-dist-both | §9.22 |
| An explicit sensor-budget allocation problem | Probe budgets 10 / 20 / 40. The per-image coverage of one probe draw follows Beta(r, n + 1 − r) (T13): 10th percentile measured 0.796 / 0.820 / 0.841, predicted 0.795 / 0.819 / 0.841; share of short images measured 15.2 / 11.4 / 6.5 %, predicted 15.3 / 11.5 / 6.6 %. With 10 probes A-dual is the only finite per-image method; with 40 its width advantage shrinks | §7.6, §9.22 |
| Applicability to depth-completion networks | T13 holds for any centre and scale built from the structured anchors only, so a completion network can replace A-dist unchanged. **Not tested empirically** | §7.6, Limitation 14 |
| Kill criterion: ≥ 10 % narrower than A-dist-both on 4 of 6 pairs, per-image validity intact | Passed: A-dual on all 10 layout × k cells (6/6 pairs each); A-dual-both on 8/10. Per-image validity: coverage equals the theoretical value; the spread is the one theory predicts | §9.22 |
| Compare against modern depth completion (Prompt Depth Anything / PromptDA++, OMNI-DC, optionally Marigold-DC), conformalized with identical anchors, splits and targets | **Not done.** OMNI-DC chosen. PromptDA rejected as the primary baseline: trained on ARKitScenes and ScanNet++, so not zero-shot there, and its prompt is a dense low-resolution LiDAR map rather than scattered points; PromptDA++ was not released at the time of the search `[VERIFY]` | `docs/GATE0_LITERATURE.md`; §C; Limitation 14 |
| Describe the anchor setting as sparse-guided monocular depth or depth completion | Done in this document and in the paper plan (§F) | §F |
| Independent sensor ground truth (ARKitScenes or ScanNet++) | ARKitScenes: base accuracy (E0), real iPad depth as anchors (E16) and A-dual / A-dual-both (E17), three models each | §5, §9.23 |
| Completion gate: ≥ 15 % lower valid interval score than conformalized completion, or distance-wise coverage spread from > 5 to < 1.5 points | Not testable without a completion model. For reference against A-dist-both: A-dual-both's interval score is 1–46 % lower on NYU/KITTI (median over pairs) in every layout. Coverage spread across distance bins holding at least 1 % of pixels (median over pairs): A-dual-both 1.3–2.2 points, A-dist-both 1.3–2.9, A-dist-split 0.3–1.4, A-dual alone 3.9–12.3 | §9.22 |
| Literature gate | Passed, web-only | `docs/GATE0_LITERATURE.md`, §G items 6 and 17 |
| Do not claim "first" | Kept | §F |
| Far-distance bins: report pixel mass; output abstention or infinite intervals rather than hiding unsupported bins by merging | Pixel mass is now reported for every bin, layout and dataset (§9.22, §9.23). **Abstention not implemented.** A-dual's per-image guarantee includes far pixels, in expectation; the farthest bin holds at most 2.0 % of pixels, and its coverage is low only on KITTI, where it holds at most 0.32 % of pixels | §9.22, Limitation 11 |
| Paper: at most three contributions; main text = scale-error motivation, failure of structured A-local, dual-anchor method, completion comparison, independent-sensor evaluation; E5, E8, E10, score/calibrator ablations, most of E15 and E12 to the supplement | Plan in §F. Without the completion comparison, the main text has four of the five requested parts | §F |
| Not to prioritise: more models or splits, a planner, active placement as the novelty, unknown-layout robustness as the centre | Followed: no model, split or planner added since the review | – |

---

## C. Not done, and why

- **Depth-completion baseline.** A completion model needs one forward pass per image and anchor set; it cannot be cached per image like a frozen monocular model. A minimal grid is about 23,000 passes. ARKitScenes caching took priority, and writing started on 1 November, so the baseline was cut on that day. As a result, the review's question "if you have LiDAR, why monocular depth?" is answered by argument only: A-dual's guarantee does not depend on how the centre is built, so the calibration layer applies to a completion network unchanged. Whether it stays *necessary and efficient* on top of a strong completion centre is untested. The smallest useful run, if it fits before the supplement deadline (23 November), is OMNI-DC on ARKitScenes with two layouts at k = 200, conformalized three ways: split conformal across images, A-dist-style binned calibration, and A-dual.
- **ScanNet++.** Access needs an application with a supervisor's signature; not obtained.
- **A-dual with real-sensor probes.** E17 on ARKitScenes gives anchors and probes their laser values; E16 uses iPad values but does not include A-dual. By T13, probes that carry the iPad's 1.6 % offset would make A-dual cover the iPad's reading rather than the truth, as A-local and LSCP do in E16 (§9.23, Limitation 15).
- **A-dual on ETH3D and DIODE.** Planned, because A-dual needs no calibration images and those datasets have too few calibration scenes. Only a local smoke test on ETH3D was run.
- **Depth Pro on ARKitScenes.** Its AbsRel of 4.08 was checked inside our pipeline but not against Apple's own `depth_pro` package `[VERIFY]` (§9.23).

---

## 5. Data: ARKitScenes (addition to §5)

| Dataset | Units | Images | Depth sensor |
|---|---|---|---|
| **ARKitScenes**, depth-upsampling validation split | 107 visits (rooms) | 5,594 | Ground truth: Faro laser scan rendered into each frame; sensor: iPad Pro LiDAR depth as delivered by ARKit, with confidence 0/1/2 |

287 videos list 5,598 frames; one video ships without confidence maps and its 5 frames are dropped (`src/data/arkitscenes.py`). Frames are reduced 3× to the 640 × 480 grid of the released VGA intrinsics; ground truth by the lower median of each 3 × 3 block (as for ETH3D), the iPad depth by its nearest low-resolution pixel. Frames are rotated upright from the metadata's sky direction (102 videos become portrait). Depth beyond 10 m is ignored. Valid pixels are kept every 8th pixel, about 4,300 per frame and 23.8 million per model. The visit is the exchangeable unit: per split about 1,960 head-train, 1,704 calibration and 1,933 test images.

**The iPad depth is an independent sensor.** Over all table pixels its median absolute log error against the laser is 0.016 (90th percentile 0.036); at the drawn anchors, the mean signed error is −0.0155 and the median absolute error 0.0164, so almost every anchor reads nearer than the laser, by about 1.6 %. 1.6–1.7 % of anchors are off by more than 10 %. ARKit marks 94.6 % of pixels with confidence 2. Unlike KITTI, where the ground truth is accumulated from the same LiDAR, this tests a real sensor against an independent truth. Two caveats: ARKit's depth is a fused, smoothed product of LiDAR and RGB, not raw returns; and it covers every pixel, so anchor positions are not restricted the way single-scan LiDAR restricts them on KITTI.

**Training overlap.** UniDepthV2's training data appear to include ARKitScenes, so its ARKitScenes results may not be zero-shot `[VERIFY]`; the DA-V2 indoor metric model is fine-tuned on Hypersim only; Depth Pro's training mixture must be checked `[VERIFY]`. No guarantee depends on this; base accuracy and widths do.

---

## 6.11 Dual-role anchors: A-dual and A-dual-both (E17)

On a budget-split draw (§6.8), k − n anchors follow the layout and n are drawn uniformly at random among the remaining valid pixels (n = 20 unless stated). E14 used the 20 random anchors for A-local-split: 10 to estimate a constant shift, 10 to calibrate. A-dual gives the two kinds different roles:

- **Prediction anchors** (the k − n structured ones) alone build A-dist's local shift m(x) and scale s(x) (§6.8): the variant chosen per draw on head-train (A-dist-auto), bandwidths, τ, g and the cap fitted on head-train with the same layout; δ(x) is the distance to the nearest structured anchor.
- **Probes** (the n random ones) are never used by m or s. Per image, the threshold is q_i = the ⌈(n + 1)(1 − α)⌉-th smallest probe score |d_j − m(x_j)| / s_γ(x_j).
- **Interval:** pred · exp(m(x) ∓ q_i s_γ(x)) on every pixel that is neither an anchor nor a probe.
- **Shrunk scale:** s_γ = med_i · (s / med_i)^γ, where med_i is the median of s over all pixels of image i, and γ = 0.5. Without it (γ = 1), a draw whose structured anchors set the shift wrong everywhere leaves a nearly flat residual; q_i is then set by the probes with the smallest s, and widths elsewhere grow by up to max s / min s. One clustered NYU draw reached a mean relative width of 5,202 (§9.22). γ was chosen on NYU / Depth Pro after that run and then fixed for every other pair and for ARKitScenes. It uses no probe value, so validity holds for any γ.
- **A-dual-both** is the hull of A-dual and A-dist-split (A-dist-auto on all k anchors of the same draw).

Calibration images are not needed for A-dual's validity; head-train images only tune m and s. A-dual is computed on the plain log base only: with a learned base the threshold became unstable, as for A-local (§9.11).

Settings: the six NYU/KITTI pairs × five layouts (uniform, rows, cluster, edges, near; §6.7), k ∈ {50, 200} (30 or 180 structured anchors plus 20 probes), 3 seeds × 2 draws, α = 0.1. Probe ablation: n ∈ {10, 40} on the same grid. ARKitScenes: three models, the same grid with n = 20. Kill criterion, taken from the review before the run: at least 10 % narrower than A-dist-both on 4 of 6 pairs while per-image validity holds.

Code: `src/calibrators/dual.py`; `experiments/e11_anchors.py --dual --dual-gamma`; `scripts/run_new.sh` with `PARTS=e17` (`E17_PROBES`, `E17_GAMMA`, `E17_PAIRS`). Tests: `tests/test_dual.py`.

---

## 7.6 Theory: dual-role anchors

**T13 – A-dual (per image, any prediction-anchor layout).** Fix image i, its finite set P_i of valid ground-truth pixels and its structured anchor set S ⊂ P_i, chosen by any rule (scan lines, a cluster, edges, the nearest pixels; the rule may depend on the image). Let m(x) and s(x) > 0 be any functions of position computed from the image, the prediction and the values at S only. Draw n probes uniformly at random without replacement from P_i \ S, let r_j = |d_j − m(x_j)| / s(x_j), and let q̂_i be the r-th smallest probe score, r = ⌈(n + 1)(1 − α)⌉ (infinite if r > n). For a uniformly random pixel J of P_i outside S and the probes,

  P(r_J ≤ q̂_i | image i, S) ≥ r / (n + 1) ≥ 1 − α  (equality when scores have no ties).

*Why:* given S, m and s are fixed functions, so every pixel of P_i \ S carries a fixed score. The n probes together with J form a uniformly random (n + 1)-subset of P_i \ S with J in a uniformly random position, so the rank of r_J among the n + 1 scores is uniform: the argument of T8, with a score that the structured anchors fix in advance. It needs no calibration images. Unlike T8, no random anchor is spent on the shift: all n probes calibrate, and the centre can use any layout. Feasibility: n ≥ 9 at α = 0.1 (A-local needs k ≥ 18). This is the split argument of conformalized matrix completion (Gui, Barber & Ma 2023: a fitting set and a calibration set of observed entries), applied within one image with a frozen monocular prior; the theory is not new.

*Per-image spread: the probe budget.* Given i and S, the coverage C_i of one probe draw over the image's remaining pixels is approximately Beta(r, n + 1 − r) when the image has many pixels and no ties. This is the classical distribution of the coverage of a split-conformal quantile conditional on the calibration set (Vovk 2012 `[VERIFY]`), here with a finite population of a few thousand pixels. The probe count therefore sets how often one image falls short:

| Probes n | r | Mean r/(n+1) | 10th percentile of C_i | Share of images below 0.85 (averaged over 2 draws) |
|---|---|---|---|---|
| 10 | 10 | 0.909 | 0.795 | 15.3 % |
| 20 | 19 | 0.905 | 0.819 | 11.5 % |
| 40 | 37 | 0.902 | 0.841 | 6.6 % |

(Monte Carlo over the Beta law; §9.22 compares with the measurements.)

*Scope:* (a) The promise is in expectation over the probe draw, not conditional on one draw. (b) It covers the values the probes report: probes from a biased sensor make A-dual cover the sensor's reading, as for T8 (§9.21, §9.23). (c) s_γ depends on s only, so it is as fixed as s given S; any γ is valid. (d) Nothing is promised about *where* in the image the misses fall: coverage can fall far from the anchors (§9.22). (e) Probes must be uniform among the pixels the guarantee is about; a sensor that cannot return on glass or sky restricts the guarantee to pixels where it can.

**T14 – A-dual-both.** The hull of A-dual and A-dist-split contains each pointwise, so by T9 it inherits T13 (per image, in expectation over probes) and T10 (per distance bin, on average over scenes). It replaces T11's A-dist-both, whose per-image half was A-local on the 20 random anchors.

---

## 9.22 Dual-role anchors (E17, 3 seeds × 2 draws)

**Finding 1 – shrinking the scale removed a heavy width tail without changing validity.** With the plain A-dist scale (γ = 1, job 44976), A-dual's median width was already 12–71 % below A-dist-both, but single runs blew up: a mean relative width of 5,202 on one clustered NYU / DA-V2 draw and 34.7 on a KITTI near-layout draw. Under clustered anchors the gate held on only 3 of 6 pairs. With γ = 0.5 (§6.11), the worst single-run mean width is 0.55, and the gate holds on 6 of 6 pairs in every layout and k. Against γ = 1, the median width changes by −3 % to +7 % and the mean width falls by 1–28 % (median over pairs, per layout and k). Coverage is unchanged (0.902–0.907), as T13 requires.

**Finding 2 – A-dual passes the kill criterion everywhere; A-dual-both in 8 of 10 cells.** Settings of §6.11; widths are median relative widths, as percentage changes against A-dist-both (median over the six pairs; range over pairs in brackets):

| Layout | k | A-dual coverage | A-dual width | A-dual-both width | Pairs ≥ 10 % narrower (A-dual-both) | A-dual-both interval score | Coverage A-dist-both / A-dual-both |
|---|---|---|---|---|---|---|---|
| uniform | 50 | 0.904–0.907 | −48 % | −37 % [−42, −24] | 6/6 | −28 % | 0.970 / 0.952 |
| uniform | 200 | 0.904–0.907 | −69 % | −60 % [−67, −46] | 6/6 | −46 % | 0.976 / 0.947 |
| rows | 50 | 0.903–0.905 | −41 % | −30 % [−34, −20] | 6/6 | −24 % | 0.970 / 0.956 |
| rows | 200 | 0.902–0.905 | −53 % | −44 % [−47, −32] | 6/6 | −32 % | 0.973 / 0.952 |
| cluster | 50 | 0.905–0.907 | −16 % | −2 % [−4, 0] | 0/6 | −1 % | 0.969 / 0.964 |
| cluster | 200 | 0.903–0.905 | −17 % | −3 % [−6, 0] | 0/6 | −4 % | 0.969 / 0.963 |
| edges | 50 | 0.904–0.906 | −24 % | −12 % [−16, −3] | 4/6 | −15 % | 0.970 / 0.959 |
| edges | 200 | 0.903–0.905 | −34 % | −23 % [−28, −14] | 6/6 | −18 % | 0.979 / 0.960 |
| near | 50 | 0.904–0.906 | −33 % | −18 % [−30, −11] | 6/6 | −18 % | 0.970 / 0.963 |
| near | 200 | 0.903–0.906 | −37 % | −22 % [−38, −12] | 6/6 | −20 % | 0.973 / 0.962 |

A-dual is at least 10 % narrower than A-dist-both on every pair in every cell. A-dual-both's interval score is lower than A-dist-both's on every pair in every cell (range over pairs −1 % to −48 %). Means over pairs and layouts:

| Method | Guarantee | Median width, k = 50 / 200 | Coverage | Short images |
|---|---|---|---|---|
| A-dist-split | per distance bin, average over scenes; needs calibration images | 0.154 / 0.133 | 0.920 | 9.8 / 8.4 % |
| A-dual | per image; no calibration images | 0.187 / 0.161 | 0.905 | 11.4 / 11.6 % |
| A-dual-both | both | 0.225 / 0.195 | 0.959 / 0.957 | 0.1 % |
| A-dist-both (E14) | both | 0.280 / 0.281 | 0.970 / 0.974 | 0.1 / 0.0 % |
| A-local-split (E14) | per image | 0.244 / 0.243 | 0.909 | 15.3 % |

Under clustered anchors A-dual-both is barely narrower than A-dist-both, although both of its halves are narrower (median width 0.18 for A-dist-split and 0.24 for A-dual, against 0.29 for the hull). The two halves are centred differently: A-dual's centre comes from the cluster alone, A-dist-split's from the cluster plus the probes. Their hull is therefore much wider than either half.

**Finding 3 – the per-image spread is the one T13 predicts.** A-dual's share of short images (11 %) is much larger than A-dist-both's (0.1 %). This is not a failure: it is what a per-image guarantee from 20 probes implies. Measured (mean over pairs and layouts, k = 50) against the Beta law of §7.6:

| Probes | Coverage, measured / theory | 10th percentile of per-image coverage, measured / theory | Short images, measured / theory |
|---|---|---|---|
| 10 | 0.909 / 0.909 | 0.796 / 0.795 | 15.2 % / 15.3 % |
| 20 | 0.905 / 0.905 | 0.820 / 0.819 | 11.4 % / 11.5 % |
| 40 | 0.903 / 0.902 | 0.841 / 0.841 | 6.5 % / 6.6 % |

On ARKitScenes (20 probes, every model, layout and k) the same quantities are 0.904–0.906, 0.816–0.821 and 10.5–12.1 %. A-dist-both's near-zero short share comes from over-coverage (0.97 for a 0.90 target), not from a stronger guarantee; A-dual-both's 0.1 % comes from the hull.

**Finding 4 – the probe budget is a real trade-off.** With 10 probes, A-local-split, and hence A-dist-both, is infinite on every image (5 calibration anchors, fewer than the 9 needed), while A-dual is finite (r = 10 ≤ 10), with median width 0.202 / 0.178 at k = 50 / 200: the only finite per-image method at that budget. With 40 probes, fewer structured anchors remain for the centre and A-dist-both itself narrows (0.248 at both k). A-dual's width barely changes (0.199 / 0.154), so the gate fails under clustered anchors (2/6 and 3/6 pairs) and for edges at k = 50 (2/6); its short share halves to 6.5–6.8 %. Twenty probes is the default: it passes everywhere and keeps 11 % of images short.

**Finding 5 – coverage by distance, and the far bins the review asked about.** Distances here are to the nearest of all anchors, probes included, so far bins are smaller than in E14 (structured anchors only). A-dual's coverage falls with distance from the nearest anchor (near layout, k = 50: 0.934 within 0.02 of the image diagonal, 0.81 at 0.1–0.2, a bin holding 2–16 % of pixels). T13 allows this: its guarantee is per image, not per region. A-dual-both restores near-flat coverage: 0.93–0.97 in every bin up to 0.2 under every layout. Coverage spread across bins holding at least 1 % of pixels, in points (median over pairs, max in brackets):

| Layout, k | A-dist-split | A-dual | A-dual-both | A-dist-both |
|---|---|---|---|---|
| uniform 50 / 200 | 0.5 / 0.3 | 7.3 / 6.2 | 1.6 / 1.3 (max 2.1) | 1.8 / 1.3 |
| rows 50 / 200 | 0.3 / 0.4 | 6.0 / 8.5 | 1.4 / 1.3 (1.8) | 1.7 / 1.9 |
| cluster 50 / 200 | 0.7 / 0.5 | 8.3 / 9.0 | 2.2 / 2.2 (2.9) | 2.7 / 2.9 |
| edges 50 / 200 | 1.0 / 1.4 | 4.1 / 3.9 | 1.2 / 1.5 (2.1) | 1.4 / 1.6 |
| near 50 / 200 | 0.4 / 0.4 | 12.3 / 9.5 | 1.5 / 1.5 (3.4) | 2.1 / 1.5 |

Pixel mass of the farthest bin (≥ 0.2 of the image diagonal from any anchor or probe), in % of test pixels, k = 50 / 200:

| Layout | NYU | KITTI | ARKitScenes |
|---|---|---|---|
| uniform | 0.075 / 0.000 | 0.018 / 0.001 | 0.074 / 0.001 |
| rows | 0.35 / 0.22 | 0.020 / 0.001 | 0.42 / 0.29 |
| cluster | 2.04 / 1.88 | 0.32 / 0.26 | 1.96 / 1.67 |
| edges | 0.45 / 0.17 | 0.093 / 0.025 | 0.60 / 0.29 |
| near | 0.86 / 0.78 | 0.010 / 0.001 | 0.80 / 0.59 |

In that bin A-dual-both covers 0.93–0.96 on NYU and on ARKitScenes wherever the bin holds more than 0.01 % of pixels. On KITTI it covers 0.75–0.92 where the bin holds at least 0.018 % of pixels, and 0.44–0.88 where it holds 0.01 % or less. A-dist-split alone covers 0.44–0.91 there on KITTI. Bins merged by A-dist are therefore not hiding a large mass of uncovered pixels; abstention was not implemented (Limitation 11).

---

## 9.23 An independent sensor: ARKitScenes (E0, E16, E17)

**Base accuracy (E0, all 5,594 frames).**

| Model | AbsRel | δ1 | Median predicted / laser depth |
|---|---|---|---|
| UniDepthV2 | 0.120 | 0.882 | 1.06 |
| Depth Anything V2 indoor (metric) | 0.466 | 0.325 | 1.36 |
| Depth Pro | 4.08 `[VERIFY]` | 0.032 | 2.68 |

DA-V2 overestimates depth on these close-range indoor frames. Depth Pro is far off in scale: its field-of-view head estimates about 48° where the true value is 62°, which explains only a factor of 1.33. The rest comes from the depth output itself, and is worst on close-ups (11.7× on a frame with a median depth of 0.69 m, where DA-V2 also struggles). Float32 and float16 give the same numbers, and the iPad depth agrees with the laser on the same frames (ratio 0.98–1.00), so ground truth and rotation are correct. Not yet checked against Apple's own `depth_pro` package `[VERIFY]`. For the anchor experiments, these per-image scale errors are exactly what anchors are meant to remove.

**E16 – real iPad depth as anchors (uniform layout, k ∈ {20, 200}, 3 seeds × 2 draws).** Two sources on the same anchor positions: *clean* (laser values) and *iPad* (the sensor's values); matched calibration and @clean as in §6.10. Range over the three models:

| Method | Coverage, clean → iPad, k = 200 | Median width, clean → iPad, k = 200 | Short images, clean → iPad, k = 200 |
|---|---|---|---|
| A-rescale | 0.925–0.928 → 0.925–0.929 | 0.145–0.644 → 0.153–0.638 | 15.5–20.6 % → 15.5–20.6 % |
| A-hybrid | 0.924–0.927 → 0.924–0.928 | 0.102–0.315 → 0.116–0.317 | 13.7–19.8 % → 13.0–19.0 % |
| A-dist | 0.923–0.925 → 0.925–0.929 | 0.025–0.045 → 0.052–0.069 | 6.6–8.4 % → 7.9–8.9 % |
| A-dist@clean | – → **0.486–0.891** | – → 0.026–0.043 | – → **19.7–98.1 %** |
| A-local | 0.901 → **0.854–0.903** | 0.086–0.260 → 0.094–0.259 | 1.2–1.6 % → 5.9–35.0 % |
| LSCP | 0.940–0.959 → **0.548–0.888** | 0.028–0.058 → 0.035–0.058 | 0 % → 19.9–94.0 % |
| A-both | 0.948–0.953 → 0.945–0.955 | 0.114–0.359 → 0.126–0.357 | 0.1–0.3 % → 0.6–5.3 % |
| A-dist-both | 0.975–0.984 → 0.969–0.986 | 0.092–0.266 → 0.109–0.267 | 0 % → 0.2–2.3 % |

The iPad source behaves like the synthetic 5 % bias of §9.21, at a third of its size:

- *Matched calibration keeps every cross-image guarantee* (0.924–0.929). The offset costs width; A-dist pays most (17–167 % wider) because it trusts its nearest anchors, yet stays 41–84 % narrower than A-hybrid.
- *Calibrating on clean anchors and deploying the iPad is unsafe for A-dist:* 0.49 on UniDepthV2 and 0.58 on DA-V2 at k = 200 (0.85–0.92 at k = 20). It fails next to the anchors, where it trusts them most: 0.57 within 0.02 of the diagonal and 0.67 at 0.02–0.05, against 0.90 at 0.1–0.2 (mean over models). Its clean half-width there, about 0.013 in log units, is smaller than the 0.016 offset. A-rescale@clean and A-hybrid@clean stay at 0.914–0.928.
- *Within-image methods cover the sensor's reading.* A-local drops to 0.854 on UniDepthV2 and 0.880 on DA-V2; LSCP to 0.548 and 0.661 at k = 200 (0.73–0.83 at k = 20). Depth Pro is barely affected (A-local 0.903, LSCP 0.888, A-dist@clean 0.891), because its intervals are wide enough to absorb a 1.6 % offset.
- *The hulls keep coverage* (A-both 0.945–0.955, A-dist-both 0.969–0.986), but their per-image half is A-local, so the per-image guarantee no longer holds on the truth: up to 5.3 % of images short for A-both and 2.3 % for A-dist-both.

This confirms §9.21's recipe with a real phone sensor and an independent ground truth: calibrate with the deployed sensor, and read within-image intervals as covering the sensor's reading.

**E17 – A-dual and A-dual-both (settings of §6.11, laser values for anchors and probes).** Median width against A-dist-both, per model (DA-V2 / Depth Pro / UniDepthV2):

| Layout | k | A-dual width | A-dual-both width | Models ≥ 10 % narrower (A-dual-both) | A-dual-both interval score |
|---|---|---|---|---|---|
| uniform | 50 | −64 / −75 / −59 % | −55 / −67 / −46 % | 3/3 | −46 / −81 / −37 % |
| uniform | 200 | −80 / −87 / −76 % | −75 / −82 / −69 % | 3/3 | −63 / −90 / −54 % |
| rows | 50 | −53 / −67 / −46 % | −44 / −55 / −35 % | 3/3 | −36 / −72 / −29 % |
| rows | 200 | −61 / −73 / −56 % | −54 / −61 / −46 % | 3/3 | −44 / −75 / −37 % |
| cluster | 50 | −11 / −16 / −15 % | 0 / −1 / −2 % | 0/3 | +2 / **+24** / −3 % |
| cluster | 200 | −16 / −23 / −17 % | −6 / −8 / −4 % | 0/3 | −3 / **+19** / −6 % |
| edges | 50 | −25 / −53 / −17 % | −18 / −46 / −4 % | 2/3 | −18 / −66 / −10 % |
| edges | 200 | −36 / −63 / −34 % | −29 / −54 / −17 % | 3/3 | −22 / −71 / −11 % |
| near | 50 | −28 / −20 / −25 % | −21 / −9 / −16 % | 2/3 | −16 / **+70** / −17 % |
| near | 200 | −34 / −36 / −30 % | −27 / −15 / −21 % | 3/3 | −5 / **+94** / −15 % |

A-dual covers 0.904–0.906 in every cell (theory 0.905), and is never infinite. A-dual-both covers 0.949–0.980, A-dist-both 0.974–0.989. Means over models and layouts, k = 50 / 200: median width A-dist-split 0.074 / 0.066, A-dual 0.115 / 0.097, A-dual-both 0.135 / 0.118, A-dist-both 0.193 / 0.194. A-dual-both is 30 % narrower than A-dist-both at k = 50, against 20 % on NYU/KITTI. A-dual-both covers 0.93–0.97 in every distance bin; A-dual alone falls to 0.70 far from anchors, as on NYU/KITTI. The scale fix (γ = 0.5) was chosen before ARKitScenes was run, so this is a held-out check of the method as frozen on NYU/KITTI.

**Caveat – Depth Pro with clustered or near anchors.** In those layouts A-dual-both is narrower in median (−1 to −15 %) but has a worse interval score (+19 to +94 %) and a larger mean width (0.61–0.63 against 0.55–0.56 under near). A-dual alone already scores 3.2–5.2 against about 2.6 for A-dist-both, with a mean lower-miss of 0.05–0.07 against 0.01: its misses, when they happen, are large. Our reading, not separated experimentally: with anchors in one region and Depth Pro's large, uneven scale errors on ARKitScenes, the centre built from those anchors extrapolates poorly to the rest of the frame. The probes widen q_i accordingly, but a single per-image threshold times a shrunk scale cannot follow where the error is. The other two models do not show this. It ties to the unverified Depth Pro numbers above.

---

## E. Statements of the 25 October report superseded here

- **Contribution 6 / T11 / §9.17:** A-dist-both as the way to add a per-image guarantee is replaced by A-dual-both (§6.11, T14): 2–60 % narrower, with the same pair of guarantees.
- **Limitation 3** (random anchors are an assumption): for the per-image guarantee, it now suffices to add n ≥ 9 uniformly random probes to any structured layout (T13). The cross-image guarantees still need the calibration layout to match deployment (E15).
- **Limitation 11** (A-dist's far bins): far-bin pixel mass is now reported (§9.22); see the updated text below.
- **Limitation 13 and Future work (1)** (no independent sensor; KITTI is circular): ARKitScenes provides one (§9.23); see the updated text below.
- **Future work (4)** (anchors on ETH3D): still open, now with A-dual as the candidate.

**Updated limitations.**

11. *(updated)* **A-dist's guarantee is per distance bin, not per pixel or per image.** Sparse bins are still merged rather than abstained on. With distances measured to all anchors including probes, the farthest bin holds at most 2.0 % of test pixels (NYU, clustered anchors) and under 0.1 % with uniform anchors; there A-dual-both covers 0.93–0.96 except on KITTI, where the bin holds at most 0.32 % of pixels and coverage falls to 0.44–0.92 (§9.22). An abstaining or infinite-interval variant for unsupported bins was not implemented.
13. *(updated)* **Anchor errors.** ARKitScenes provides a real sensor checked against an independent laser ground truth. The iPad depth is a fused ARKit product that covers every pixel, so it does not test sparse returns, dropouts on glass, or correlated scan-line errors. E17 on ARKitScenes uses laser values for anchors and probes. Correlated sensor errors remain untested.
14. *(new)* **No depth-completion baseline.** No completion network (OMNI-DC, PromptDA, Marigold-DC) was conformalized, so it is not shown that the calibration layer stays necessary and efficient on top of a strong completion centre, and the review's completion gate was not run. A-dist is a lightweight local interpolation from sparse metric measurements; the setting should be read as sparse-guided monocular depth.
15. *(new)* **A-dual's guarantee is per image only in expectation over the probe draw, and it covers the probes' values.** With 20 probes, 11–12 % of images fall more than 5 points short in a given pair of draws, exactly as the Beta law predicts; more probes reduce this (6.6 % with 40) at the cost of fewer structured anchors. Probes from a biased sensor make A-dual cover the sensor's reading: with the iPad's 1.6 % offset and A-dual's narrow intervals (median half-width about 0.014 in log units for UniDepthV2 and DA-V2 at k = 200), we expect a loss of coverage like A-local's and LSCP's in E16. A-dual was not run with iPad probes. A-dual-both's A-dist-split half, calibrated with the same sensor, keeps its cross-image guarantee under such an offset (E16).
16. *(new)* **The scale shrinkage γ was chosen after a first run.** γ = 0.5 was picked on NYU / Depth Pro after job 44976 showed the width tail, then fixed. Validity does not depend on γ (T13), but the NYU/KITTI width comparisons were made on the data used to choose it; ARKitScenes is the held-out check.
17. *(new)* **Depth Pro on ARKitScenes is unverified** (AbsRel 4.08, §9.23), and under clustered or near anchors A-dual-both's interval score is worse than A-dist-both's for that model.
18. *(new)* **Possible training overlap on ARKitScenes** (UniDepthV2; Depth Pro to check) `[VERIFY]`. It does not affect any guarantee, only accuracy and widths.
19. *(new)* **Scope of E17.** One α (0.1), the plain log base only, k ∈ {50, 200}, one probe budget on ARKitScenes, no run on ETH3D or DIODE, and 3 seeds × 2 draws per cell.

**Reproducibility of this version.** The E17 and ARKitScenes results record `"git": "uncommitted"` with code hashes `9b2cef1db6cef1f0` (job 44976), `6d17e1aaf946f05d` (job 46095) and `d08fe3d97db43f9a` (jobs 46165, 46166, 46203). The code of this week (`src/calibrators/dual.py`, `src/data/arkitscenes.py`, `scripts/fetch_arkitscenes.py`, `tests/test_dual.py` and changes to 14 existing files) is not yet committed; it will be committed and tagged `v1.2-report` before the paper. Commands:

- E17: `PARTS=e17 bash scripts/run_new.sh` (20 probes, γ = 0.5; results in `results/e17_p20_g05/`); probe ablation with `E17_PROBES="10 40"`; ARKitScenes with `E17_PAIRS="arkitscenes:dav2-indoor-large arkitscenes:unidepthv2-large arkitscenes:depthpro"`.
- E16 on ARKitScenes: `PARTS=e16 E16_EXP=e16_arkit E16_SOURCES="clean lidar" E16_PAIRS="arkitscenes:…"` (results in `results/e16_arkit/`).
- ARKitScenes data: `scripts/fetch_arkitscenes.py` (validation split, about 15 GB), predictions with `scripts/cache_predictions.py`, tables with `experiments/prepare_tables.py` (stride 8).

---

## F. Plan for the paper (paper gate)

**Framing:** *sparse-guided monocular depth.* A frozen monocular foundation model supplies a dense prior, sparse metric measurements correct it, and conformal calibration states what remains uncertain. No claim of being first; the statistics are credited to Gui, Barber & Ma and to Mao, Martin & Reich.

**Three contributions.**

1. **Why per-image guarantees from sparse depth fail in practice.** Most per-image miscoverage is one global scale error per image (§9.10), which sparse depth can remove. But the within-image guarantee needs uniformly random anchors, and it fails silently under scan-line, clustered, edge and near-only layouts (§9.16).
2. **Dual-role sparse depth.** Structured anchors build the centre and scale, and a few independent random probes set a per-image conformal threshold. This gives a per-image guarantee for any anchor layout, without calibration images, whose spread across images follows an exact Beta law in the probe count, so the probe budget becomes a design choice. On three models and five layouts it meets its theoretical coverage and is 16–69 % narrower than the previous hull-based per-image method; its hull with distance-binned calibration adds a guarantee by distance to the nearest anchor at 2–60 % less width.
3. **Validation with an independent sensor.** On ARKitScenes (iPad depth against laser ground truth) the method behaves as on NYU and KITTI, and the iPad's 1.6 % offset shows the operational rule: calibrate with the deployed sensor, and read within-image intervals as covering the sensor's reading.

**Main text:** scale-error motivation and structured-anchor failure (E11, E13); A-dual, T13 and the probe budget (E17); ARKitScenes (E16, E17); one qualitative figure. A completion-model comparison would go here if it is run (§C). **Supplement:** the calibrator and score study (E2–E4, E6, E9), E5, E8, E10, E12, E14 in detail, most of E15, the synthetic E16 sources, Metric3D v2 and ETH3D.

**Draft abstract.** Monocular metric depth foundation models predict metres for every pixel but give no reliable statement of their error, and most of that error, per image, is a single global scale factor. A few sparse metric measurements per image remove much of it, but the within-image conformal guarantee that makes such measurements trustworthy requires them to be placed uniformly at random, which real sensors do not do: under scan-line, clustered or edge layouts the guarantee fails silently. We split the sparse-depth budget into two roles. Structured anchors, following the sensor's layout, build a locally adaptive centre and scale; a few uniformly random probes, never used by the predictor, set a per-image conformal threshold. The result is a per-image coverage guarantee for any anchor layout, with no calibration images, whose spread across images follows an exact Beta law in the number of probes. On NYU and KITTI with three frozen models and five layouts it meets its theoretical coverage and is 16–69 % narrower than a hull-based per-image baseline; combined with distance-binned calibration it adds a guarantee by distance to the nearest anchor. On ARKitScenes, with iPad depth checked against laser ground truth, the method behaves the same, and the sensor's 1.6 % offset shows that calibration must use the deployed sensor.

---

## G. `[VERIFY]` items added or still open

Still open from the 25 October report: items 6–12 of Appendix B (prior-art search before submission; several reference details; ETH3D zero-shot status). New:

13. Depth Pro on ARKitScenes: reproduce AbsRel 4.08 on about 20 frames with Apple's `depth_pro` package.
14. Training data of UniDepthV2 and Depth Pro: do they include ARKitScenes?
15. Vovk (2012): confirm the reference for the Beta distribution of conditional split-conformal coverage, or cite an equivalent textbook statement.
16. PromptDA++: confirm whether code and weights have been released since the 25 October search.
17. Repeat the Gate 0 literature search (conformal depth completion, within-image conformal sampling, dual-role sensing) on arXiv and Google Scholar before submission.

---

## New references and glossary entries

- Baruch, G., Chen, Z., Dehghan, A., et al. ARKitScenes: A Diverse Real-World Dataset for 3D Indoor Scene Understanding Using Mobile RGB-D Data. *NeurIPS Datasets and Benchmarks* 2021; arXiv 2111.08897.
- Lin, H., et al. Prompting Depth Anything for 4K Resolution Accurate Metric Depth Estimation. *CVPR* 2025 `[VERIFY: author list]`.
- Taguchi, et al. Uncertainty-Aware Interactive LiDAR Sampling for Deep Depth Completion. *WACV* 2023 `[VERIFY: author list]`.
- Vovk, V. Conditional validity of inductive conformal predictors. *ACML* 2012, PMLR 25 `[VERIFY]`.
- Yeshwanth, C., Liu, Y.-C., Nießner, M., Dai, A. ScanNet++: A High-Fidelity Dataset of 3D Indoor Scenes. *ICCV* 2023.
- Zuo, Y., et al. OMNI-DC: Highly Robust Depth Completion with Multiresolution Depth Integration. *ICCV* 2025 `[VERIFY: author list]`.

Already in the report: Gui, Barber & Ma (2023); Mao, Martin & Reich (2024); Wieczorek (2023).

| Term | Meaning |
|---|---|
| **A-dual** | structured anchors build A-dist's shift and scale; n random probes, never used by them, set one conformal threshold per image (§6.11, T13) |
| **A-dual-both** | hull of A-dual and A-dist-split: per-image and per-distance-bin guarantees (T14) |
| **Probe** | a uniformly random sparse-depth pixel used only to calibrate, never to predict |
| **Prediction anchor** | a structured sparse-depth pixel used to build the centre and scale |
| **γ (gamma)** | in A-dual, the shrinkage of the scale toward its per-image median (0.5) |
