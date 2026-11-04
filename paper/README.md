# Structured Anchors Predict, Random Probes Calibrate: A Per-Image Conformal Layer for Sparse-Guided Depth — CVPR 2027 submission draft (v4, 3 Nov 2026)

Deadlines: abstract registration **10 Nov 2026**, paper **16 Nov 2026 AoE**. **Author decisions and data needed: see `AUTHOR_ASKS.md`.**

## Build

```bash
cd paper
python3 tools/gen_numbers.py          # only after result CSVs change (stdlib only)
pdflatex main && bibtex main && pdflatex main && pdflatex main
pdflatex supplement && bibtex supplement && pdflatex supplement && pdflatex supplement
```

Official CVPR author kit `CVPR2027-v1` (github.com/cvpr-org/author-kit): `cvpr.sty` and `ieeenat_fullname.bst` are unchanged. Review mode: anonymous, with line numbers.

Status on 3 Nov (v4):
- `main.pdf` compiles with 0 errors, 0 undefined references and 0 overfull boxes. It has 10 pages. With the real Fig. 3 and the v4.2 trims, the Conclusion ends in p. 8's right column (line 593) and the references (53 entries) start just below it. That leaves **about 4 lines of slack** with the envelope in the main paper, and more with `\envelopemainfalse`.
- `supplement.pdf` compiles with 0 errors, 0 undefined references and 0 overfull boxes. It has 7 pages.
- Both envelope states compile clean. In the `\envelopemainfalse` state the main paper is still 10 pages with the text ending on p. 8, and the supplement has 7 pages.

## Envelope switch

The switch is `\envelopemaintrue` / `\envelopemainfalse`, the single switch in `macros.tex`. It is currently **true**, and it is the gate from `AUTHOR_ASKS.md` item 6: keep the envelope in the main paper only if truth coverage with iPad probes is ≥ 0.9 **and** the IS beats A-dist-both by ≥ 10 %.
- **true**: §4.6 contains the envelope and Prop. 2, plus the gate `\TODO`. Tab. 3b has the A-dual+env row, and §6.5 has the envelope placeholder.
- **false**: §4.6 becomes a one-sentence "Sensor-valued probes" pointer to Supp. D. The env row disappears, §6.5 ends with "did not restore truth coverage here", Limitations gains a sentence, and the pipeline caption points to Supp. D. Supp. D then holds the full envelope text plus the proof, with Proposition 2 keeping its number.
- The envelope text is written once, in `sections/envelope.tex`, and input by both documents.

## Section status

| File | State | Placeholders (what fills them) |
|---|---|---|
| `sections/abstract.tex` | fully written; E18/E19 results in words (v4 policy) | — (0.26–0.47 and 16–69 % in orange) |
| `sections/intro.tex` | fully written | — |
| `sections/related.tex` | fully written | — |
| `sections/setup.tex` | fully written | — |
| `sections/method.tex` | fully written; Tab. 1 single-draw only | §4.6 envelope behind `\ifenvelopemain` with gate `\TODO`; design-rule box: `\pending{measured, $r=n$ arm}` |
| `sections/envelope.tex` | envelope body (α_s = α_e = 0.05; one CRC over class-adaptive quantile bands, amended 4 Nov), shared by §4.6 and Supp. D | — |
| `sections/experiments.tex` | fully written: training facts, single-draw short share, primary IS and uninformative share, confirmatory protocol | — |
| `sections/results.tex` | fully written; Tab. 2 and Tab. 3b frozen-model rows (E18) exploratory; Tab. 3a and the OMNI-DC rows of Tab. 3b from the confirmatory E19c (except iPad-anchor OMNI-DC cells) | 2 boxes: confirmatory replication; envelope (not run). Tab. 2 short column and short-share text: `[confirmatory]`. Tab. 3: A-dual+env row. Fig. 3: real figure (`figures/e17_results.pdf`) |
| `sections/limitations.tex`, `conclusion.tex` | fully written | — |
| `figures/teaser.tex` | caption written | 4 placeholder panels (render) |
| `figures/pipeline.tex` | done | — |
| `supplement.tex` | proofs A–D, with D also holding the envelope body when the switch is off; A-dist details E, guarantees table F; data, training data of the models and E19 implementation G; E18 H; full E19 grid (Tab. S2), full E18 grid (Tab. S3), E17 ranges (Tab. S4) | the I "Additional results" box (confirmatory, r = n arm, γ ablation, Metric3D v2, envelope run, qualitative, E19 metrics) |
| `numbers.tex` | hand-written macros plus a **generated block** (E17 Tab. 2 and Tab. S4, single-draw short-share text macros, E19 Tab. 3 and Tab. S2, E18 Tab. 3b and Tab. S3, derived text numbers, design-rule theory values) | pending macros below; stale lines to delete by hand (`AUTHOR_ASKS.md` item 8) |
| `tools/gen_numbers.py` | rewrites the generated block from `results/e17_p20_g05/*/summary.csv` (and `results/e17_p40_g05` for n = 40), `results_omnidc/e19_conf_s*/*/runs.csv` (confirmatory E19c; the iPad-anchor OMNI-DC cells still from `e19_summary_3seeds.csv`) and `results/e18_sensor/arkitscenes__{unidepthv2-large,dav2-indoor-large}{,__cluster}/summary.csv`. Backward compatible: it reads `frac_img_short`, `rel_interval_score_imgmed` (primary; else `rel_interval_score_robust`) and `frac_uninformative` (E19 Uninf. column) when present, and otherwise falls back (short cells → `[confirmatory]`, IS → exploratory mean over images). E19 text macros are printed plain (confirmatory cells only), except `\NumIpadAnchorDual*` / `\NumIpadAnchorDist*` (`\prelim`) | point it at the confirmatory output paths |

## Draft markers (remove or switch off before submission)

- `\prelim{}` marks an exploratory number, shown in **orange**. The per-cell `\ENineteen` lookup (numbers.tex L167, hand part) still makes every E19 table cell orange; the fix is in AUTHOR_ASKS item 8. Set `\showprelimfalse` in `macros.tex` for black.
- `\pending{}` and `\placeholder{title}{text}` mark a still-running item, shown in red.
- `\verify{}` marks an unverified fact, shown in red. None are left.
- `\TODO{}` marks a writing task. Only the envelope gate is left.

## `\pending` macros and cells

| Item | Waits for |
|---|---|
| `\NumEnvTruthCov`, `\NumEnvISGain`, A-dual+env row | the envelope run (α_s = α_e = 0.05, decided) |
| `[confirmatory]`: E17/E18 short cells, `\NumESeventeenShort`, `\NumESeventeenShortForty`, `\NumArkShortOne`, `\NumEEighteenShort`, `\NumEEighteenShortII` | `frac_img_short` in the confirmatory summaries |
| `\pending{measured, $r=n$ arm}` (design-rule box) | the r = n = 20 arm of the confirmatory E17 |
| `[primary IS]` (2 exploratory iPad-anchor OMNI-DC cluster cells; Tab. S2 prints them as --) | not rerun by design (E19c uses laser anchors only) |
| `\NumOmniAbsRelUniform`, `\NumOmniAbsRelCluster` | AbsRel from the cached OMNI-DC outputs |
| Teaser panels | render (being produced; `AUTHOR_ASKS.md` item 9). Fig. 3 is done (`figures/e17_results.pdf`) |
| every orange number | confirmatory replication |

The 8-image OMNI-DC smoke numbers, the 400-image E18 smoke value and the exploding per-seed IS values appear **only as comments** in `numbers.tex`.

## `[VERIFY]` items

There is no `\verify` left in the text or in `refs.bib`: the training-data facts came from the authors in v4, and all references were checked against official pages on 3 Nov. The `[VERIFY]` tags left in `AUTHOR_ASKS.md` concern deadlines and policy only.

## Changes in v4 (author decisions)

- **Title**: option (b).
- **Abstract, intro and conclusion**: E18 and E19 results are stated in words. The results section keeps the orange numbers.
- **Short images are single-draw everywhere** (law: 17.6 % at n = 20, 13.0 % at n = 40).
  - Tab. 1 lost its two-draw column.
  - Tab. 2's theory row shows 17.6.
  - The two-draw law appears only in a Supp. B footnote.
  - The E17 and E18 short cells print `[confirmatory]` until `frac_img_short` exists.
  - The text-level short macros include their own `\%`.
- **Primary relative IS** (amended 3 Nov): per image, the mean over pixels of IS / y, then the median over images. The secondary metric is the share of pixels with width > 10 y. Both are defined in §5 and Supp. G. Tab. 3 and Tab. S3 note that exploratory values are means over images.
- **Envelope**: α_s = α_e = 0.05. Amended 4 Nov: one two-sided CRC picks a single λ for class-adaptive bands [Q_c(λ/2), Q_c(1−λ/2)], replacing per-class CRC, which was infeasible with 37 calibration scenes; see `AUTHOR_ASKS.md` item 6. The text sits behind the single switch (see above).
- **Training data**:
  - main §5: one sentence per model plus the safe wording;
  - new Supp. G paragraph: full lists for UniDepthV2, DA-V2, Depth Pro, Metric3D v2 and OMNI-DC;
  - DA-V2 checkpoint wording fixed;
  - Depth Pro excluded from ARKitScenes for two reasons.
- **Protocol**:
  - exploratory runs chose γ and the design;
  - all reported comparisons come from the confirmatory replication, whose plan was fixed before the runs (E17 20 × 10 plus the r = n arm; E18 20 × 10; E19 10 splits with their own masks, OMNI-DC from laser anchors);
  - validity is one-sided (mean ≥ 1−α − 2 SE), with an A-dual theory check |mean − r/(n+1)| ≤ 2 SE, and claims need a paired-bootstrap CI that excludes zero.
- **Cuts to stay at 8 pages**: a shorter frozen-models paragraph (detail moved to Supp. G), a shorter results placeholder, one sentence in §6.4 merged, and a shorter conclusion sentence.

## Changes in v3 (E18 and literature)

- **E18 integrated** (UniDepthV2 and DA-V2, 20 visit splits × 2 draws; Depth Pro excluded). It fills Tab. 3b, §6.5, the abstract, the intro, the conclusion, Supp. H and Tab. S3. `\NumEEighteenCov{XY}`: X is the structured-anchor source and Y the probe source (L = laser, I = iPad), A-dual truth coverage under uniform anchors, as a range over the two models.
- **E18 vs. the v2 text.**
  - "Only split-CP covers the truth in every cell" was wrong: A-dist-both does too (≥ 0.92), through width.
  - A-dist-split's failure with iPad anchors is uniform-only (0.53–0.70; cluster 0.87–0.94).
  - A-dual-both rescues iPad/iPad under clustered anchors only.
- **Literature.** 34 → 53 references. Added: the depth-completion lineage, uncertainty-aware completion, Prior Depth Anything, MiDaS and VI-Depth alignment, K-RCPS, conformal segmentation, Kandinsky CP (×2), Barber et al. (×2), CQR, gentle introduction, deep ensembles, MEMS LiDAR. The Feldman et al. citations are corrected; MRAC and Deng et al. are described.
- **Usefulness.**
  - The intro opens on the AR and indoor-robot audience and on the failure result.
  - New "Where random probes come from" paragraph in §3, with the KITTI scope.
  - Boxed design rule in §4.3.
  - The cluster-centre reversal is flagged as resting on one dataset and three splits.
  - Limitations state that no downstream use is shown.

## Resolved source conflicts (v1 report)

1. **Model set.** Resolved on 2 Nov: UniDepthV2, DA-V2 and Depth Pro on NYU/KITTI; Metric3D v2 in the supplement only; UniDepthV2 and DA-V2 on ARKitScenes.
2. **ARKit spread without Depth Pro.** Recomputed: 10.5–12.1 % short, p10 0.817–0.821.
3. **"Coverage gate".** The source's meaning is used: ≥ 10 % narrower than A-dist-both, at theoretical coverage.
4. **Interval-score range.** The range over pairs, −1 to −48 %, is used.
5. **γ choice.** It was chosen after a first NYU/KITTI run.
6. **p10 at n = 10.** 0.795 as in the source; scipy gives 0.794.
7. **`docs/CVPR_PLAN.md` is stale** (it still headlines E12). E12 is not used.
8. **Beta law wording.** "Exact Beta law" became "beta-binomial law, approximated by Beta".

## Changes in v2 (E19)

- **Framing.** A-dual is a calibration layer for any centre:
  - completion network under uniform anchors (the kill criterion of ≥ 15 % fired: OMNI-DC is 17–23 % better);
  - the frozen model with our centre under clustered anchors (37 % better);
  - probes certify what they measure.
- **E19 data vs. the v1 text.**
  - v1 expected "the layer is needed on top of the strongest centre". Under uniform anchors, split-CP on OMNI-DC has a per-image spread similar to A-dual's (p10 0.817, 19.1 % short), so the text now says A-dual adds the *guarantee*, not a better empirical spread.
  - A-dist-split is the most efficient method when values are clean. The text says so, and that it has no per-image guarantee and fails with iPad anchors (0.52–0.62).
- **A-dist-split's calibration source.** The method section said it "needs dense reference depth". In the code it calibrates on dense values of the *measurement source*. Corrected.
- **Laser anchors with iPad probes.** These widen A-dual by 2.5× (UniDepthV2) and 3.4× (OMNI-DC), not "~2.5×" as in the brief.
