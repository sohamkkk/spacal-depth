# What the draft needs from the authors (3 Nov 2026, v4)

Deadlines: abstract registration **10 Nov**, paper **16 Nov AoE**, supplement 23 Nov `[VERIFY: CVPR 2027 supplement deadline]`.
Priority: **P0** blocks registration; **P1** blocks the paper; **P2** before the supplement or camera-ready. Each open item lists the decision or data needed, the options with a recommendation (★), and what it unblocks. Item numbers are kept from v3 so earlier references still work.

## Resolved in v4 (3 Nov)

1. ~~**Title.**~~ Option (b): *Structured Anchors Predict, Random Probes Calibrate: A Per-Image Conformal Layer for Sparse-Guided Depth*, in `main.tex` and `supplement.tex`.
2. ~~**Registered abstract.**~~ It keeps 0.26–0.47 and 16–69 %. The E19 result (3 splits × 1 draw) and the E18 result (20 splits, not yet multi-draw or bootstrapped) are now in words. The same rule applies to the intro contributions (the raw-ratio number in intro para 2 also became words) and the conclusion. Results keep the orange numbers.
4. ~~**Short-image definition.**~~ It is now **single-draw** everywhere (theory 17.6 % at n = 20). The two-draw law survives only as a Supp. B footnote (implementation check). The E17 and E18 short cells and text macros print `[confirmatory]` until the summaries carry `frac_img_short`.
5. ~~**Interval score.**~~ Amended on 3 Nov, before any confirmatory output (`docs/ANALYSIS_PLAN_2026-11-03.md`, amendment log). The **primary** score is `rel_interval_score_imgmed`: per image, the mean over pixels of IS / y, then the median over images. A per-pixel median (`rel_interval_score_robust`) mostly ignores misses at 90 % coverage (smoke test, A-dual: 0.047 vs. 0.088 vs. 0.103 plain mean). **Secondary:** `frac_uninformative`, the share of pixels with width > 10 y, infinite included, which should explain the E19 OMNI-DC cluster blow-ups. Both are defined in §5 Metrics and Supp. G. `gen_numbers.py` takes IS in the order imgmed → robust → exploratory. It adds an Uninf. column to Tab. S2 when the E19 summary has `frac_uninformative`. The 8 `[primary IS]` cells stay until E19c.
6. ~~**Envelope α split and loss.**~~ α_s = α_e = 0.05 (the 0.07 / 0.03 option is dropped). **Amended 4 Nov, before any envelope run** (analysis plan, amendment log): one two-sided conformal risk control at α_e over a class-adaptive one-parameter family. Each confidence class gets the band [Q_c(λ/2), Q_c(1−λ/2)] of the corrected head-train error, clipped to contain 0. λ̂ is the largest grid value with U/(U+1)·L̂ + 1/(U+1) ≤ α_e, where L̂ is the mean over calibration scenes of the fraction of pixels outside their class band. Per-class CRC at π_c·α_e/2 was infeasible: CRC cannot certify below 1/(U+1) = 0.026 with U = 37, and the class targets were 0.0003 / 0.001 / 0.024. So the π_c normalisation question is moot. Prop. 2 is unchanged, and its proof is now one union bound (Theorem 1's α_s plus the single CRC's α_e). The text is updated in `sections/envelope.tex` (main §4.6 or Supp. D) and the Supp. D proof, which also gives illustrative 3-split fit values in orange (λ̂ 0.016–0.024; high-confidence band ≈ ±6 %). The gate is unchanged, and the fallback is the `\envelopemainfalse` switch in `macros.tex`.
26. ~~**Envelope: run it or drop it.**~~ Decided 4 Nov: the envelope will be implemented and run as E20 (`--envelope`, arms `A-dual-s` / `A-dual-env`; ARKitScenes × {UniDepthV2, DA-V2} × {uniform, cluster}, 20 splits × 10 draws). It uses the amended single-CRC rule (item 6). The gate is unchanged and evaluated per layout against matched A-dist-both on the primary metric. The `\envelopemainfalse` fallback text stays ready. If the gate passes in one layout only, placement is an author decision, to be recorded in the plan before the run.
7. ~~**Confirmatory go-ahead.**~~ The design is in the Protocol paragraph and in `docs/ANALYSIS_PLAN_2026-11-03.md`, which the author is writing:
   - E17: 20 × 10, γ frozen, plus an r = n = 20 arm;
   - E18: 20 × 10;
   - E19: 10 splits with their own anchor masks, OMNI-DC from laser anchors; the iPad-anchor OMNI-DC cells stay exploratory (Supp. G). **Done (E19c, seeds 0–9); integrated in v4.4.** Tab. 3, Tab. S2, §6.4 and Supp. G now report it as confirmatory. Abstract and intro drop "in exploratory runs" for the cluster claim; Limitations keeps the one-network / one-dataset caveat. OMNI-DC split-CP under clusters is reported as not valid (0.895 < 0.9 − 2 SE).
12. ~~**Training-data `\verify` items.**~~ These are filled from the author's facts. Main §5 gives the short form ("NYU and KITTI are not in the training mixtures of these models"). The new "Training data of the models" paragraph in Supp. G gives the full lists. The DA-V2 wording is fixed (indoor/Hypersim on NYU and ARKitScenes; outdoor/Virtual KITTI 2 on KITTI). Depth Pro is excluded from ARKitScenes for two reasons. There is no `\verify` left in the text or in `refs.bib`.
20. ~~**Design-rule arm.**~~ The r = n = 20 arm is part of the confirmatory plan. The box in §4.3 has `\pending{measured, $r=n$ arm}` for its width cost (Supp. I item iii).

## P0 — before 10 Nov (registration)

3. **Authors, affiliations, subject areas and conflicts** for OpenReview registration. These are not in the PDF, which stays anonymous. Also check whether the registered abstract can still be edited until 16 Nov `[VERIFY: CVPR 2027 OpenReview rules]`.

## P1 — before 16 Nov (paper)

8. **Hand-edit `numbers.tex` (one-time cleanup).** The read-budget hook blocks Read/StrReplace on `numbers.tex`, so these lines are left to you. The PDF is correct either way: the generator now writes the four E18 coverage macros as `\providecommand` + `\renewcommand`, so deleting the old lines needs **no** generator change. Line numbers are as of 3 Nov v4. Work from the bottom up so they stay valid.
   - L147 `\newcommand{\NumOmniSeeds}{3}` → `\newcommand{\NumOmniSeeds}{10}` (**needed now**: Tab. 3 caption, §5 Methods, §6.4 and Limitations print it).
   - L167 `\ENineteen`: it still wraps every per-cell E19 value in `\prelim` (orange), although only the OMNI-DC iPad-anchor cells are exploratory. (`gen_numbers.py` already prints the E19 text macros plain, except `\NumIpadAnchorDual*` / `\NumIpadAnchorDist*`.) Exact replacement for L167, two lines, tested on the supplement (no errors, no overfull):
     ```latex
     \newcommand{\ENineteen}[6]{\ifcsname e19/#1/#2/#3/#4/#5/#6\endcsname\ifcsname e19x/#2/#3\endcsname\prelim{\csname e19/#1/#2/#3/#4/#5/#6\endcsname}\else\csname e19/#1/#2/#3/#4/#5/#6\endcsname\fi\else\pending{?}\fi}
     \expandafter\def\csname e19x/omnidc/lidar\endcsname{}
     ```
     The second line lists the exploratory (centre, anchor source) pairs; only `omnidc/lidar` is orange afterwards. Then drop "(table cells still orange)" from the results placeholder.
   - L152–153 `\NumOmniAbsRelUniform` / `\NumOmniAbsRelCluster`: unused. Delete them, or fill them with the confirmatory values now printed plain in Supp. G (OMNI-DC 0.0064 / 0.0738; add e.g. `\NumUniScaledAbsRelU` = 0.0293, `\NumUniScaledAbsRelC` = 0.0394 for UniDepthV2 rescaled by the same anchors) and replace the hard-coded numbers in Supp. G and §6.4 (\"AbsRel 0.074, vs. 0.039\").
   - Optional new generated macro for the hard-coded, plain `17` in §6.4 (OMNI-DC A-dual vs. split-CP, cluster, −17.0 %, CI excludes 0; confirmatory), e.g. `\NumOmniDualGainCluster`.
   - L161–163: delete. They are the `% A-dual short share with 40 probes (two-draw mean)…` comment, `\NumShortFortyLo` and `\NumShortFortyHi`, all unused; the generated `\NumESeventeenShortForty` replaces them.
   - L140 `\newcommand{\NumEnvAlphaS}{\pending{$\alpha_s$}}` → `\newcommand{\NumEnvAlphaS}{0.05}`.
   - L141 `\newcommand{\NumEnvAlphaE}{\pending{$\alpha_e$}}` → `\newcommand{\NumEnvAlphaE}{0.05}`. Both macros are currently unused; the text writes 0.05.
   - L129–136: delete. This is the stale "E18 … (running)" header, its three comment lines and the four `\newcommand{\NumEEighteenCov{LL,IL,LI,II}}{\pending{…}}`.
   - L121–122: delete `\NumArkShortLo` and `\NumArkShortHi`. They are unused and two-draw; the generated `\NumArkShortOne` replaces them.
   - L119 `% UPD 9.22 Finding 3 (ARKitScenes, laser-valued probes, n = 20): short share (avg. of 2 draws) and p10.` → `% UPD 9.22 Finding 3 (ARKitScenes, laser-valued probes, n = 20): p10 (one draw); short share is generated (\NumArkShortOne).`
   - L103 `\NumMeasShortTwoTwenty`: **keep**; the Supp. B footnote uses it.
   - L94 and L96: delete `\NumBetaShortTwoTen` and `\NumBetaShortTwoForty`, which are unused. Keep L95 `\NumBetaShortTwoTwenty` (Supp. B footnote).
   - L93 `% [theory] share of images below 0.85 when coverage is averaged over two independent probe draws (Monte Carlo).` → `% [theory] two-draw-average short share (Monte Carlo): implementation check only, Supp. B footnote.`

9. **Figures that need real renders.** Fig. 3 is done; the teaser is open.
   - **Teaser** (`figures/teaser.tex`): one ARKitScenes frame, cluster layout, k = 200, UniDepthV2. Panels: RGB with anchors and probes; A-local miss map; A-dual miss map; per-image coverage histogram. `experiments/qualitative.py` draws K1/A-both maps only and needs an A-local vs. A-dual mode.
   - ~~**Fig. 3**~~ Resolved 4 Nov: `figures/e17_results.pdf` (from `experiments/make_figures.py fig_e17`). (a) near layout, k = 50, n = 20; median and range over the six NYU/KITTI pairs; pixel-share bars. (b) is the CDF of the Beta per-image coverage law for n = 10/20/40, with the measured p10 as dots, **not** a histogram, because per-image coverages are not stored. Caption and text checked.

   The teaser is still open (being rendered next). Keep it within the current placeholder height. After v4.4, the Conclusion ends one line above the bottom of p. 8 with the envelope in the main paper (References now start on p. 9); with the switch off there is more room.

10. ~~**OMNI-DC accuracy on unmeasured pixels.**~~ From `results_omnidc/e19c_confirm/absrel.csv` (confirmatory splits; image-mean AbsRel on test pixels outside the structured anchors; mean over 10 seeds). OMNI-DC: 0.0064 uniform / 0.0738 cluster. UniDepthV2 rescaled per image by the median log(laser/pred) at the same 180 anchors: 0.0293 / 0.0394; raw (no-anchor reference) 0.121. δ<1.25: 0.997 vs. 0.988 (uniform) and 0.944 vs. 0.981 (cluster). The comparison is against the anchor-scaled model, not the raw prediction (v4.5 correction). Printed **plain** (hard-coded) in Supp. G with min–max over seeds, plus one clause in §6.4 cluster (0.074 vs. 0.039). Macro suggestion in item 8.

13. **Repeat the prior-art search on arXiv and Google Scholar**, up to 16 Nov. Earlier searches were web-only. Queries:
   - "conformal depth completion"
   - "conformal prediction sparse depth / LiDAR"
   - "per-image / instance-wise conformal prediction"
   - "spatial conformal prediction sampling design"
   - "calibration probes / held-out measurements depth"
   - "uncertainty depth completion coverage guarantee"
   - "conformal monocular depth"

   Anything close goes into Related Work.

23. ~~**New summary columns.**~~ Resolved in code. The experiment code writes per-run columns and their summary means: `rel_interval_score_robust`, `rel_interval_score_imgmed` (primary), `frac_img_short` (single draw, < 1−α−0.05) and `frac_uninformative`. It also has the new arm `A-dual-rmax` (r = n = 20, nominal 20/21) in E17c. `gen_numbers.py` already reads the columns. The numbers arrive with the confirmatory runs: E17c and E18c at 20 splits × 10 draws on the cluster, E19c at 10 splits on Colab, analysed with `experiments/confirm_analysis.py`. Still to do once the outputs exist:
   - point `gen_numbers.py` at the E17c/E18c/E19c paths;
   - fill the design-rule box's `\pending{measured, $r=n$ arm}` from `A-dual-rmax`.

24. ~~**Validity criterion.**~~ The plan uses: valid = mean ≥ 1−α − 2 SE; theory check for A-dual = |mean − r/(n+1)| ≤ 2 SE. §5 states both.

25. **UniDepthV2 and ARKitScenes wording.** Because UniDepthV2 does not report which ARKitScenes split it trained on, §5 says our validation-split frames are "*likely* unseen". The brief's version was "unseen". ★ Keep "likely" unless the UniDepthV2 authors confirm training-split-only use.

## P2 — supplement (23 Nov) and camera-ready

14. **Supplement packaging.** ★ Keep it as a separate PDF, as now: it compiles alone and can be finished after 16 Nov. The alternative is appending it to the main PDF, as the kit allows. Cross-references are hand-numbered (Supp. A–I, main Tab. 1–3; E18 is Supp. H / Tab. S3; training data and E19 details are Supp. G). Recheck them if sections move. With `\envelopemainfalse`, Supp. D becomes "Sensor-error envelope (Proposition 2)" and the supplement grows to 7 pages.
15. **E19 scope.** Only uniform and cluster layouts at k = 200 with UniDepthV2. Reviewers may ask about rows, near or edges, k = 50, or DA-V2 as the frozen centre. Options: ★ add rows and near at k = 200 to the supplement if GPU time allows, or state the scope (already in Limitations).
16. **Depth Pro on ARKitScenes.** It is now excluded for two stated reasons: ARKitScenes is in its stage-1 training data, and our run is not reproduced. ★ Drop it for good. The diagnosis of the broken run (not in the paper):
   - E18 laser/laser, uniform layout: rel. IS of A-dist-both 2.52 vs. 0.23–0.31 for the other models;
   - A-local-split's median relative width is 0.295 vs. 0.109;
   - the likely cause is the intrinsics, resolution or units passed to Depth Pro.
17. **Metric3D v2**: E13–E17 runs for the supplement. **Running** (author decision, 4 Nov). The promise in §5 stays. Supp. I item (vi) fills when they finish.
18. **Method names** (`macros.tex`, one place). Options: ★ keep A-dual / A-dual-both / A-dist-split, or switch to descriptive names such as "probe-calibrated (PC)" and "PC + distance hull".
19. **Camera-ready only:** author block, affiliations, acknowledgements and funding, and a code or data release statement with URL. None of these may appear in the review PDF.
21. **Reading of E18 for the text (please confirm).**
    - A-dist-split's failure with iPad anchors is uniform-only: 0.53–0.70 uniform, but 0.87–0.94 cluster.
    - The hull A-dual-both rescues iPad/iPad truth coverage under clustered anchors (0.95–0.96), not under uniform ones (0.67–0.74).
    - A-dist-both covers the truth in every cell (≥ 0.92) through width: 2.1–2.7× A-dual's IS under uniform anchors, about equal under cluster.
22. **Policy check**: the CVPR 2027 author guidelines (page limit, limitations and broader-impact expectations, LLM-use policy) `[VERIFY]`. The draft follows the 8-page main-text rule from the kit.
