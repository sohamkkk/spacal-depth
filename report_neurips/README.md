# Course project report (NeurIPS 2025 style)

Course report for the SpaCal-Depth project, written in the professor's NeurIPS template. It covers the whole project: the broad conformal study (E0–E16) as the foundation, and the dual-anchor method (A-dual, E17–E20) as the main contribution.

> **Do not post this report publicly** (course websites, arXiv, GitHub, personal pages) while the CVPR 2027 submission is under double-blind review. The report shares the method, results and figures of that submission and names the authors. [VERIFY: the CVPR 2027 policy on course reports and technical reports during review.]

## Build

From this folder:

```
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Then remove the aux files: `rm -f main.aux main.log main.out main.blg main.bbl`.

## Single source with `../paper`

Nothing under `../paper` is modified; it is read only.

- **Macros:** `../paper/macros.tex` and `../paper/numbers.tex` are `\input` directly, so every number, and every `\prelim` (orange, exploratory) or `\pending` (red) marker, stays in step with the paper.
  - `shims.tex` supplies what `cvpr.sty` normally provides (`xspace`, `enumitem`, `\eg`, `\ie`, `\etal`).
  - `report_macros.tex` defines the broad-study method names. It also re-defines the per-cell `\ENineteen` lookup so that only the exploratory OMNI-DC iPad-anchor cells are orange. The hand part of `numbers.tex` (L167) still makes every E19 cell orange in the paper; AUTHOR_ASKS item 8 has the fix.
- **Bibliography:** `../paper/refs.bib` plus `extra.bib`. `extra.bib` has five entries needed only for the broad study: RCPS, DIODE, ETH3D, conformal decision theory, and hierarchical conformal prediction. They are transcribed from the reference list of `docs/PROJECT_REPORT.md` [VERIFY: bibliographic details].
- **Figures:**
  - `../paper/figures/pipeline.tex` (TikZ) and `../paper/figures/e17_results.pdf`;
  - `../results/figures/e8_calsize__nyu__dav2-indoor-large.pdf` and `e14_distance__S2-log.pdf`.

## Structure

The headings follow the professor's list:

| Part | File |
|---|---|
| Abstract | `sections/abstract.tex` |
| 1 Introduction | `sections/intro.tex` |
| 2 Literature Review | `sections/literature.tex` |
| 3 Problem Statement | `sections/problem.tex` |
| 4 Proposed Method | `sections/method.tex` |
| 5 Experiments and Results | `sections/experiments.tex` |
| 6 Conclusions | `sections/conclusions.tex` |
| References | — |
| Appendix | `sections/appendix.tex` |

The appendix holds the proofs, the guarantees table, the A-dist details, the datasets table, the broad-study details, the probe-budget ablation, the E19c interval-score blow-up, the experiment index and reproducibility notes.

There is no NeurIPS checklist (author decision). The style is loaded with `[final]`: names visible, no line numbers.

**Deviation from the template:** with `[final]`, the template prints "39th Conference on Neural Information Processing Systems (NeurIPS 2025)" at the bottom of page 1. That is false for a course report, so `main.tex` replaces only this notice string with "Course project report. [TODO: course name, code and semester]".

## Pending

- **Author block** (placeholder in `main.tex`): `\TODO{authors, roll numbers, emails}` and `\TODO{department and institute}`.
- **Page-1 notice:** `\TODO{course name, code and semester}`.
- **Code link:** `\TODO{public code link …}` in Appendix "Reproducibility", if permitted after the review period.
- **Confirmatory runs still in progress:**
  - E17c and E18c (orange numbers in Tables 3 and 5, the short-share `[confirmatory]` cells and the `r = n` width cost);
  - E20, the envelope: the `[E20]` row of Table 5, the placeholder box in Sec. 5.5, and `\NumEnvTruthCov` / `\NumEnvISGain`.
  
  They fill in automatically once `paper/numbers.tex` is regenerated.
- **E19c is confirmatory and complete.**
  - `paper/numbers.tex` L147 still has `\NumOmniSeeds{3}`. The report does not use that macro; it writes "ten splits".
  - The anchor-scaled AbsRel numbers (Sec. 5.4) are hard-coded, as in the paper's supplement.
- **Broad-study numbers** (E0–E16) are typed from `docs/PROJECT_REPORT.md` and printed in black; they are final for that study. Where the paper has a macro for the same quantity (for example A-local under clustered anchors), the macro is used, so it is orange until E17c replaces it.
