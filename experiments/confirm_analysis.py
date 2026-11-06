"""Confirmatory analysis (docs/ANALYSIS_PLAN_2026-11-03.md): validity and paired-bootstrap comparisons over splits.

Reads every runs.csv matched by --runs (shards of one experiment: <exp>/<dataset>__<model>[__<layout>][__seeds<a>_<b>]),
averages each metric over draws within a split (seed), then per stratum (--by) and arm (--arm-cols):
  validity.csv  n_splits, mean coverage, SE over splits, valid (mean >= 1 - alpha - 2 SE),
                exact (|mean - (1 - alpha)| <= 2 SE), and the means of the reported metrics
  compare.csv   for each --compare A:B, per metric: means, mean paired difference A - B (per split), its 95 %
                percentile-bootstrap CI over splits, and claim = CI lies below 0 (every compared metric is
                lower-is-better) and A is valid

    python -m experiments.confirm_analysis --runs 'results/e18_conf/*/runs.csv' \
        --by source pred_source probe_source --arm-cols method \
        --compare A-dual:A-dist-both A-dual-both:A-dist-both --out results/e18_conf/_confirm
    python -m experiments.confirm_analysis --runs 'results/e19_conf/*/runs.csv' \
        --by pattern pred_source probe_source --arm-cols centre method \
        --compare omnidc/A-dual:unidepth/A-dual omnidc/split-CP:unidepth/A-dual --out results/e19_conf/_confirm
"""

from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = ["cov_image_mean", "cov_image_p10", "frac_img_short", "rel_interval_score_robust",
           "rel_interval_score_imgmed", "rel_interval_score", "median_rel_width", "median_width_over_gt",
           "frac_uninformative"]


def load(pattern: str) -> pd.DataFrame:
    frames = []
    for path in sorted(glob.glob(pattern)):
        d = pd.read_csv(path)
        d["source"] = re.sub(r"__seeds\d+_\d+$", "", Path(path).parent.name)
        frames.append(d)
    if not frames:
        raise SystemExit(f"no runs.csv matches {pattern}")
    runs = pd.concat(frames, ignore_index=True)
    dup = runs.duplicated(["source", "seed", "draw", *[c for c in ("k", "alpha", "base", "pattern", "centre",
                                                                     "pred_source", "probe_source", "method")
                                                         if c in runs]])
    if dup.any():
        raise SystemExit(f"{int(dup.sum())} duplicate (split, draw, arm) rows: overlapping shards?")
    return runs


def bootstrap_ci(diff: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    idx = np.random.default_rng(seed).integers(0, len(diff), size=(n_boot, len(diff)))
    means = diff[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="glob of runs.csv files (quote it)")
    ap.add_argument("--by", nargs="+", default=["source"], help="stratum columns")
    ap.add_argument("--arm-cols", nargs="+", default=["method"], help="columns naming an arm (joined by '/')")
    ap.add_argument("--compare", nargs="*", default=[], help="A:B arm pairs; difference A - B")
    ap.add_argument("--metrics", nargs="+", default=METRICS)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--nominal", nargs="*", default=["A-dual=0.904762", "A-dual-rmax=0.952381"],
                    help="arm-suffix=level for the exactness check (A-dual: r / (n + 1), n = 20); default 1 - alpha")
    ap.add_argument("--base", default="S2-log", help="keep only this base when the runs have one")
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    runs = load(args.runs)
    if "base" in runs:
        runs = runs[runs.base == args.base]
    runs = runs[np.isclose(runs.alpha, args.alpha)]
    metrics = [m for m in args.metrics if m in runs]
    runs = runs.assign(arm=runs[args.arm_cols].astype(str).agg("/".join, axis=1))
    per_split = runs.groupby([*args.by, "arm", "seed"])[metrics].mean().reset_index()
    target = 1 - args.alpha
    nominal = {s.split("=")[0]: float(s.split("=")[1]) for s in args.nominal}

    val = []
    for key, g in per_split.groupby([*args.by, "arm"]):
        n = len(g)
        cov = g["cov_image_mean"].to_numpy()
        se = cov.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
        nom = nominal.get(key[-1].split("/")[-1], target)
        val.append({**dict(zip([*args.by, "arm"], key)), "n_splits": n, "cov_se": se, "nominal": nom,
                    "valid": bool(cov.mean() >= target - 2 * se), "exact": bool(abs(cov.mean() - nom) <= 2 * se),
                    **{m: float(g[m].mean()) for m in metrics}})
    val = pd.DataFrame(val)
    valid = {tuple(r[[*args.by, "arm"]]): r["valid"] for _, r in val.iterrows()}

    cmp_rows = []
    for spec in args.compare:
        a, b = spec.split(":")
        for key, g in per_split.groupby(args.by):
            key = key if isinstance(key, tuple) else (key,)
            ga, gb = g[g.arm == a].set_index("seed"), g[g.arm == b].set_index("seed")
            common = ga.index.intersection(gb.index)
            if len(common) < 2:
                continue
            for m in metrics:
                if m.startswith("cov_") or m == "frac_img_short":
                    continue
                diff = (ga.loc[common, m] - gb.loc[common, m]).to_numpy(dtype=np.float64)
                if not np.all(np.isfinite(diff)):
                    lo = hi = np.nan
                else:
                    lo, hi = bootstrap_ci(diff, args.n_boot, seed=len(cmp_rows))
                mb = float(gb.loc[common, m].mean())
                excl = bool(np.isfinite(lo) and (hi < 0 or lo > 0))
                favours_a = bool(np.isfinite(hi) and hi < 0)
                cmp_rows.append({**dict(zip(args.by, key)), "A": a, "B": b, "metric": m, "n_splits": len(common),
                                 "mean_A": float(ga.loc[common, m].mean()), "mean_B": mb,
                                 "diff": float(diff.mean()), "rel_diff_pct": 100 * float(diff.mean()) / mb if mb else np.nan,
                                 "ci_lo": lo, "ci_hi": hi, "ci_excludes_0": excl,
                                 "A_valid": valid.get((*key, a)), "B_valid": valid.get((*key, b)),
                                 "claim": favours_a and bool(valid.get((*key, a)))})

    args.out.mkdir(parents=True, exist_ok=True)
    val.to_csv(args.out / "validity.csv", index=False)
    pd.DataFrame(cmp_rows).to_csv(args.out / "compare.csv", index=False)
    print(val.round(4).to_string(index=False))
    if cmp_rows:
        show = [*args.by, "A", "B", "metric", "rel_diff_pct", "ci_lo", "ci_hi", "claim"]
        print(pd.DataFrame(cmp_rows)[show].round(4).to_string(index=False))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
