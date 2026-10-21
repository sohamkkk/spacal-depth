"""Aggregate run files into paper tables + apply the pre-registered decision rule.

Decision rule (rules/conformal-validity): method A "beats" baseline B on a
(dataset, model, alpha) cell iff
  1. A is valid: mean coverage of its target notion >= nominal - tol,
     tol = 2 * run-to-run SE (so random shortfall is not punished), and
  2. the paired (same seed) difference in relative interval score
     A - B has a 95 % bootstrap CI entirely below 0.
Validity target per calibrator: V4 -> v4_rate >= 1 - alpha;
K2 -> patch_event_rate >= 1 - alpha; K1-scene -> cov_scene_mean >= 1 - alpha;
V4-scene -> v4_rate_scene; K2-scene -> patch_event_rate_scene;
everything else -> cov_image_mean >= 1 - alpha.

    python -m experiments.summarize --exp-id main
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd

from experiments.common import results_dir, to_markdown
from src.config import RESULTS_ROOT

KEYS = ["dataset", "model", "alpha", "score", "calibrator"]
METRICS = ["cov_pixel", "cov_image_mean", "cov_scene_mean", "cov_image_sd", "frac_img_cov_below_1ma_m10", "v4_rate",
           "v4_rate_scene", "patch_event_rate", "patch_event_rate_scene", "rel_width", "rel_interval_score",
           "interval_score", "worst_group_gap", "frac_infinite"]


def validity_metric(calibrator: str) -> str:
    return {"V4": "v4_rate", "K2": "patch_event_rate", "K1-scene": "cov_scene_mean",
            "V4-scene": "v4_rate_scene", "K2-scene": "patch_event_rate_scene"}.get(calibrator, "cov_image_mean")


def bootstrap_ci(x: np.ndarray, n_boot: int = 5000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = rng.choice(x, (n_boot, len(x)), replace=True).mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def aggregate(runs: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in METRICS if c in runs]
    g = runs.groupby(KEYS)[cols]
    agg = g.mean().add_suffix("_mean").join(g.std().add_suffix("_sd")).join(g.size().rename("n_seeds"))
    return agg.reset_index()


def decide(runs: pd.DataFrame, baselines: list[tuple[str, str]]) -> pd.DataFrame:
    out = []
    for (ds, model, alpha), cell in runs.groupby(["dataset", "model", "alpha"]):
        cell = cell[np.isfinite(cell.rel_interval_score)]
        methods = cell.groupby(["score", "calibrator"])
        for b_score, b_cal in baselines:
            base = cell[(cell.score == b_score) & (cell.calibrator == b_cal)].set_index("seed")
            if base.empty:
                continue
            for (score, cal), a in methods:
                if (score, cal) == (b_score, b_cal):
                    continue
                a = a.set_index("seed")
                common = a.index.intersection(base.index)
                if len(common) < 3:
                    continue
                vm = validity_metric(cal)
                v = a.loc[common, vm].to_numpy(dtype=float)
                nominal = 1 - alpha
                tol = 2 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
                valid = bool(np.nanmean(v) >= nominal - tol)
                diff = (a.loc[common, "rel_interval_score"] - base.loc[common, "rel_interval_score"]).to_numpy()
                lo, hi = bootstrap_ci(diff)
                out.append({"dataset": ds, "model": model, "alpha": alpha,
                            "baseline": f"{b_score}/{b_cal}", "method": f"{score}/{cal}",
                            "validity_metric": vm, "validity_mean": float(np.nanmean(v)), "valid": valid,
                            "d_rel_IS_mean": float(diff.mean()), "d_rel_IS_ci_lo": lo, "d_rel_IS_ci_hi": hi,
                            "beats_baseline": valid and hi < 0, "n_seeds": len(common)})
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", default="main")
    ap.add_argument("--baselines", nargs="*", default=["S2-log/K1", "S1-rel/K1", "S3-flip/K1"],
                    help="score/calibrator pairs every method is compared to")
    args = ap.parse_args()

    files = sorted(glob.glob(str(RESULTS_ROOT / args.exp_id / "*" / "runs_*.csv")), key=os.path.getmtime)
    if not files:
        raise SystemExit(f"no run files under {RESULTS_ROOT / args.exp_id}")
    runs = pd.concat(map(pd.read_csv, files), ignore_index=True)
    dup = runs.duplicated(subset=KEYS + ["seed"], keep="last")
    if dup.any():
        print(f"note: {int(dup.sum())} (seed, method) rows superseded by newer run files (e.g. a --raw-only rerun)")
        runs = runs[~dup]
    runs = runs[runs.score != "raw-conf"]  # dropped baseline (src/scores/raw_baselines.py); older run files still have it
    out = results_dir(args.exp_id, "_summary")

    agg = aggregate(runs)
    agg.to_csv(out / "aggregate.csv", index=False)
    baselines = [tuple(b.split("/")) for b in args.baselines]
    dec = decide(runs, baselines)
    dec.to_csv(out / "decisions.csv", index=False)

    md = [f"# Summary of `{args.exp_id}` ({len(files)} run files, {runs.seed.nunique()} seeds)\n"]
    show = ["score", "calibrator", "cov_pixel_mean", "cov_image_mean_mean", "cov_scene_mean_mean",
            "cov_image_sd_mean", "v4_rate_mean", "v4_rate_scene_mean",
            "rel_width_mean", "rel_interval_score_mean", "rel_interval_score_sd", "worst_group_gap_mean"]
    for (ds, model, alpha), t in agg.groupby(["dataset", "model", "alpha"]):
        md.append(f"\n## {ds} / {model} / alpha={alpha}\n")
        md.append(to_markdown(t[[c for c in show if c in t]].sort_values(["calibrator", "score"])))
    if not dec.empty:
        md.append("\n## Decision rule: methods that beat a baseline\n")
        wins = dec[dec.beats_baseline]
        md.append(to_markdown(wins[["dataset", "model", "alpha", "baseline", "method", "validity_mean",
                                    "d_rel_IS_mean", "d_rel_IS_ci_lo", "d_rel_IS_ci_hi"]]) if not wins.empty else "_none_\n")
    (out / "summary.md").write_text("".join(md))
    print(f"wrote {out}/summary.md, aggregate.csv, decisions.csv")


if __name__ == "__main__":
    main()
