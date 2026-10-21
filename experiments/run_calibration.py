"""Main calibration study (E2 scores, E3 calibrators/validity notions,
E4 conditional coverage, E6 raw-vs-conformal, E9 alpha sweep).

For each seed: scene-level split into head-train / calibration / test;
fit every scale model on head-train only; compute calibration residuals;
for each calibrator and alpha produce test intervals and metrics.

    python -m experiments.run_calibration --dataset nyu --model dav2-indoor-large
    python -m experiments.run_calibration --dataset kitti --model unidepthv2-large --seed-start 10 --n-seeds 10

Outputs (results/<exp-id>/<dataset>__<model>/):
    runs_s<a>-<b>.csv    one row per (seed, score, calibrator, alpha) + raw baselines
    groups_s<a>-<b>.csv  conditional coverage per analysis group
    diag_s<a>-<b>.csv    scale-model diagnostics (e.g. R^2 of the image factor a(x))
    spec.json
"""

import argparse
import json
import time

import numpy as np
import pandas as pd

from experiments.common import load_table, results_dir, split_tables, write_spec
from src.calibrators.registry import CALIBRATORS, calibrate
from src.metrics.interval_metrics import evaluate_intervals
from src.scores.groups import analysis_groups
from src.scores.learned import FactorizedScale, LearnedVarianceScale
from src.scores.raw_baselines import RAW_SIGMA, available_raw, raw_interval
from src.scores.scale_maps import SCORES, available_scores, build_scale_model, interval, residual


def run_seed(table, seed, args, scores, log):
    head, cal, test = split_tables(table, seed, args.frac_head, args.frac_cal)
    groups_test = analysis_groups(test)
    base = {"seed": seed, "n_head": head.n_images, "n_cal": cal.n_images, "n_test": test.n_images}
    rows, grows, diag = [], [], []

    for name in scores:
        spec = SCORES[name]
        t0 = time.perf_counter()
        model = build_scale_model(spec.scale, beta=args.beta, seed=seed).fit(head)
        s_cal, s_test = model.predict(cal), model.predict(test)
        fit_s = time.perf_counter() - t0
        if isinstance(model, FactorizedScale):
            diag.append({**base, "score": name, "a_r2_head": model.r2_head,
                         "a_test_cv": float(np.std(model.image_factor(test)) / np.mean(model.image_factor(test)))})
        r_cal = residual(cal, spec.space) / s_cal
        r_test = residual(test, spec.space) / s_test
        for alpha in args.alphas:
            for method in args.calibrators:
                res = calibrate(method, cal, r_cal, test, r_test, alpha, args.beta, args.delta)
                lo, hi = interval(test.pix["pred"], res.lam_pix * s_test, spec.space)
                m, g = evaluate_intervals(test, lo, hi, alpha, args.beta, groups_test)
                tag = {**base, "score": name, "calibrator": method, "alpha": alpha, "beta": args.beta}
                rows.append({**tag, **m, "lambda": res.lam_summary, "calib_seconds": res.seconds,
                             "scale_fit_seconds": fit_s,
                             "lambda_groups": json.dumps(res.extra.get("lambda_groups", {})),
                             "patch_event_rate": res.extra.get("patch_event_rate", np.nan),
                             "patch_event_rate_scene": res.extra.get("patch_event_rate_scene", np.nan)})
                grows += [{**tag, **r} for r in g]
        log(f"  seed {seed} {name}: done ({fit_s:.0f}s fit)")

    if args.raw:
        sigmas = {k: RAW_SIGMA[k](test) for k in available_raw(test)}
        var_model = LearnedVarianceScale(seed=seed).fit(head)
        sigmas["raw-learnedvar"] = var_model.predict(test)
        for name, sigma in sigmas.items():
            for alpha in args.alphas:
                lo, hi = raw_interval(test.pix["pred"], sigma, alpha)
                m, g = evaluate_intervals(test, lo, hi, alpha, args.beta, groups_test)
                tag = {**base, "score": name, "calibrator": "raw", "alpha": alpha, "beta": args.beta}
                rows.append({**tag, **m, "lambda": np.nan})
                grows += [{**tag, **r} for r in g]
    return rows, grows, diag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--exp-id", default="main")
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--n-seeds", type=int, default=20)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.05, 0.1, 0.2])
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--frac-head", type=float, default=0.3)
    ap.add_argument("--frac-cal", type=float, default=0.35)
    ap.add_argument("--scores", nargs="*", default=None, help="default: all available for this table")
    ap.add_argument("--calibrators", nargs="*", default=list(CALIBRATORS))
    ap.add_argument("--no-raw", dest="raw", action="store_false")
    ap.add_argument("--raw-only", action="store_true",
                    help="only the raw baselines (writes runs_s<a>-<b>_raw.csv; summarize prefers newer files)")
    ap.add_argument("--max-images", type=int, default=0, help="debug: use only the first N images")
    args = ap.parse_args()

    table = load_table(args.dataset, args.model)
    if args.max_images:
        table = table.subset(np.arange(min(args.max_images, table.n_images)))
    scores = [] if args.raw_only else available_scores(table, args.scores)
    args.raw = args.raw or args.raw_only
    out = results_dir(args.exp_id, f"{args.dataset}__{args.model}")
    seeds = range(args.seed_start, args.seed_start + args.n_seeds)
    suffix = f"s{seeds[0]}-{seeds[-1]}" + ("_raw" if args.raw_only else "")
    write_spec(out, {**vars(args), "scores_run": scores, "table_meta": table.meta,
                     "n_images": table.n_images, "n_pixels": table.n_pixels}, f"spec_{suffix}.json")
    print(f"{args.dataset}/{args.model}: {table.n_images} images, {table.n_pixels:,} px; scores={scores}")

    all_rows, all_groups, all_diag = [], [], []
    for seed in seeds:
        t0 = time.perf_counter()
        rows, grows, diag = run_seed(table, seed, args, scores, print)
        all_rows += rows
        all_groups += grows
        all_diag += diag
        # rewrite after every seed so a crash never loses finished work
        pd.DataFrame(all_rows).assign(dataset=args.dataset, model=args.model).to_csv(out / f"runs_{suffix}.csv", index=False)
        pd.DataFrame(all_groups).assign(dataset=args.dataset, model=args.model).to_csv(out / f"groups_{suffix}.csv", index=False)
        if all_diag:
            pd.DataFrame(all_diag).assign(dataset=args.dataset, model=args.model).to_csv(out / f"diag_{suffix}.csv", index=False)
        print(f"seed {seed} finished in {time.perf_counter() - t0:.0f}s")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
