"""E5: distribution shift. Calibrate on the source dataset, test on
  (a) held-out source scenes (in-domain reference),
  (b) all target images (shifted; no guarantee — this measures the damage),
  (c) target recalibrated on half its scenes (what a few target labels buy).

    python -m experiments.e5_shift --source nyu --target diode_indoor --model dav2-indoor-large
    python -m experiments.e5_shift --source kitti --target diode_outdoor --model dav2-outdoor-large
"""

import argparse

import numpy as np
import pandas as pd

from experiments.common import exchange_units, load_table, results_dir, split_tables, write_spec
from src.calibrators.registry import calibrate
from src.data.splits import scene_level_split
from src.metrics.interval_metrics import evaluate_intervals
from src.scores.groups import analysis_groups
from src.scores.scale_maps import SCORES, available_scores, build_scale_model, interval, residual

DEFAULT_SCORES = ["S2-log", "S3-flip", "S4c-conf", "S5-learned", "S6-fact", "S6-fact-conf"]
DEFAULT_CALIBRATORS = ["K1", "K1-scene", "V4", "K3-depth"]


def evaluate_all(rows, grows, tag, model, spec, cal, test, args):
    s_cal, s_test = model.predict(cal), model.predict(test)
    r_cal = residual(cal, spec.space) / s_cal
    groups = analysis_groups(test)
    for alpha in args.alphas:
        for method in args.calibrators:
            res = calibrate(method, cal, r_cal, test, None, alpha, args.beta)
            lo, hi = interval(test.pix["pred"], res.lam_pix * s_test, spec.space)
            m, g = evaluate_intervals(test, lo, hi, alpha, args.beta, groups)
            t = {**tag, "calibrator": method, "alpha": alpha, "lambda": res.lam_summary}
            rows.append({**t, **m})
            grows += [{**t, **r} for r in g]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.1])
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--scores", nargs="*", default=DEFAULT_SCORES)
    ap.add_argument("--calibrators", nargs="*", default=DEFAULT_CALIBRATORS)
    args = ap.parse_args()

    src_t, tgt_t = load_table(args.source, args.model), load_table(args.target, args.model)
    scores = [s for s in available_scores(src_t, args.scores) if s in available_scores(tgt_t, args.scores)]
    out = results_dir("e5", f"{args.source}_to_{args.target}__{args.model}")
    write_spec(out, {**vars(args), "scores_run": scores})

    rows, grows = [], []
    for seed in range(args.n_seeds):
        head, cal, test = split_tables(src_t, seed)
        tgt_units = exchange_units(tgt_t)
        tgt_pools = scene_level_split(list(tgt_units), 0.0, 0.5, seed)
        tgt_cal = tgt_t.subset(np.flatnonzero(np.isin(tgt_units, list(tgt_pools["calibration"]))))
        tgt_test = tgt_t.subset(np.flatnonzero(np.isin(tgt_units, list(tgt_pools["test"]))))
        for name in scores:
            spec = SCORES[name]
            model = build_scale_model(spec.scale, beta=args.beta, seed=seed).fit(head)
            base = {"seed": seed, "score": name, "source": args.source, "target": args.target, "model": args.model}
            evaluate_all(rows, grows, {**base, "eval": "source_test"}, model, spec, cal, test, args)
            evaluate_all(rows, grows, {**base, "eval": "target_shifted"}, model, spec, cal, tgt_test, args)
            evaluate_all(rows, grows, {**base, "eval": "target_recalibrated"}, model, spec, tgt_cal, tgt_test, args)
        pd.DataFrame(rows).to_csv(out / "runs.csv", index=False)
        pd.DataFrame(grows).to_csv(out / "groups.csv", index=False)
        print(f"seed {seed} done")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
