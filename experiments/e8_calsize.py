"""E8: calibration-set size. For n_cal images (whole scenes), how does the
run-to-run spread of realised test coverage behave? Theory for split CP
over exchangeable images predicts the Beta spread with n = #images; a
pixel-level view (n = #pixels) predicts a spread ~100x too small.

    python -m experiments.e8_calsize --dataset nyu --model dav2-indoor-large
"""

import argparse

import numpy as np
import pandas as pd

from experiments.common import load_table, results_dir, split_tables, subsample_scenes, to_markdown, write_spec
from experiments.e1_synthetic import beta_sd
from src.calibrators.registry import calibrate
from src.metrics.interval_metrics import evaluate_intervals
from src.scores.scale_maps import SCORES, build_scale_model, interval, residual


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--sizes", type=int, nargs="+", default=[25, 50, 100, 200, 400])
    ap.add_argument("--n-seeds", type=int, default=50)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--scores", nargs="*", default=["S2-log", "S5-learned"])
    ap.add_argument("--calibrators", nargs="*", default=["K0", "K1", "K1-scene", "V4"])
    args = ap.parse_args()

    table = load_table(args.dataset, args.model)
    out = results_dir("e8", f"{args.dataset}__{args.model}")
    write_spec(out, vars(args))
    rows = []
    for seed in range(args.n_seeds):
        head, cal_pool, test = split_tables(table, seed)
        rng = np.random.default_rng(10_000 + seed)
        models = {n: build_scale_model(SCORES[n].scale, beta=args.beta, seed=seed).fit(head) for n in args.scores}
        s_test = {n: m.predict(test) for n, m in models.items()}
        for size in args.sizes:
            if size > cal_pool.n_images:
                continue
            cal = subsample_scenes(cal_pool, size, rng)
            for name, model in models.items():
                spec = SCORES[name]
                r_cal = residual(cal, spec.space) / model.predict(cal)
                for method in args.calibrators:
                    res = calibrate(method, cal, r_cal, test, None, args.alpha, args.beta)
                    lo, hi = interval(test.pix["pred"], res.lam_pix * s_test[name], spec.space)
                    m, _ = evaluate_intervals(test, lo, hi, args.alpha, args.beta)
                    rows.append({"seed": seed, "n_cal_target": size, "n_cal_images": cal.n_images,
                                 "n_cal_pixels": cal.n_pixels, "score": name, "calibrator": method,
                                 "lambda": res.lam_summary, **m})
        pd.DataFrame(rows).to_csv(out / "runs.csv", index=False)
        print(f"seed {seed} done")

    df = pd.DataFrame(rows)
    summ = df.groupby(["n_cal_target", "score", "calibrator"]).agg(
        n_cal_images=("n_cal_images", "mean"), n_cal_pixels=("n_cal_pixels", "mean"),
        cov_image_mean=("cov_image_mean", "mean"), cov_run_sd=("cov_image_mean", "std"),
        v4_rate=("v4_rate", "mean"), v4_run_sd=("v4_rate", "std"), frac_inf=("frac_infinite", "mean"),
        cov_image_sd=("cov_image_sd", "mean"), n_test=("n_test_images", "mean"),
    ).reset_index()
    # finite test set adds its own noise to the run-to-run spread
    summ["test_noise_sd"] = summ.cov_image_sd / summ.n_test**0.5
    summ["beta_sd_n_images"] = [beta_sd(int(n), args.alpha) for n in summ.n_cal_images]
    summ["beta_sd_n_pixels"] = [beta_sd(int(n), args.alpha) for n in summ.n_cal_pixels]
    summ.to_csv(out / "summary.csv", index=False)
    (out / "summary.md").write_text(to_markdown(summ))
    print(summ.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
