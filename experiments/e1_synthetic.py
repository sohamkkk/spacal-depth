"""E1: controlled simulation of why the image (not the pixel) is the unit.

Per-image log errors e_ij = g_i + sigma_l * z_ij, with g_i ~ N(0, sigma_g)
a global scale error and z a Gaussian random field of correlation length
ell. Score r = |e|. For each (ell, sigma_g) and R repetitions:
  - lambda from K0 (pooled pixels), K1 (image CRC), V4 (image event)
  - test: mean pixel coverage, per-image coverage SD, V4 rate
  - across repetitions: SD of the realised mean coverage, against the
    Beta spread predicted with n = #calibration images vs n = #pixels.
Finding to check: K0 and K1 give nearly the same lambda, but run-to-run
spread follows the image count, and only V4 controls per-image coverage.

    python -m experiments.e1_synthetic --reps 100
"""

import argparse

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist

from experiments.common import results_dir, to_markdown, write_spec
from src.calibrators.crc import crc_lambda_exact
from src.calibrators.split_cp import image_event_lambda, pooled_pixel_lambda


def grf_batch(rng, n, size, ell):
    """n unit-variance Gaussian random fields (squared-exponential spectrum)."""
    noise = rng.standard_normal((n, size, size))
    if ell <= 0:
        return noise
    k = np.fft.fftfreq(size)
    kk = k[:, None] ** 2 + k[None, :] ** 2
    filt = np.exp(-2 * (np.pi * ell) ** 2 * kk)
    f = np.real(np.fft.ifft2(np.fft.fft2(noise) * np.sqrt(filt)))
    return f / f.std(axis=(1, 2), keepdims=True)


def beta_sd(n, alpha):
    """SD of the coverage of split CP with n exchangeable scores."""
    l = np.floor((n + 1) * alpha)
    return float(beta_dist(n + 1 - l, l).std()) if l >= 1 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--n-cal", type=int, default=200)
    ap.add_argument("--n-test", type=int, default=300)
    ap.add_argument("--size", type=int, default=48)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--ells", type=float, nargs="+", default=[0, 2, 8, 24])
    ap.add_argument("--sigma-gs", type=float, nargs="+", default=[0.0, 0.05, 0.1, 0.2])
    ap.add_argument("--sigma-l", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    m = args.size**2
    off_cal = np.arange(args.n_cal + 1) * m

    rows = []
    for ell in args.ells:
        for sg in args.sigma_gs:
            for rep in range(args.reps):
                n = args.n_cal + args.n_test
                e = rng.normal(0, sg, (n, 1, 1)) + args.sigma_l * grf_batch(rng, n, args.size, ell)
                r = np.abs(e).reshape(n, m)
                r_cal, r_test = r[: args.n_cal].ravel(), r[args.n_cal :]
                lams = {
                    "K0": pooled_pixel_lambda(r_cal, args.alpha),
                    "K1": crc_lambda_exact(r_cal, off_cal, args.alpha),
                    "V4": image_event_lambda(r_cal, off_cal, args.alpha, args.beta),
                }
                for k, lam in lams.items():
                    cov_img = (r_test <= lam).mean(axis=1)
                    rows.append({"ell": ell, "sigma_g": sg, "rep": rep, "calibrator": k, "lambda": lam,
                                 "cov_mean": cov_img.mean(), "cov_img_sd": cov_img.std(ddof=1),
                                 "v4_rate": float(np.mean(cov_img >= 1 - args.beta))})
            print(f"ell={ell} sigma_g={sg}: done")

    df = pd.DataFrame(rows)
    out = results_dir("e1")
    df.to_csv(out / "runs.csv", index=False)
    summ = df.groupby(["ell", "sigma_g", "calibrator"]).agg(
        lambda_mean=("lambda", "mean"), cov_mean=("cov_mean", "mean"), cov_run_sd=("cov_mean", "std"),
        cov_img_sd=("cov_img_sd", "mean"), v4_rate=("v4_rate", "mean"),
    ).reset_index()
    summ["beta_sd_n_images"] = beta_sd(args.n_cal, args.alpha)
    summ["beta_sd_n_pixels"] = beta_sd(args.n_cal * m, args.alpha)
    summ.to_csv(out / "summary.csv", index=False)
    (out / "summary.md").write_text(to_markdown(summ))
    write_spec(out, {"experiment": "E1 synthetic", **vars(args)})
    print(summ.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
