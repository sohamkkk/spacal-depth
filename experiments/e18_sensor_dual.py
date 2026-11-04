"""E18: A-dual under independent prediction-anchor and probe value sources (review 2 Nov).

On ARKitScenes (or any table with a `lidar` field), the budget-split draw of E17 is kept, but the values
the two roles report are chosen independently:
  pred_source   values at the structured prediction anchors (and head-train fit of m, s)
  probe_source  values at the random probes (the per-image conformal scores)
Coverage is always of the ground truth. The four cells Faro/Faro, iPad/Faro, Faro/iPad, iPad/iPad are the
pairs of (pred_source, probe_source) in {clean, lidar}² ("clean" = Faro / laser GT residual).

Kill (review): if a cell that uses iPad-valued probes claims truth coverage and the coverage CI excludes
0.90, that uncorrected claim is dead. Matched clean/clean should recover E17.

    python -m experiments.e18_sensor_dual --dataset arkitscenes --model unidepthv2-large \
        --pattern uniform --ks 200 --n-seeds 3 --draws 2
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from experiments.common import exchange_units, load_table, results_dir, split_tables, to_markdown, write_spec
from experiments.e11_anchors import (
    DIST_EDGES,
    a_dist,
    coverage_by_distance,
    eval_view,
    image_of,
    interval_from,
    log_err,
    prepare_dist,
    score_row,
)
from src.calibrators import anchor_dist as ad
from src.calibrators.anchor_noise import anchor_values
from src.calibrators.anchors import (
    PATTERNS,
    anchor_keys,
    draw_split_anchors,
    local_halfwidth,
    nearest_anchor_distance,
    with_values,
)
from src.calibrators.dual import dual_interval, probe_quantile, split_roles

SOURCES = ("clean", "lidar")
CELLS = [(p, q) for p in SOURCES for q in SOURCES]  # pred_source, probe_source


ENV_GRID = np.geomspace(1e-6, 0.5, 400)


def envelope_fit(head, cal, e_head: np.ndarray, e_cal: np.ndarray, alpha_e: float) -> dict:
    """Sensor-error envelope (Supp. D; analysis-plan amendment of 4 Nov). e = log reading - log truth.
    Bias: median of e on head-train pixels. Band of class c at lam: [Q_c(lam/2), Q_c(1 - lam/2)] of the corrected
    head-train error, clipped to contain 0. lam: the largest grid value with U/(U+1) L(lam) + 1/(U+1) <= alpha_e,
    L(lam) = mean over calibration scenes of the fraction of the scene's pixels outside their class band (conformal
    risk control; L is non-decreasing in lam). No feasible lam gives infinite bands."""
    beta = float(np.nanmedian(e_head))
    eh, ec = e_head - beta, e_cal - beta
    ch, cc = head.pix["lidar_conf"].astype(int), cal.pix["lidar_conf"].astype(int)
    n_cls = int(max(ch.max(), cc.max())) + 1
    srt = [np.sort(eh[(ch == c) & np.isfinite(eh)]) for c in range(n_cls)]

    def bands(lam):
        a, b = np.full(n_cls, np.inf), np.full(n_cls, np.inf)
        for c, s in enumerate(srt):
            if len(s) and lam > 0:
                a[c] = max(0.0, -s[int(np.floor(lam / 2 * (len(s) - 1)))])
                b[c] = max(0.0, s[int(np.ceil((1 - lam / 2) * (len(s) - 1)))])
        return a, b

    _, unit_img = np.unique(exchange_units(cal), return_inverse=True)
    unit_px = np.repeat(unit_img, cal.counts)
    n_units = int(unit_img.max()) + 1
    npx = np.bincount(unit_px, minlength=n_units)

    def loss(lam):
        a, b = bands(lam)
        out = ~((ec >= -a[cc]) & (ec <= b[cc]))
        return float(np.mean(np.bincount(unit_px, out, minlength=n_units) / npx))

    ok = lambda lam: n_units / (n_units + 1) * loss(lam) + 1 / (n_units + 1) <= alpha_e
    lo, hi = -1, len(ENV_GRID)  # invariant: ok(ENV_GRID[lo]) (lo = -1: lam = 0), not ok(ENV_GRID[hi])
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if ok(ENV_GRID[mid]) else (lo, mid)
    lam = float(ENV_GRID[lo]) if lo >= 0 else 0.0
    a, b = bands(lam)
    return {"beta": beta, "lam": lam, "a": a, "b": b, "cal_loss": loss(lam) if lam > 0 else 0.0, "n_units": n_units}


def mix_obs(d_pred: np.ndarray, d_probe: np.ndarray, probes: np.ndarray) -> np.ndarray:
    """Structured positions keep d_pred; probe positions take d_probe."""
    out = np.asarray(d_pred, dtype=np.float64).copy()
    out[probes] = np.asarray(d_probe, dtype=np.float64)[probes]
    return out


def run_seed(table, seed, args, log):
    head, cal, test = split_tables(table, seed)
    tabs = {"head": head, "cal": cal, "test": test}
    n_fit = min(head.n_images, args.dist_head_images)
    fit_img = np.zeros(head.n_images, bool)
    fit_img[np.random.default_rng([seed, 7]).choice(head.n_images, n_fit, replace=False)] = True
    head_fit = head.subset(np.flatnonzero(fit_img))
    fit_px = np.repeat(fit_img, head.counts)
    tabs["head_fit"] = head_fit
    d_gt = {n: log_err(t) for n, t in tabs.items()}
    d_gt["head_fit"] = d_gt["head"][fit_px]
    half_a = np.random.default_rng([seed, 8]).random(head_fit.n_images) < 0.5
    s = {n: np.ones(t.n_pixels) for n, t in tabs.items()}
    img_t = image_of(test)
    diag = np.hypot(test.img["height"], test.img["width"]).astype(np.float64)
    base_tag = {"seed": seed, "n_cal": cal.n_images, "n_test": test.n_images, "pattern": args.pattern}
    rows, cond, dist_rows, cov_store, env_rows = [], [], [], {}, []

    obs = {}
    for spec in SOURCES:
        rng_v = np.random.default_rng([seed, 1800 + (0 if spec == "clean" else 1)])
        obs[spec] = {n: anchor_values(tabs[n], d_gt[n], spec, rng_v) for n in tabs}

    env = None
    if args.envelope:
        env = envelope_fit(head, cal, obs["lidar"]["head"] - d_gt["head"], obs["lidar"]["cal"] - d_gt["cal"],
                           args.alpha_e)
        c_test = test.pix["lidar_conf"].astype(int)
        env_a, env_b = env["a"][c_test], env["b"][c_test]
        env_rows.append({**base_tag, "beta": env["beta"], "lam": env["lam"], "cal_loss": env["cal_loss"],
                         "n_cal_units": env["n_units"], "alpha_e": args.alpha_e,
                         **{f"a{c}": v for c, v in enumerate(env["a"])}, **{f"b{c}": v for c, v in enumerate(env["b"])}})
        log(f"  seed {seed} envelope: beta {env['beta']:+.4f} lam {env['lam']:.2e} a {np.round(env['a'], 4)} "
            f"b {np.round(env['b'], 4)} cal loss {env['cal_loss']:.4f} ({env['n_units']} scenes)")

    for k in args.ks:
        if k <= args.dist_k_random:
            raise ValueError(f"k={k} must exceed --dist-k-random={args.dist_k_random}")
        for draw_i in range(args.draws):
            t0 = time.perf_counter()
            rng_d = np.random.default_rng([seed, int(k) + 1, draw_i, 99])

            def draw_one(name, t):
                key = None if args.pattern == "uniform" else anchor_keys(args.pattern, t.pix, t.offsets, rng_d)
                return draw_split_anchors(d_gt[name], t.offsets, k, args.dist_k_random, rng_d, key)

            sdraws = {n: draw_one(n, tabs[n]) for n in ("cal", "test", "head_fit")}
            vt2, st2 = eval_view(test, sdraws["test"].eval)
            dist_bin = np.digitize(
                nearest_anchor_distance(test.pix["row"], test.pix["col"], test.offsets,
                                        sdraws["test"].anchor, diag)[st2],
                DIST_EDGES,
            )
            probes = {n: split_roles(sdraws[n], d_gt[n], tabs[n].offsets)[1] for n in ("cal", "test", "head_fit")}

            for pred_src, probe_src in args.cells:
                d_pred = obs[pred_src]
                d_probe = obs[probe_src]
                pdraws = {n: split_roles(sdraws[n], d_pred[n], tabs[n].offsets)[0]
                          for n in ("cal", "test", "head_fit")}
                d_mix = {n: mix_obs(d_pred[n], d_probe[n], probes[n]) for n in ("cal", "test", "head_fit")}
                sdraws_obs = {n: with_values(sdraws[n], d_mix[n], tabs[n].offsets)
                              for n in ("cal", "test", "head_fit")}

                cfg_fields = prepare_dist("auto", head_fit, d_pred["head_fit"], half_a, args.alphas[0],
                                          pdraws, tabs, d_pred)
                cfg_split = prepare_dist("auto", head_fit, d_mix["head_fit"], half_a, args.alphas[0],
                                         sdraws_obs, tabs, d_mix)

                for alpha in args.alphas:
                    tag = {**base_tag, "k": str(k), "draw": draw_i, "base": "S2-log", "alpha": alpha,
                           "pred_source": pred_src, "probe_source": probe_src,
                           "dual_kind": cfg_fields[0].kind, "dual_gamma": args.dual_gamma,
                           "dist_split_kind": cfg_split[0].kind}
                    methods, local_inf = {}, {}

                    cfg3, fields3, _ = cfg_fields
                    model3, _ = ad.fit_dist_model(
                        head_fit, d_pred["head_fit"], s["head_fit"], pdraws["head_fit"],
                        cfg3, alpha, half_a, use_g=not args.dist_no_g, single_bin=args.dist_single_bin,
                        fields=fields3["head_fit"])
                    q3 = probe_quantile(model3, d_probe["test"], s["test"], fields3["test"], probes["test"],
                                        test.offsets, alpha, args.dual_gamma)
                    lo3, hi3 = dual_interval(model3, test.pix["pred"], s["test"], fields3["test"], q3,
                                             test.offsets, args.dual_gamma)
                    methods["A-dual"] = (lo3[st2], hi3[st2])
                    local_inf["A-dual"] = float(np.mean(~np.isfinite(q3)))
                    if env is not None and probe_src == "lidar":
                        qs = probe_quantile(model3, d_probe["test"] - env["beta"], s["test"], fields3["test"],
                                            probes["test"], test.offsets, args.alpha_s, args.dual_gamma)
                        los, his = dual_interval(model3, test.pix["pred"], s["test"], fields3["test"], qs,
                                                 test.offsets, args.dual_gamma)
                        methods["A-dual-s"] = (los[st2], his[st2])
                        methods["A-dual-env"] = (los[st2] * np.exp(-env_b[st2]), his[st2] * np.exp(env_a[st2]))
                        local_inf["A-dual-s"] = local_inf["A-dual-env"] = float(np.mean(~np.isfinite(qs)))

                    dt_probe = with_values(sdraws["test"], d_probe["test"], test.offsets)
                    qloc = local_halfwidth(d_probe["test"], s["test"], test.offsets, dt_probe, alpha)
                    methods["A-local-split"] = interval_from(test, dt_probe.shift_half, qloc[img_t[st2]], st2)
                    local_inf["A-local-split"] = float(np.mean(~np.isfinite(qloc)))

                    cfg2, fields2, _ = cfg_split
                    ld, hd = a_dist(cfg2, fields2, s, sdraws_obs, tabs, d_mix, half_a, alpha, args)
                    methods["A-dist-split"] = (ld, hd)
                    methods["A-dual-both"] = (np.minimum(lo3[st2], ld), np.maximum(hi3[st2], hd))
                    local_inf["A-dual-both"] = local_inf["A-dual"]
                    ll, hl = methods["A-local-split"]
                    methods["A-dist-both"] = (np.minimum(ld, ll), np.maximum(hd, hl))
                    local_inf["A-dist-both"] = local_inf["A-local-split"]

                    for name, (lo, hi) in methods.items():
                        m, cov_img = score_row(vt2, lo, hi, alpha, args.beta)
                        rows.append({**tag, "method": name, **m, "frac_img_local_inf": local_inf.get(name, 0.0)})
                        cov_store.setdefault((str(k), pred_src, probe_src, name, alpha), []).append(cov_img)
                        dist_rows += coverage_by_distance(vt2, lo, hi, dist_bin, {**tag, "method": name})

            log(f"  seed {seed} k={k} draw {draw_i}: {time.perf_counter() - t0:.0f}s")

    for (k, pred_src, probe_src, name, alpha), covs in cov_store.items():
        if len({len(c) for c in covs}) != 1:
            continue
        avg = np.mean(covs, axis=0)
        cond.append({**base_tag, "k": k, "pred_source": pred_src, "probe_source": probe_src,
                     "method": name, "alpha": alpha, "n_draws": len(covs),
                     "cond_short_frac": float(np.mean(avg < 1 - alpha - 0.05)),
                     "cond_p05": float(np.quantile(avg, 0.05)), "cond_min": float(avg.min())})
    return rows, cond, dist_rows, env_rows


def summarize(runs: pd.DataFrame, cond: pd.DataFrame) -> pd.DataFrame:
    keys = ["k", "pred_source", "probe_source", "method", "alpha"]
    cols = ["cov_image_mean", "cov_image_sd", "cov_image_p10", "frac_img_cov_below_1ma_m10", "v4_rate",
            "rel_width", "median_rel_width", "rel_interval_score", "rel_interval_score_robust",
            "rel_interval_score_imgmed", "frac_img_short", "frac_infinite", "frac_uninformative", "lower_miss_mean",
            "lower_miss_img_p90"]
    summ = runs.groupby(keys)[cols].mean()
    summ["cov_image_mean_se"] = runs.groupby(keys)["cov_image_mean"].std() / np.sqrt(runs.groupby(keys).size())
    return summ.join(cond.groupby(keys)[["cond_short_frac", "cond_p05"]].mean()).reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--exp-id", default="e18_sensor")
    ap.add_argument("--pattern", choices=PATTERNS, default="uniform")
    ap.add_argument("--ks", type=int, nargs="+", default=[200])
    ap.add_argument("--draws", type=int, default=2)
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.1])
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--dist-k-random", type=int, default=20)
    ap.add_argument("--dual-gamma", type=float, default=0.5)
    ap.add_argument("--dist-head-images", type=int, default=200)
    ap.add_argument("--dist-single-bin", action="store_true")
    ap.add_argument("--dist-no-g", action="store_true")
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--cells", nargs="+", default=[f"{p}/{q}" for p, q in CELLS],
                    help="pred_source/probe_source cells, e.g. clean/lidar lidar/lidar")
    ap.add_argument("--envelope", action="store_true",
                    help="add A-dual-s (bias-corrected probes, level 1 - alpha_s) and A-dual-env (Supp. D) in cells "
                         "with iPad probes")
    ap.add_argument("--alpha-s", type=float, default=0.05)
    ap.add_argument("--alpha-e", type=float, default=0.05)
    ap.add_argument("--seed-start", type=int, default=None,
                    help="run seeds seed_start .. seed_start + n_seeds - 1 (output dir gets a __seeds<a>_<b> suffix)")
    args = ap.parse_args()
    args.cells = [tuple(c.split("/")) for c in args.cells]
    if any(p not in SOURCES or q not in SOURCES for p, q in args.cells):
        raise SystemExit(f"--cells must be pairs from {SOURCES}")

    table = load_table(args.dataset, args.model)
    table.pix.pop("label", None)
    if args.max_images:
        table = table.subset(np.arange(min(args.max_images, table.n_images)))
    if not table.has("lidar"):
        raise SystemExit(f"{args.dataset}/{args.model} has no lidar field")
    if args.envelope and not table.has("lidar_conf"):
        raise SystemExit(f"{args.dataset}/{args.model} has no lidar_conf field (needed by --envelope)")

    suffix = "" if args.pattern == "uniform" else f"__{args.pattern}"
    seed0 = args.seed_start or 0
    if args.seed_start is not None:
        suffix += f"__seeds{seed0}_{seed0 + args.n_seeds - 1}"
    out = results_dir(args.exp_id, f"{args.dataset}__{args.model}{suffix}")
    write_spec(out, {**vars(args), "cells": [f"{p}/{q}" for p, q in args.cells],
                     "n_images": table.n_images, "n_pixels": table.n_pixels})
    print(f"{args.dataset}/{args.model}: {table.n_images} images, {table.n_pixels:,} px; "
          f"ks={args.ks} pattern={args.pattern} cells={args.cells}", flush=True)

    rows, cond, dists, envs = [], [], [], []
    for seed in range(seed0, seed0 + args.n_seeds):
        t0 = time.perf_counter()
        r, c, dr, er = run_seed(table, seed, args, print)
        rows += r
        cond += c
        dists += dr
        envs += er
        runs_df, cond_df = pd.DataFrame(rows), pd.DataFrame(cond)
        runs_df.to_csv(out / "runs.csv", index=False)
        cond_df.to_csv(out / "cond.csv", index=False)
        if dists:
            pd.DataFrame(dists).to_csv(out / "dist.csv", index=False)
        if envs:
            pd.DataFrame(envs).to_csv(out / "env.csv", index=False)
        print(f"seed {seed} finished in {time.perf_counter() - t0:.0f}s", flush=True)

    if dists:
        dist_summ = (pd.DataFrame(dists)
                     .groupby(["k", "pred_source", "probe_source", "method", "alpha", "dist_bin"], sort=False)
                     [["cov_image_mean", "n_img", "frac_px"]].mean().reset_index())
        dist_summ.to_csv(out / "dist_summary.csv", index=False)
    summ = summarize(runs_df, cond_df)
    summ.to_csv(out / "summary.csv", index=False)
    (out / "summary.md").write_text(to_markdown(summ) + "\n")
    show = ["k", "pred_source", "probe_source", "method", "cov_image_mean", "cov_image_mean_se",
            "median_rel_width", "cond_short_frac"]
    print(summ[show].round(4).to_string(index=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
