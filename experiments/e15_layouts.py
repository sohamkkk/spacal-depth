"""E15: anchor layout at calibration vs at deployment, and the LSCP baseline.

Every cross-image anchor guarantee assumes calibration and test images get their anchors by the same rule.
Test images here use --pattern; the calibration images are drawn under each of the five layouts, giving
  <method>@<layout>  calibrated under that layout (the matched case is <method>@<--pattern>)
  <method>@mix       each calibration image gets one of the five layouts at random (valid for that mixture)
  <method>@max       the largest threshold over the five single-layout calibrations (per distance bin for
                     A-dist): valid under each of the five layouts, whichever is met at test time
for <method> in A-rescale, A-hybrid, A-dist (the A-dist-auto variant: kind and bandwidths with the best head
score). All head-train fits (tau, A-dist's kind, bandwidths, tau, g, bin merges, cap) use the mixture of
layouts, so the variants of one method differ only in their calibration anchors.
Methods without calibration images: A-local, LSCP (src/calibrators/lscp.py, S2-log only); and the hulls
A-both@max = A-hybrid@max hull A-local, A-dist-both@max = A-dist@max hull A-local.

    python -m experiments.e15_layouts --dataset kitti --model dav2-outdoor-large --pattern rows

Outputs (results/e15/<dataset>__<model>__<pattern>/): runs.csv, cond.csv, dist.csv, dist_summary.csv,
summary.{csv,md}, spec.json
"""

import argparse
import time

import numpy as np
import pandas as pd

from experiments.common import load_table, results_dir, split_tables, to_markdown, write_spec
from experiments.e11_anchors import (
    DIST_EDGES,
    coverage_by_distance,
    crc_units,
    eval_view,
    image_of,
    interval_from,
    log_err,
    pick_tau,
    score_row,
    shape_target_table,
    summarize,
)
from src.calibrators import anchor_dist as ad
from src.calibrators.anchors import (
    PATTERNS,
    anchor_keys,
    draw_anchors,
    local_halfwidth,
    mix_draws,
    nearest_anchor_distance,
    pick,
)
from src.calibrators.lscp import lscp_offsets
from src.data.splits import exchange_units
from src.scores.learned import LearnedQuantileScale

CAL_VARIANTS = (*PATTERNS, "mix", "max")


def draw_layout(t, d, k, layout, rng):
    key = None if layout == "uniform" else anchor_keys(layout, t.pix, t.offsets, rng)
    return draw_anchors(d, t.offsets, k, rng, key)


def draw_mix(t, d, k, rng):
    """Every image under one of the five layouts, chosen uniformly at random."""
    per_layout = [draw_layout(t, d, k, lay, rng) for lay in PATTERNS]
    return mix_draws(per_layout, rng.integers(len(PATTERNS), size=t.n_images), t.offsets)


def run_seed(table, seed, args, log):
    head, cal, test = split_tables(table, seed)
    d = {n: log_err(t) for n, t in (("head", head), ("cal", cal), ("test", test))}
    n_fit = min(head.n_images, args.dist_head_images)
    fit_img = np.zeros(head.n_images, bool)
    fit_img[np.random.default_rng([seed, 7]).choice(head.n_images, n_fit, replace=False)] = True
    fit_px = np.repeat(fit_img, head.counts)
    head_fit = head.subset(np.flatnonzero(fit_img))
    d["head_fit"] = d["head"][fit_px]
    half_a = np.random.default_rng([seed, 8]).random(head_fit.n_images) < 0.5

    t0 = time.perf_counter()
    s = {"S2-log": {n: np.ones(t.n_pixels) for n, t in (("head", head), ("cal", cal), ("test", test))}}
    if "S5-shape" in args.bases:
        shape = LearnedQuantileScale(quantile=1 - args.beta, seed=seed).fit(shape_target_table(head))
        s["S5-shape"] = {n: shape.predict(t) for n, t in (("head", head), ("cal", cal), ("test", test))}
        lo, hi = np.quantile(s["S5-shape"]["head"], [0.01, 0.99])
        s["S5-shape"] = {n: np.clip(v, lo, hi) for n, v in s["S5-shape"].items()}
    for v in s.values():
        v["head_fit"] = v["head"][fit_px]
    log(f"  seed {seed}: scale models fit in {time.perf_counter() - t0:.0f}s")

    img_c, img_t = image_of(cal), image_of(test)
    units_full = exchange_units(cal)
    diag = np.hypot(test.img["height"], test.img["width"]).astype(np.float64)
    base_tag = {"seed": seed, "n_cal": cal.n_images, "n_test": test.n_images, "test_pattern": args.pattern}
    rows, cond, dist_rows, cov_store = [], [], [], {}

    for k in args.ks:
        for draw_i in range(args.draws):
            t0 = time.perf_counter()
            rng = np.random.default_rng([seed, k + 1, draw_i, 15])
            dt = draw_layout(test, d["test"], k, args.pattern, rng)
            vt, st = eval_view(test, dt.eval)
            dist_bin = np.digitize(nearest_anchor_distance(test.pix["row"], test.pix["col"], test.offsets,
                                                           dt.anchor, diag)[st], DIST_EDGES)
            cal_choice = rng.integers(len(PATTERNS), size=cal.n_images)
            dc = {lay: draw_layout(cal, d["cal"], k, lay, rng) for lay in PATTERNS}
            dc["mix"] = mix_draws([dc[lay] for lay in PATTERNS], cal_choice, cal.offsets)
            views = {c: eval_view(cal, dr.eval) for c, dr in dc.items()}
            units = {c: exchange_units(v) for c, (v, _) in views.items()}
            dh, dhf = draw_mix(head, d["head"], k, rng), draw_mix(head_fit, d["head_fit"], k, rng)

            cfg = ad.select_model(head_fit, d["head_fit"], np.ones(head_fit.n_pixels), dhf, "auto",
                                  args.alphas[0], half_a).cfg
            f_hf = ad.anchor_fields(head_fit, d["head_fit"], dhf, cfg)
            f_cal = {lay: ad.anchor_fields(cal, d["cal"], dc[lay], cfg) for lay in PATTERNS}
            px_choice = cal_choice[img_c]
            f_cal["mix"] = ad.Fields(*(pick([getattr(f_cal[lay], a) for lay in PATTERNS], px_choice)
                                       for a in ("shift", "sigma", "delta")))
            f_t = ad.anchor_fields(test, d["test"], dt, cfg)

            local, lscp = {}, {}
            pred_t = test.pix["pred"][st].astype(np.float64)
            lscp_off = lscp_offsets(test, d["test"], dt, args.alphas, seed=1000 * seed + 10 * k + draw_i)
            for alpha in args.alphas:
                q = local_halfwidth(d["test"], np.ones(test.n_pixels), test.offsets, dt, alpha)
                local[alpha] = interval_from(test, dt.shift_half, q[img_t[st]], st)
                lo, hi = lscp_off[alpha]
                with np.errstate(over="ignore"):
                    lscp[alpha] = (pred_t * np.exp(lo), pred_t * np.exp(hi))

            for base in args.bases:
                s_c, s_t = s[base]["cal"], s[base]["test"]
                for alpha in args.alphas:
                    tag = {**base_tag, "k": str(k), "draw": draw_i, "base": base, "alpha": alpha,
                           "dist_auto_kind": cfg.kind}
                    tau = pick_tau(head, d["head"], s[base]["head"], dh, alpha, rng)
                    model, _ = ad.fit_dist_model(head_fit, d["head_fit"], s[base]["head_fit"], dhf, cfg, alpha,
                                                 half_a, fields=f_hf)
                    lam = {"A-rescale": {}, "A-hybrid": {}, "A-dist": {}}
                    for c in (*PATTERNS, "mix"):
                        (vc, sc), dr = views[c], dc[c]
                        res = np.abs(d["cal"] - dr.shift[img_c])
                        lam["A-rescale"][c] = crc_units((res / s_c)[sc], vc.offsets, units[c], alpha)
                        lam["A-hybrid"][c] = crc_units((res / (s_c * (dr.sigma[img_c] + tau)))[sc], vc.offsets,
                                                       units[c], alpha)
                        lam["A-dist"][c] = ad.calibrate(model, cal, d["cal"], s_c, dr, alpha, units=units_full,
                                                        fields=f_cal[c])
                    for v in lam.values():
                        v["max"] = np.max([v[lay] for lay in PATTERNS], axis=0)

                    methods = {}
                    den_t = (s_t * (dt.sigma[img_t] + tau))[st]
                    for c in CAL_VARIANTS:
                        methods[f"A-rescale@{c}"] = interval_from(test, dt.shift, lam["A-rescale"][c] * s_t[st], st)
                        methods[f"A-hybrid@{c}"] = interval_from(test, dt.shift, lam["A-hybrid"][c] * den_t, st)
                        lo, hi = ad.interval(model, test, s_t, lam["A-dist"][c], f_t)
                        methods[f"A-dist@{c}"] = (lo[st], hi[st])
                    if base == "S2-log":
                        methods["A-local"], methods["LSCP"] = local[alpha], lscp[alpha]
                    (ll, hl) = local[alpha]
                    for name in ("A-hybrid", "A-dist"):
                        lo, hi = methods[f"{name}@max"]
                        both = "A-both" if name == "A-hybrid" else "A-dist-both"
                        methods[f"{both}@max"] = (np.minimum(lo, ll), np.maximum(hi, hl))

                    for name, (lo, hi) in methods.items():
                        m, cov_img = score_row(vt, lo, hi, alpha, args.beta)
                        rows.append({**tag, "method": name, **m})
                        cov_store.setdefault((str(k), base, name, alpha), []).append(cov_img)
                        dist_rows += coverage_by_distance(vt, lo, hi, dist_bin, {**tag, "method": name})
            log(f"  seed {seed} k={k} draw {draw_i}: {cfg.kind}, {time.perf_counter() - t0:.0f}s")

    for (k, base, name, alpha), covs in cov_store.items():
        if len({len(c) for c in covs}) != 1:
            continue
        avg = np.mean(covs, axis=0)
        cond.append({**base_tag, "k": k, "base": base, "method": name, "alpha": alpha, "n_draws": len(covs),
                     "cond_short_frac": float(np.mean(avg < 1 - alpha - 0.05)),
                     "cond_p05": float(np.quantile(avg, 0.05)), "cond_min": float(avg.min())})
    return rows, cond, dist_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--exp-id", default="e15")
    ap.add_argument("--pattern", choices=PATTERNS, required=True, help="test-time anchor layout")
    ap.add_argument("--ks", type=int, nargs="+", default=[20, 50, 200])
    ap.add_argument("--draws", type=int, default=2)
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.1, 0.05])
    ap.add_argument("--bases", nargs="+", choices=("S2-log", "S5-shape"), default=["S2-log"])
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--dist-head-images", type=int, default=200)
    ap.add_argument("--max-images", type=int, default=0, help="debug: use only the first N images")
    args = ap.parse_args()

    table = load_table(args.dataset, args.model)
    table.pix.pop("label", None)
    if args.max_images:
        table = table.subset(np.arange(min(args.max_images, table.n_images)))
    out = results_dir(args.exp_id, f"{args.dataset}__{args.model}__{args.pattern}")
    write_spec(out, {**vars(args), "n_images": table.n_images, "n_pixels": table.n_pixels})
    print(f"{args.dataset}/{args.model}: {table.n_images} images; test pattern {args.pattern}, ks={args.ks}")

    rows, cond, dists = [], [], []
    for seed in range(args.n_seeds):
        t0 = time.perf_counter()
        r, c, dr = run_seed(table, seed, args, print)
        rows, cond, dists = rows + r, cond + c, dists + dr
        pd.DataFrame(rows).to_csv(out / "runs.csv", index=False)
        pd.DataFrame(cond).to_csv(out / "cond.csv", index=False)
        pd.DataFrame(dists).to_csv(out / "dist.csv", index=False)
        print(f"seed {seed} finished in {time.perf_counter() - t0:.0f}s")

    dist_summ = (pd.DataFrame(dists).groupby(["k", "base", "method", "alpha", "dist_bin"], sort=False)
                 [["cov_image_mean", "n_img", "frac_px"]].mean().reset_index())
    dist_summ.to_csv(out / "dist_summary.csv", index=False)
    summ = summarize(pd.DataFrame(rows), pd.DataFrame(cond))
    summ.to_csv(out / "summary.csv", index=False)
    (out / "summary.md").write_text(to_markdown(summ) + "\n")
    show = ["k", "base", "method", "alpha", "cov_image_mean", "median_rel_width", "cond_short_frac"]
    print(summ[show].round(4).to_string(index=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
