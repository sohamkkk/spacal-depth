"""E16: anchors whose values are wrong — sensor noise, gross outliers, a scale bias, and real LiDAR on KITTI.

Anchor positions follow --pattern (lidar: uniform among the pixels with a raw LiDAR return); their values come
from the anchor source (src/calibrators/anchor_noise.py). Coverage is always of the ground truth. Per source:
  <method>          head-train and calibration anchors from the same sensor as the test anchors (matched)
  <method>@clean    head-train and calibration anchors at the same positions but with GT values, test anchors
                    from the sensor: calibrating with clean anchors and deploying a noisy sensor
for <method> in A-rescale, A-hybrid, A-dist (auto variant); plus A-local and LSCP, which use only the test
image's anchor values, and the hulls A-both = A-hybrid hull A-local, A-dist-both = A-dist hull A-local.
A-dist's variant and bandwidths are chosen once per anchor draw on clean head-train anchors (a head-train-only
choice, so validity is unaffected); its tau, g, bin merges and cap are fitted per source. S2-log base only.

    python -m experiments.e16_noisy --dataset kitti --model dav2-outdoor-large
    python -m experiments.e16_noisy --dataset nyu --model dav2-indoor-large --sources clean gauss:0.05

Outputs (results/e16/<dataset>__<model>__<pattern>/): runs.csv, cond.csv, dist.csv, anchor_err.csv,
dist_summary.csv, summary.{csv,md}, spec.json
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
    summarize,
)
from src.calibrators import anchor_dist as ad
from src.calibrators.anchor_noise import anchor_values, available, parse
from src.calibrators.anchors import PATTERNS, anchor_keys, draw_anchors, local_halfwidth, nearest_anchor_distance, with_values
from src.calibrators.lscp import lscp_offsets
from src.data.splits import exchange_units

SOURCES = ("clean", "gauss:0.02", "gauss:0.05", "outlier:0.05", "outlier:0.1", "bias:0.05", "lidar")
NAMES = ("head", "head_fit", "cal", "test")


def draw_positions(tabs, d, k, pattern, lidar, rng):
    """Anchor positions (with GT values) on every table; lidar: uniform among pixels with a return."""
    out = {}
    for n in NAMES:
        t = tabs[n]
        key = None if lidar or pattern == "uniform" else anchor_keys(pattern, t.pix, t.offsets, rng)
        out[n] = draw_anchors(d[n], t.offsets, k, rng, key, avail=available(t, "lidar") if lidar else None)
    return out


def calibrate_all(tabs, d, dr, vals, cfg, args, half_a, units_full, rng_seed):
    """Head-train fits and calibration lambdas for A-rescale, A-hybrid and A-dist from anchors dr / values vals.
    A-dist's variant and bandwidths (cfg) are given; tau, g, bin merges and the cap are fitted here."""
    head, head_fit, cal = tabs["head"], tabs["head_fit"], tabs["cal"]
    ones = {n: np.ones(tabs[n].n_pixels) for n in NAMES}
    f_hf = ad.anchor_fields(head_fit, vals["head_fit"], dr["head_fit"], cfg)
    f_cal = ad.anchor_fields(cal, vals["cal"], dr["cal"], cfg)
    vc, sc = eval_view(cal, dr["cal"].eval)
    units = exchange_units(vc)
    res = np.abs(d["cal"] - dr["cal"].shift[image_of(cal)])
    hyb_den = dr["cal"].sigma[image_of(cal)]
    out = {"cfg": cfg}
    for alpha in args.alphas:
        tau = pick_tau(head, d["head"], ones["head"], dr["head"], alpha, np.random.default_rng(rng_seed))
        model, _ = ad.fit_dist_model(head_fit, d["head_fit"], ones["head_fit"], dr["head_fit"], cfg, alpha, half_a,
                                     fields=f_hf)
        out[alpha] = {
            "tau": tau,
            "model": model,
            "A-rescale": crc_units(res[sc], vc.offsets, units, alpha),
            "A-hybrid": crc_units((res / (hyb_den + tau))[sc], vc.offsets, units, alpha),
            "A-dist": ad.calibrate(model, cal, d["cal"], ones["cal"], dr["cal"], alpha, units=units_full,
                                   fields=f_cal),
        }
    return out


def cross_image_intervals(c, test, dt, vals_t, st, alpha):
    img_t = image_of(test)
    p = c[alpha]
    f_t = ad.anchor_fields(test, vals_t, dt, c["cfg"])
    lo, hi = ad.interval(p["model"], test, np.ones(test.n_pixels), p["A-dist"], f_t)
    return {
        "A-rescale": interval_from(test, dt.shift, np.full(st.sum(), p["A-rescale"]), st),
        "A-hybrid": interval_from(test, dt.shift, p["A-hybrid"] * (dt.sigma[img_t] + p["tau"])[st], st),
        "A-dist": (lo[st], hi[st]),
    }


def anchor_error_row(tag, d_t, vals_t, dt):
    e = (vals_t - d_t)[dt.anchor]
    a = np.abs(e)
    return {**tag, "n_anchor_mean": float(dt.k_img.mean()), "n_anchor_min": int(dt.k_img.min()),
            "err_median_abs": float(np.median(a)), "err_p90_abs": float(np.quantile(a, 0.9)),
            "err_mean": float(e.mean()), "frac_err_gt_10pct": float(np.mean(a > np.log(1.1)))}


def run_seed(table, seed, args, sources, log):
    head, cal, test = split_tables(table, seed)
    tabs = {"head": head, "cal": cal, "test": test}
    n_fit = min(head.n_images, args.dist_head_images)
    fit_img = np.zeros(head.n_images, bool)
    fit_img[np.random.default_rng([seed, 7]).choice(head.n_images, n_fit, replace=False)] = True
    tabs["head_fit"] = head.subset(np.flatnonzero(fit_img))
    d = {n: log_err(t) for n, t in tabs.items()}
    half_a = np.random.default_rng([seed, 8]).random(tabs["head_fit"].n_images) < 0.5
    units_full = exchange_units(cal)
    img_t = image_of(test)
    diag = np.hypot(test.img["height"], test.img["width"]).astype(np.float64)
    base_tag = {"seed": seed, "n_cal": cal.n_images, "n_test": test.n_images, "pattern": args.pattern}
    rows, cond, dist_rows, err_rows, cov_store = [], [], [], [], {}

    for k in args.ks:
        for draw_i in range(args.draws):
            t0 = time.perf_counter()
            rng = np.random.default_rng([seed, k + 1, draw_i, 16])
            pos = {False: draw_positions(tabs, d, k, args.pattern, False, rng)}
            if any(parse(s)[0] == "lidar" for s in sources):
                pos[True] = draw_positions(tabs, d, k, args.pattern, True, rng)
            # A-dist's variant is chosen once per position set, on clean head-train anchors
            cfg = {lid: ad.select_model(tabs["head_fit"], d["head_fit"], np.ones(tabs["head_fit"].n_pixels),
                                        p["head_fit"], "auto", args.alphas[0], half_a).cfg for lid, p in pos.items()}
            clean_cal = {}
            for si, spec in enumerate(sources):
                lidar = parse(spec)[0] == "lidar"
                p = pos[lidar]
                vrng = np.random.default_rng([seed, k + 1, draw_i, 1600 + si])
                vals = {n: anchor_values(tabs[n], d[n], spec, vrng) for n in NAMES}
                dr = {n: with_values(p[n], vals[n], tabs[n].offsets) for n in NAMES}
                dt = dr["test"]
                vt, st = eval_view(test, dt.eval)
                dist_bin = np.digitize(nearest_anchor_distance(test.pix["row"], test.pix["col"], test.offsets,
                                                               dt.anchor, diag)[st], DIST_EDGES)
                tag0 = {**base_tag, "k": str(k), "draw": draw_i, "source": spec}
                err_rows.append(anchor_error_row(tag0, d["test"], vals["test"], dt))

                cal_args = (cfg[lidar], args, half_a, units_full, [seed, k, draw_i, 99])
                cals = {"": calibrate_all(tabs, d, dr, vals, *cal_args)}
                if spec != "clean":
                    if lidar not in clean_cal:
                        clean_cal[lidar] = calibrate_all(tabs, d, p, d, *cal_args)
                    cals["@clean"] = clean_cal[lidar]

                lscp = lscp_offsets(test, vals["test"], dt, args.alphas, seed=1000 * seed + 10 * k + draw_i) \
                    if args.lscp else {}
                pred_t = test.pix["pred"][st].astype(np.float64)
                for alpha in args.alphas:
                    tag = {**tag0, "alpha": alpha, "dist_auto_kind": cals[""]["cfg"].kind}
                    methods = {}
                    for suffix, c in cals.items():
                        for name, iv in cross_image_intervals(c, test, dt, vals["test"], st, alpha).items():
                            methods[name + suffix] = iv
                    q = local_halfwidth(vals["test"], np.ones(test.n_pixels), test.offsets, dt, alpha)
                    ll, hl = methods["A-local"] = interval_from(test, dt.shift_half, q[img_t[st]], st)
                    if args.lscp:
                        lo, hi = lscp[alpha]
                        with np.errstate(over="ignore"):
                            methods["LSCP"] = (pred_t * np.exp(lo), pred_t * np.exp(hi))
                    for name, both in (("A-hybrid", "A-both"), ("A-dist", "A-dist-both")):
                        lo, hi = methods[name]
                        methods[both] = (np.minimum(lo, ll), np.maximum(hi, hl))

                    for name, (lo, hi) in methods.items():
                        m, cov_img = score_row(vt, lo, hi, alpha, args.beta)
                        rows.append({**tag, "method": name, **m})
                        cov_store.setdefault((str(k), spec, name, alpha), []).append(cov_img)
                        dist_rows += coverage_by_distance(vt, lo, hi, dist_bin, {**tag, "method": name})
            log(f"  seed {seed} k={k} draw {draw_i}: {time.perf_counter() - t0:.0f}s")

    for (k, spec, name, alpha), covs in cov_store.items():
        if len({len(c) for c in covs}) != 1:
            continue
        avg = np.mean(covs, axis=0)
        cond.append({**base_tag, "k": k, "source": spec, "method": name, "alpha": alpha, "n_draws": len(covs),
                     "cond_short_frac": float(np.mean(avg < 1 - alpha - 0.05)),
                     "cond_p05": float(np.quantile(avg, 0.05)), "cond_min": float(avg.min())})
    return rows, cond, dist_rows, err_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--exp-id", default="e16")
    ap.add_argument("--pattern", choices=PATTERNS, default="uniform", help="anchor layout (not used by lidar)")
    ap.add_argument("--sources", nargs="+", default=list(SOURCES),
                    help="anchor value sources; lidar is skipped on tables without the field")
    ap.add_argument("--ks", type=int, nargs="+", default=[20, 200])
    ap.add_argument("--draws", type=int, default=2)
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.1])
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--dist-head-images", type=int, default=200)
    ap.add_argument("--no-lscp", dest="lscp", action="store_false")
    ap.add_argument("--max-images", type=int, default=0, help="debug: use only the first N images")
    args = ap.parse_args()

    table = load_table(args.dataset, args.model)
    table.pix.pop("label", None)
    if args.max_images:
        table = table.subset(np.arange(min(args.max_images, table.n_images)))
    for s in args.sources:
        parse(s)
    sources = [s for s in args.sources if parse(s)[0] != "lidar" or table.has("lidar")]
    if len(sources) < len(args.sources):
        print("no `lidar` field in this table: skipping the lidar source")
    if "lidar" in sources:
        # every source on the same images: those with at least max(ks) returns
        n_ret = np.bincount(table.image_of_pixel()[table.pix["lidar"] > 0], minlength=table.n_images)
        keep = n_ret >= max(args.ks)
        print(f"lidar: keeping {keep.sum()}/{table.n_images} images with >= {max(args.ks)} returns")
        table = table.subset(np.flatnonzero(keep))
    out = results_dir(args.exp_id, f"{args.dataset}__{args.model}__{args.pattern}")
    write_spec(out, {**vars(args), "sources_run": sources, "n_images": table.n_images, "n_pixels": table.n_pixels,
                     "table_meta": table.meta})
    print(f"{args.dataset}/{args.model}: {table.n_images} images; pattern {args.pattern}, sources {sources}")

    rows, cond, dists, errs = [], [], [], []
    for seed in range(args.n_seeds):
        t0 = time.perf_counter()
        r, c, dr, e = run_seed(table, seed, args, sources, print)
        rows, cond, dists, errs = rows + r, cond + c, dists + dr, errs + e
        pd.DataFrame(rows).to_csv(out / "runs.csv", index=False)
        pd.DataFrame(cond).to_csv(out / "cond.csv", index=False)
        pd.DataFrame(dists).to_csv(out / "dist.csv", index=False)
        pd.DataFrame(errs).to_csv(out / "anchor_err.csv", index=False)
        print(f"seed {seed} finished in {time.perf_counter() - t0:.0f}s")

    keys = ["k", "source", "method", "alpha"]
    dist_summ = (pd.DataFrame(dists).groupby([*keys, "dist_bin"], sort=False)
                 [["cov_image_mean", "n_img", "frac_px"]].mean().reset_index())
    dist_summ.to_csv(out / "dist_summary.csv", index=False)
    runs = pd.DataFrame(rows)
    runs["base"] = runs["source"]  # summarize() groups by k, base, method, alpha
    cnd = pd.DataFrame(cond).assign(base=lambda x: x["source"])
    summ = summarize(runs, cnd).rename(columns={"base": "source"})
    summ.to_csv(out / "summary.csv", index=False)
    (out / "summary.md").write_text(to_markdown(summ) + "\n")
    err_summ = pd.DataFrame(errs).groupby(["k", "source"], sort=False).mean(numeric_only=True).reset_index()
    print(err_summ[["k", "source", "n_anchor_mean", "err_median_abs", "frac_err_gt_10pct"]].round(4).to_string(index=False))
    show = ["k", "source", "method", "alpha", "cov_image_mean", "median_rel_width", "cond_short_frac"]
    print(summ[show].round(4).to_string(index=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
