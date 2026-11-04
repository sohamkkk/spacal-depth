"""E11: sparse-anchor conformal calibration (src/calibrators/anchors.py).

For each seed: scene-level split; fit s(x) on head-train only; then for each
anchor budget k and anchor draw, calibrate A-rescale / A-hybrid on the
calibration images (K1-scene, budget counts units) and A-local (s = 1) inside
each test image, and evaluate on the test images' non-anchor pixels. A-local
rows appear under base S2-log only; A-both = A-hybrid(base) hull A-local.

Also written (shape-vs-scale diagnostic): per test image the oracle global
scale error e_i = median_j (log y - log yhat) and the k = 0 per-image
coverage, plus an "oracle" row that removes e_i exactly (the k -> inf limit).

    python -m experiments.e11_anchors --dataset kitti --model dav2-outdoor-large
    python -m experiments.e11_anchors --dataset kitti --model unidepthv2-large --ks 0 20 --n-seeds 2 --draws 3

E13 (structured anchors) reuses this script with --pattern (see anchor_keys):
    python -m experiments.e11_anchors --exp-id e13 --pattern rows --dataset kitti --model dav2-outdoor-large \
        --ks 20 50 200 --no-oracle

E14 (A-dist, src/calibrators/anchor_dist.py) adds, with --dist-kinds:
  A-dist-<kind>   local shift + distance-aware scale, one K1-scene lambda per distance-to-anchor bin;
                  bandwidths chosen per draw on a head-train subset (S2-log, first alpha), tau / g / bin
                  merges per (base, alpha) on the same subset; bins with too few calibration scenes
                  share their nearer neighbour's lambda
  A-dist-auto     the kind with the best head score in this draw (with >= 2 kinds; column dist_auto_kind)
  A-local-split   A-local on k_random uniform anchors of a budget-split draw (k - k_random by the pattern)
  A-dist-split    A-dist-<--dist-both-kind> on that draw (default auto; column dist_split_kind)
  A-dist-both     hull of A-dist-split and A-local-split (k > k_random)
  A-hybrid-split, A-both-split   A-hybrid and its hull with A-local-split on the same draw (references)
    python -m experiments.e11_anchors --exp-id e14 --pattern cluster --dataset nyu --model depthpro \
        --ks 5 20 50 200 --no-oracle --anchor-dist --dist-kinds knn kernel bilateral

E17 adds, with --dual (src/calibrators/dual.py), on the same budget-split draw:
  A-dual          A-dist-<--dist-both-kind> fields from the structured anchors only, one per-image threshold
                  from the k_random probes (columns dual_kind, dual_gamma); per-image guarantee, no
                  calibration images; base S2-log only
  A-dual-both     hull of A-dual and A-dist-split: A-dual's per-image guarantee and A-dist-split's coverage
                  by distance to the nearest anchor
    python -m experiments.e11_anchors --exp-id e17 --pattern cluster --dataset nyu --model depthpro \
        --ks 50 200 --no-oracle --anchor-dist --dist-kinds median --dual --dist-k-random 20

Outputs (results/<exp-id>/<dataset>__<model>[__<pattern>]/): runs.csv, cond.csv, images.csv, summary.{csv,md}, spec.json
"""

import argparse
import time

import numpy as np
import pandas as pd

from experiments.common import load_table, results_dir, split_tables, to_markdown, write_spec
from src.calibrators import anchor_dist as ad
from src.calibrators.anchors import (
    PATTERNS,
    AnchorDraw,
    anchor_keys,
    draw_anchors,
    draw_split_anchors,
    local_halfwidth,
    nearest_anchor_distance,
)
from src.calibrators.crc import InfeasibleRiskError, crc_lambda_exact, crc_lambda_exact_units
from src.calibrators.dual import dual_interval, probe_quantile, split_roles
from src.data.prepared import PixelTable
from src.data.splits import exchange_units
from src.metrics.interval_metrics import _per_image_mean, evaluate_intervals
from src.scores.learned import LearnedQuantileScale

TAU_GRID = (0.25, 0.5, 1.0, 2.0, 4.0)
DIST_EDGES = (0.02, 0.05, 0.1, 0.2)  # distance to the nearest anchor, as a fraction of the image diagonal
DIST_LABELS = ("<0.02", "0.02-0.05", "0.05-0.1", "0.1-0.2", ">=0.2")


def log_err(t: PixelTable) -> np.ndarray:
    return np.log(t.pix["gt"].astype(np.float64)) - np.log(t.pix["pred"].astype(np.float64))


def image_of(t: PixelTable) -> np.ndarray:
    return t.image_of_pixel()


def oracle_shift(t: PixelTable, d: np.ndarray) -> np.ndarray:
    return np.array([np.median(d[t.offsets[i]:t.offsets[i + 1]]) for i in range(t.n_images)])


def shape_target_table(head: PixelTable) -> PixelTable:
    """Same features (raw prediction), GT shifted so log residual = shape residual d - e_i."""
    e = oracle_shift(head, log_err(head))
    pix = dict(head.pix)
    pix["gt"] = (head.pix["gt"].astype(np.float64) * np.exp(-e[image_of(head)])).astype(np.float32)
    return PixelTable(pix=pix, img=head.img, offsets=head.offsets, meta=head.meta)


def oracle_draw(t: PixelTable, d: np.ndarray) -> AnchorDraw:
    none = np.zeros(t.n_pixels, dtype=bool)
    e = oracle_shift(t, d)
    return AnchorDraw(-1, none, none, none, e, e, np.full(t.n_images, np.nan), np.zeros(t.n_images, int))


def eval_view(t: PixelTable, mask: np.ndarray) -> tuple[PixelTable, np.ndarray]:
    """Lean table of the masked pixels (what evaluate_intervals / K1-scene need)."""
    counts = np.bincount(image_of(t)[mask], minlength=t.n_images)
    keep_img = counts > 0
    sel = mask & keep_img[image_of(t)]
    view = PixelTable(
        pix={"gt": t.pix["gt"][sel], "pred": t.pix["pred"][sel]},
        img={k: t.img[k][keep_img] for k in ("scene_id", "sample_id") if k in t.img},
        offsets=np.concatenate([[0], np.cumsum(counts[keep_img])]).astype(np.int64),
        meta=t.meta,
    )
    return view, sel


def pick_tau(head, d_h, s_h, draw: AnchorDraw, alpha: float, rng) -> float:
    """Floor tau = c * median(sigma) chosen on head-train halves by log interval score."""
    sig = draw.sigma
    base = float(np.nanmedian(sig))
    perm = rng.permutation(head.n_images)
    half_a = np.zeros(head.n_images, bool)
    half_a[perm[: head.n_images // 2]] = True
    img = image_of(head)
    best, best_c = np.inf, TAU_GRID[0]
    for c in TAU_GRID:
        denom = s_h * (np.nan_to_num(sig, nan=base)[img] + c * base)
        r = np.abs(d_h - draw.shift[img]) / denom
        ev = draw.eval
        va, sa = eval_view(head, ev & half_a[img])
        try:
            lam = crc_lambda_exact(r[sa], va.offsets, alpha)
        except InfeasibleRiskError:
            continue
        sb = ev & ~half_a[img]
        q = lam * denom[sb]
        score = 2 * q + (2 / alpha) * np.maximum(r[sb] * denom[sb] - q, 0)
        if (m := float(score.mean())) < best:
            best, best_c = m, c
    return best_c * base


def crc_units(r, offsets, units, alpha) -> float:
    """K1-scene; inf (trivial intervals) when there are too few calibration units for alpha."""
    try:
        return crc_lambda_exact_units(r, offsets, units, alpha)
    except InfeasibleRiskError:
        return float("inf")


def interval_from(t, center_img, q, sel):
    pred = t.pix["pred"][sel].astype(np.float64)
    c = center_img[image_of(t)[sel]]
    with np.errstate(over="ignore"):
        return pred * np.exp(c - q), pred * np.exp(c + q)


def score_row(view, lo, hi, alpha, beta):
    m, _ = evaluate_intervals(view, lo, hi, alpha, beta)
    y = view.pix["gt"].astype(np.float64)
    lower_miss = _per_image_mean(y < lo, view.offsets)
    cov_img = _per_image_mean((y >= lo) & (y <= hi), view.offsets)
    m.update(lower_miss_mean=float(lower_miss.mean()), lower_miss_img_p90=float(np.quantile(lower_miss, 0.9)))
    return m, cov_img


def prepare_dist(kind, head_fit, d_hf, half_a, alpha, draws, tables, d):
    """Bandwidths for one A-dist kind (or kind="auto"), chosen on the head-train subset (S2-log), their head
    score, and the fields of the head subset, calibration and test tables under their anchor draws."""
    model = ad.select_model(head_fit, d_hf, np.ones(head_fit.n_pixels), draws["head_fit"], kind, alpha, half_a)
    cfg = model.cfg
    return cfg, {n: ad.anchor_fields(tables[n], d[n], draws[n], cfg) for n in ("head_fit", "cal", "test")}, \
        model.head_score


def a_dist(cfg, fields, base_s, draws, tables, d, half_a, alpha, args):
    """A-dist interval on the test evaluation pixels of draws["test"] (tau, g, bins refit for this base / alpha)."""
    hf, cal, test = tables["head_fit"], tables["cal"], tables["test"]
    model, _ = ad.fit_dist_model(hf, d["head_fit"], base_s["head_fit"], draws["head_fit"], cfg, alpha, half_a,
                                 use_g=not args.dist_no_g, single_bin=args.dist_single_bin,
                                 fields=fields["head_fit"])
    lams = ad.calibrate(model, cal, d["cal"], base_s["cal"], draws["cal"], alpha,
                        units=exchange_units(cal), fields=fields["cal"])
    lo, hi = ad.interval(model, test, base_s["test"], lams, fields["test"])
    ev = draws["test"].eval
    return lo[ev], hi[ev]


def run_seed(table, seed, args, log):
    head, cal, test = split_tables(table, seed)
    d = {n: log_err(t) for n, t in (("head", head), ("cal", cal), ("test", test))}
    dist_on = bool(args.dist_kinds) or args.dual
    lean = args.lean
    if dist_on:
        # A-dist model fitting (bandwidths, tau, g, bin merges) on a head-train subset, for speed
        n_fit = min(head.n_images, args.dist_head_images)
        fit_img = np.zeros(head.n_images, bool)
        fit_img[np.random.default_rng([seed, 7]).choice(head.n_images, n_fit, replace=False)] = True
        fit_px = np.repeat(fit_img, head.counts)
        head_fit = head.subset(np.flatnonzero(fit_img))
        d["head_fit"] = d["head"][fit_px]
        half_a = np.random.default_rng([seed, 8]).random(head_fit.n_images) < 0.5
    t0 = time.perf_counter()
    s = {"S2-log": {n: np.ones(t.n_pixels) for n, t in (("head", head), ("cal", cal), ("test", test))}}
    learned = []
    if not lean:
        learned = [("S5-learned", LearnedQuantileScale(quantile=1 - args.beta, seed=seed).fit(head)),
                   ("S5-shape", LearnedQuantileScale(quantile=1 - args.beta, seed=seed).fit(shape_target_table(head)))]
    for name, model in learned:
        s[name] = {n: model.predict(t) for n, t in (("head", head), ("cal", cal), ("test", test))}
        # bounded away from 0: a near-zero s gives an anchor or calibration pixel a huge normalised score
        lo, hi = np.quantile(s[name]["head"], [0.01, 0.99])
        s[name] = {n: np.clip(v, lo, hi) for n, v in s[name].items()}
    if dist_on:
        for v in s.values():
            v["head_fit"] = v["head"][fit_px]
    log(f"  seed {seed}: scale models fit in {time.perf_counter() - t0:.0f}s")
    rows, cond, cov_store, dist_rows = [], [], {}, []
    base_tag = {"seed": seed, "n_cal": cal.n_images, "n_test": test.n_images}
    img_rows = pd.DataFrame({"seed": seed, "sample_id": test.img["sample_id"],
                             "scale_err": oracle_shift(test, d["test"])})

    def draw(name, t, k, rng, k_random=None):
        key = None if args.pattern == "uniform" else anchor_keys(args.pattern, t.pix, t.offsets, rng)
        if k_random is not None:
            return draw_split_anchors(d[name], t.offsets, k, k_random, rng, key)
        return draw_anchors(d[name], t.offsets, k, rng, key)

    def dist_bins(dt, st):
        if not args.anchor_dist:
            return None
        diag = np.hypot(test.img["height"], test.img["width"]).astype(np.float64)
        return np.digitize(nearest_anchor_distance(test.pix["row"], test.pix["col"], test.offsets,
                                                   dt.anchor, diag)[st], DIST_EDGES)

    tables = {"cal": cal, "test": test, **({"head_fit": head_fit} if dist_on else {})}

    budgets = list(args.ks) + (["oracle"] if args.oracle else [])
    for k in budgets:
        n_draws = 1 if k in (0, "oracle") else args.draws
        bases = ["S2-log"] if lean else (["S2-log", "S5-learned"] if k == 0 else ["S2-log", "S5-shape"])
        for draw_i in range(n_draws):
            rng = np.random.default_rng([seed, 0 if k == "oracle" else int(k) + 1, draw_i])
            if k == "oracle":
                dc, dt = oracle_draw(cal, d["cal"]), oracle_draw(test, d["test"])
            else:
                dc, dt = draw("cal", cal, k, rng), draw("test", test, k, rng)
            vc, sc = eval_view(cal, dc.eval)
            vt, st = eval_view(test, dt.eval)
            units_cal = exchange_units(vc)
            img_c, img_t = image_of(cal), image_of(test)
            dh = draw("head", head, k, rng) if isinstance(k, int) and k >= 3 and not lean else None
            dist_bin = dist_bins(dt, st) if isinstance(k, int) and k > 0 else None
            # A-dist: per kind, bandwidths and fields once per draw; budget-split draw for A-dist-both
            dist_prep, split, views, dual = {}, None, {}, None
            if dist_on and isinstance(k, int) and k > 0:
                # own stream, so the other methods' results do not depend on --dist-kinds
                rng_d = np.random.default_rng([seed, int(k) + 1, draw_i, 99])
                draws = {"cal": dc, "test": dt, "head_fit": draw("head_fit", head_fit, k, rng_d)}
                for kind in args.dist_kinds:
                    dist_prep[kind] = prepare_dist(kind, head_fit, d["head_fit"], half_a, args.alphas[0],
                                                   draws, tables, d)
                if k > args.dist_k_random:
                    sdraws = {n: draw(n, tables[n], k, rng_d, args.dist_k_random) for n in ("cal", "test", "head_fit")}
                    vt2, st2 = eval_view(test, sdraws["test"].eval)
                    vc2, sc2 = eval_view(cal, sdraws["cal"].eval)
                    split = (sdraws, st2, sc2, vc2, exchange_units(vc2),
                             prepare_dist(args.dist_both_kind, head_fit, d["head_fit"], half_a,
                                          args.alphas[0], sdraws, tables, d))
                    views = {n: (vt2, dist_bins(sdraws["test"], st2))
                             for n in ("A-local-split", "A-dist-split", "A-dist-both", "A-hybrid-split", "A-both-split")}
                    if args.dual:
                        roles = {n: split_roles(sdraws[n], d[n], tables[n].offsets) for n in ("cal", "test", "head_fit")}
                        pdraws = {n: r[0] for n, r in roles.items()}
                        dual = (pdraws, roles["test"][1],
                                prepare_dist(args.dist_both_kind, head_fit, d["head_fit"], half_a, args.alphas[0],
                                             pdraws, tables, d))
                        views["A-dual"] = views["A-dual-both"] = views["A-dual-rmax"] = views["A-dist-both"]
                    local_split = {}
                    for alpha in args.alphas:
                        q2 = local_halfwidth(d["test"], s["S2-log"]["test"], test.offsets, sdraws["test"], alpha)
                        local_split[alpha] = (interval_from(test, sdraws["test"].shift_half, q2[img_t[st2]], st2),
                                              float(np.mean(~np.isfinite(q2))))
            # A-local always on s = 1: with ~10 calibration anchors its threshold is close to the largest
            # normalised anchor residual, and any s with a wide range turns that into widths of 1e9
            local = {}
            if isinstance(k, int) and k >= 2:
                for alpha in args.alphas:
                    qloc = local_halfwidth(d["test"], s["S2-log"]["test"], test.offsets, dt, alpha)
                    local[alpha] = (interval_from(test, dt.shift_half, qloc[img_t[st]], st),
                                    float(np.mean(~np.isfinite(qloc))))
            for base in bases:
                s_c, s_t = s[base]["cal"], s[base]["test"]
                for alpha in args.alphas:
                    tag = {**base_tag, "k": str(k), "draw": draw_i, "base": base, "alpha": alpha}
                    methods = {}
                    if not lean:
                        r = np.abs(d["cal"] - dc.shift[img_c]) / s_c
                        lam = crc_units(r[sc], vc.offsets, units_cal, alpha)
                        methods["A-rescale"] = interval_from(test, dt.shift, lam * s_t[st], st)
                    if dh is not None:
                        tau = pick_tau(head, d["head"], s[base]["head"], dh, alpha, rng)
                        den_c = s_c * (dc.sigma[img_c] + tau)
                        den_t = s_t * (dt.sigma[img_t] + tau)
                        r = np.abs(d["cal"] - dc.shift[img_c]) / den_c
                        lam_h = crc_units(r[sc], vc.offsets, units_cal, alpha)
                        methods["A-hybrid"] = interval_from(test, dt.shift, lam_h * den_t[st], st)
                    if alpha in local and base == "S2-log":
                        methods["A-local"] = local[alpha][0]
                    if "A-hybrid" in methods and alpha in local:
                        (lh, hh), (ll, hl) = methods["A-hybrid"], local[alpha][0]
                        methods["A-both"] = (np.minimum(lh, ll), np.maximum(hh, hl))
                    for kind, (cfg, fields, _) in dist_prep.items():
                        methods[f"A-dist-{kind}"] = a_dist(cfg, fields, s[base], draws, tables, d, half_a, alpha, args)
                    if len(dist_prep) > 1:
                        auto_kind = min(dist_prep, key=lambda kd: dist_prep[kd][2])
                        methods["A-dist-auto"] = methods[f"A-dist-{auto_kind}"]
                        tag = {**tag, "dist_auto_kind": auto_kind}
                    local_inf = {"A-local": local.get(alpha, (None, 0.0))[1], "A-both": local.get(alpha, (None, 0.0))[1]}
                    if split is not None:
                        sdraws, st2, sc2, vc2, units_c2, (cfg2, fields2, _) = split
                        tag = {**tag, "dist_split_kind": cfg2.kind}
                        ld, hd = a_dist(cfg2, fields2, s[base], sdraws, tables, d, half_a, alpha, args)
                        ll, hl = local_split[alpha][0]
                        if not lean:
                            # A-hybrid on the same split draw: the like-for-like reference for A-dist-split / -both
                            sc_, st_ = sdraws["cal"], sdraws["test"]
                            tau2 = pick_tau(head_fit, d["head_fit"], s[base]["head_fit"], sdraws["head_fit"], alpha,
                                            rng_d)
                            den_c = s_c * (sc_.sigma[img_c] + tau2)
                            r = np.abs(d["cal"] - sc_.shift[img_c]) / den_c
                            lam2 = crc_units(r[sc2], vc2.offsets, units_c2, alpha)
                            lh, hh = interval_from(test, st_.shift, lam2 * (s_t * (st_.sigma[img_t] + tau2))[st2], st2)
                            methods["A-hybrid-split"] = (lh, hh)
                            methods["A-both-split"] = (np.minimum(lh, ll), np.maximum(hh, hl))
                        if base == "S2-log":
                            methods["A-local-split"] = (ll, hl)
                        methods["A-dist-split"] = (ld, hd)
                        methods["A-dist-both"] = (np.minimum(ld, ll), np.maximum(hd, hl))
                        local_inf.update({n: local_split[alpha][1]
                                          for n in ("A-local-split", "A-dist-both", "A-both-split")})
                    # S2-log only, as A-local: with ~20 probes the threshold is near the largest normalised
                    # probe residual, and a learned base with a wide range turns that into enormous widths
                    if dual is not None and base == "S2-log":
                        pdraws, probes, (cfg3, fields3, _) = dual
                        tag = {**tag, "dual_kind": cfg3.kind, "dual_gamma": args.dual_gamma}
                        model3, _ = ad.fit_dist_model(head_fit, d["head_fit"], s[base]["head_fit"], pdraws["head_fit"],
                                                      cfg3, alpha, half_a, use_g=not args.dist_no_g, single_bin=True,
                                                      fields=fields3["head_fit"])
                        q3 = probe_quantile(model3, d["test"], s_t, fields3["test"], probes, test.offsets, alpha,
                                            args.dual_gamma)
                        lo3, hi3 = dual_interval(model3, test.pix["pred"], s_t, fields3["test"], q3, test.offsets,
                                                 args.dual_gamma)
                        methods["A-dual"] = (lo3[st2], hi3[st2])
                        local_inf["A-dual"] = float(np.mean(~np.isfinite(q3)))
                        if args.dual_rmax:
                            # rank r = n (the largest probe score): ceil((n + 1)(1 - a)) = n for a in [1, 2)/(n + 1)
                            a_r = 1.5 / (args.dist_k_random + 1)
                            q4 = probe_quantile(model3, d["test"], s_t, fields3["test"], probes, test.offsets, a_r,
                                                args.dual_gamma)
                            lo4, hi4 = dual_interval(model3, test.pix["pred"], s_t, fields3["test"], q4, test.offsets,
                                                     args.dual_gamma)
                            methods["A-dual-rmax"] = (lo4[st2], hi4[st2])
                            local_inf["A-dual-rmax"] = float(np.mean(~np.isfinite(q4)))
                        if "A-dist-split" in methods:
                            (ld, hd) = methods["A-dist-split"]
                            methods["A-dual-both"] = (np.minimum(lo3[st2], ld), np.maximum(hi3[st2], hd))
                            local_inf["A-dual-both"] = local_inf["A-dual"]
                    for name, (lo, hi) in methods.items():
                        view, dbin = views.get(name, (vt, dist_bin))
                        m, cov_img = score_row(view, lo, hi, alpha, args.beta)
                        rows.append({**tag, "method": name, **m, "frac_img_local_inf": local_inf.get(name, 0.0)})
                        cov_store.setdefault((str(k), base, name, alpha), []).append(cov_img)
                        if dbin is not None:
                            dist_rows += coverage_by_distance(view, lo, hi, dbin, {**tag, "method": name})
                        if k == 0 and alpha == args.alphas[0]:
                            img_rows[f"cov_k0_{base}"] = cov_img
                        if k == "oracle" and alpha == args.alphas[0]:
                            img_rows[f"cov_oracle_{base}"] = cov_img
            log(f"  seed {seed} k={k} draw {draw_i}: done")

    # conditional (per-image) coverage, averaged over anchor draws; only images present in every draw
    for (k, base, name, alpha), covs in cov_store.items():
        if len({len(c) for c in covs}) != 1:
            continue
        avg = np.mean(covs, axis=0)
        cond.append({**base_tag, "k": k, "base": base, "method": name, "alpha": alpha, "n_draws": len(covs),
                     "cond_short_frac": float(np.mean(avg < 1 - alpha - 0.05)),
                     "cond_p05": float(np.quantile(avg, 0.05)), "cond_min": float(avg.min())})
    return rows, cond, img_rows, dist_rows


def coverage_by_distance(view, lo, hi, dist_bin, tag, min_px: int = 10) -> list[dict]:
    """Image-weighted coverage of evaluation pixels by distance to the nearest anchor
    (images with fewer than min_px pixels in a bin are left out of that bin)."""
    y = view.pix["gt"].astype(np.float64)
    nb = len(DIST_LABELS)
    key = view.image_of_pixel() * nb + dist_bin
    cnt = np.bincount(key, minlength=view.n_images * nb).reshape(-1, nb)
    hit = np.bincount(key, weights=(y >= lo) & (y <= hi), minlength=view.n_images * nb).reshape(-1, nb)
    ok = cnt >= min_px
    return [{**tag, "dist_bin": DIST_LABELS[b], "cov_image_mean": float((hit[ok[:, b], b] / cnt[ok[:, b], b]).mean()),
             "n_img": int(ok[:, b].sum()), "frac_px": float(cnt[:, b].sum() / cnt.sum())}
            for b in range(nb) if ok[:, b].any()]


def summarize(runs: pd.DataFrame, cond: pd.DataFrame) -> pd.DataFrame:
    keys = ["k", "base", "method", "alpha"]
    cols = ["cov_image_mean", "cov_scene_mean", "cov_image_sd", "cov_image_p10", "frac_img_cov_below_1ma_m10",
            "v4_rate", "rel_width", "median_rel_width", "rel_interval_score", "rel_interval_score_robust",
            "rel_interval_score_imgmed", "frac_img_short", "frac_infinite", "frac_uninformative", "lower_miss_mean",
            "lower_miss_img_p90"]
    summ = runs.groupby(keys)[cols].mean()
    summ["cov_image_mean_se"] = runs.groupby(keys)["cov_image_mean"].std() / np.sqrt(runs.groupby(keys).size())
    summ = summ.join(cond.groupby(keys)[["cond_short_frac", "cond_p05"]].mean()).reset_index()
    k_num = pd.to_numeric(summ.k, errors="coerce").fillna(np.inf)  # "oracle" sorts last
    return summ.assign(_k=k_num).sort_values(["alpha", "_k", "base", "method"]).drop(columns="_k")


def scale_share(images: pd.DataFrame) -> dict:
    """How much of the k = 0 per-image coverage variation the global scale error explains."""
    out = {}
    for col in [c for c in images if c.startswith("cov_k0_")]:
        y = images[col].to_numpy()
        e = images["scale_err"].to_numpy()
        X = np.column_stack([np.ones_like(e), np.abs(e), e**2])
        resid = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
        out[col] = {"r2_scale": float(1 - resid.var() / y.var()),
                    "spearman_abs_err_vs_miscov": float(pd.Series(np.abs(e)).corr(pd.Series(1 - y), method="spearman")),
                    "sd_k0": float(y.std())}
        oc = col.replace("cov_k0_", "cov_oracle_").replace("S5-learned", "S5-shape")
        if oc in images:
            out[col]["sd_oracle"] = float(images[oc].std())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--exp-id", default="e11")
    ap.add_argument("--pattern", choices=PATTERNS, default="uniform", help="anchor placement (E13: not uniform)")
    ap.add_argument("--ks", type=int, nargs="+", default=[0, 1, 3, 5, 9, 20, 50, 200])
    ap.add_argument("--draws", type=int, default=5, help="anchor draws per (seed, k > 0)")
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.1])
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--no-oracle", dest="oracle", action="store_false")
    ap.add_argument("--anchor-dist", action="store_true",
                    help="also write coverage by distance to the nearest anchor (dist.csv, dist_summary.csv)")
    ap.add_argument("--dist-kinds", nargs="*", choices=ad.KINDS, default=[],
                    help="E14: A-dist local-shift variants to run (none by default)")
    ap.add_argument("--dist-k-random", type=int, default=20,
                    help="uniform anchors reserved for A-local-split / A-dist-both (run when k > this)")
    ap.add_argument("--dist-both-kind", choices=(*ad.KINDS, "auto"), default="auto")
    ap.add_argument("--dual", action="store_true",
                    help="E17: also A-dual (structured anchors predict, the k_random probes calibrate per image)")
    ap.add_argument("--dual-gamma", type=float, default=0.5,
                    help="A-dual scale shrinkage toward the per-image median (1 = the plain A-dist scale)")
    ap.add_argument("--dual-rmax", action="store_true",
                    help="also A-dual-rmax: per-image threshold = the largest probe score (rank r = n)")
    ap.add_argument("--lean", action="store_true",
                    help="S2-log base only; skip A-rescale / A-hybrid / A-both(-split) and the learned scale fits")
    ap.add_argument("--seed-start", type=int, default=None,
                    help="run seeds seed_start .. seed_start + n_seeds - 1 (output dir gets a __seeds<a>_<b> suffix)")
    ap.add_argument("--dist-head-images", type=int, default=200, help="head-train images used to fit A-dist")
    ap.add_argument("--dist-single-bin", action="store_true", help="ablation: one lambda for all distances")
    ap.add_argument("--dist-no-g", action="store_true", help="ablation: no distance inflation g")
    ap.add_argument("--max-images", type=int, default=0, help="debug: use only the first N images")
    ap.add_argument("--unit-frac", type=float, default=1.0,
                    help="keep a random fraction of whole scenes/scans (low-memory machines)")
    args = ap.parse_args()

    table = load_table(args.dataset, args.model)
    table.pix.pop("label", None)
    if args.max_images:
        table = table.subset(np.arange(min(args.max_images, table.n_images)))
    if args.unit_frac < 1.0:
        units = exchange_units(table)
        uniq = np.unique(units)
        keep = np.random.default_rng(12345).choice(uniq, max(3, int(round(args.unit_frac * len(uniq)))), replace=False)
        table = table.subset(np.flatnonzero(np.isin(units, keep)))
    suffix = "" if args.pattern == "uniform" else f"__{args.pattern}"
    seed0 = args.seed_start or 0
    if args.seed_start is not None:
        suffix += f"__seeds{seed0}_{seed0 + args.n_seeds - 1}"
    out = results_dir(args.exp_id, f"{args.dataset}__{args.model}{suffix}")
    write_spec(out, {**vars(args), "n_images": table.n_images, "n_pixels": table.n_pixels})
    print(f"{args.dataset}/{args.model}: {table.n_images} images, {table.n_pixels:,} px; ks={args.ks} pattern={args.pattern}")

    rows, cond, images, dists = [], [], [], []
    for seed in range(seed0, seed0 + args.n_seeds):
        t0 = time.perf_counter()
        r, c, im, dr = run_seed(table, seed, args, print)
        rows += r
        cond += c
        dists += dr
        images.append(im)
        runs_df, cond_df, img_df = pd.DataFrame(rows), pd.DataFrame(cond), pd.concat(images, ignore_index=True)
        runs_df.to_csv(out / "runs.csv", index=False)
        cond_df.to_csv(out / "cond.csv", index=False)
        img_df.to_csv(out / "images.csv", index=False)
        if dists:
            pd.DataFrame(dists).to_csv(out / "dist.csv", index=False)
        print(f"seed {seed} finished in {time.perf_counter() - t0:.0f}s")

    if dists:
        dist_summ = (pd.DataFrame(dists).groupby(["k", "base", "method", "alpha", "dist_bin"], sort=False)
                     [["cov_image_mean", "n_img", "frac_px"]].mean().reset_index())
        dist_summ.to_csv(out / "dist_summary.csv", index=False)
    summ = summarize(runs_df, cond_df)
    summ.to_csv(out / "summary.csv", index=False)
    share = scale_share(img_df)
    (out / "summary.md").write_text(to_markdown(summ) + "\n\nShape-vs-scale (k = 0 per-image coverage):\n\n"
                                    + "\n".join(f"- {k}: {v}" for k, v in share.items()) + "\n")
    show = ["k", "base", "method", "alpha", "cov_image_mean", "cov_image_sd", "frac_img_cov_below_1ma_m10",
            "rel_width", "rel_interval_score", "lower_miss_mean", "cond_short_frac"]
    print(summ[show].round(4).to_string(index=False))
    for k, v in share.items():
        print(k, {a: round(b, 3) for a, b in v.items()})
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
