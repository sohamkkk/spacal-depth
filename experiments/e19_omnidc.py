"""E19: a depth-completion centre (OMNI-DC, ICCV 2025) conformalized three ways vs A-dual on the frozen model.

OMNI-DC saw RGB + the structured anchors of an exported dual-anchor mask (scripts/export_dual_anchor_masks.py,
scripts/omnidc_forward.py); the probes of the same mask stayed hidden from it. On the same positions:
  centre unidepth        the frozen monocular model (pred_source clean or lidar for its structured anchors)
  centre omnidc-<src>    OMNI-DC completed from <src> anchors (clean = laser GT, lidar = iPad)
each with probe_source in {clean, lidar}, and methods
  split-CP       ordinary split conformal on |log gt - log centre| over the calibration images' evaluation pixels
                 (needs dense GT on calibration images; marginal over pixels)
  A-dist-split   distance-binned anchor calibration across images
  A-dual         per-image probe quantile (T13), and A-dual-both = hull(A-dual, A-dist-split) (T14)
Widths of different centres are compared by rel_interval_score (normalized by GT) and median width / GT.

Kill (review 2 Nov): if conformalized OMNI-DC beats A-dual's valid interval score by >= 15 %, A-dual on the
frozen model is not the method to sell; check whether A-dual on the OMNI-DC centre keeps that gain.

    python -m experiments.e19_omnidc --dataset arkitscenes --model unidepthv2-large \
        --anchors omni_anchors/arkitscenes --omni results/omnidc/arkitscenes --patterns uniform cluster
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.common import load_table, results_dir, split_tables, to_markdown, write_spec
from experiments.e11_anchors import a_dist, eval_view, log_err, prepare_dist, score_row
from experiments.e18_sensor_dual import mix_obs
from src.calibrators import anchor_dist as ad
from src.calibrators.anchor_noise import anchor_values
from src.calibrators.anchors import AnchorDraw, rank_by_key, with_values
from src.calibrators.dual import dual_interval, probe_quantile, split_roles
from src.data.prepared import PixelTable
from src.data.protocols import PROTOCOLS, clamp_pred

SOURCES = ("clean", "lidar")
TABS = ("cal", "test", "head_fit")


def mask_draw(structured: np.ndarray, probe: np.ndarray, offsets: np.ndarray, k: int, seed: int) -> AnchorDraw:
    """Positions of an exported mask as a budget-split draw (probes halved per image for the A-local roles)."""
    img = np.repeat(np.arange(len(offsets) - 1), np.diff(offsets))
    n_probe = np.bincount(img[probe], minlength=len(offsets) - 1)
    half_rank = rank_by_key(offsets, np.where(probe, np.random.default_rng([seed, 19]).random(len(img)), np.inf))
    scale_half = probe & (half_rank < (n_probe // 2)[img])
    anchor = structured | probe
    k_img = np.bincount(img[anchor], minlength=len(offsets) - 1)
    z = np.zeros(len(offsets) - 1)
    return AnchorDraw(k, anchor, scale_half, probe & ~scale_half, z, z, z, k_img)


def subset_draw(draw: AnchorDraw, t_full: PixelTable, idx: np.ndarray) -> AnchorDraw:
    keep = np.zeros(t_full.n_images, bool)
    keep[idx] = True
    px = np.repeat(keep, t_full.counts)
    z = np.zeros(len(idx))
    return AnchorDraw(draw.k, draw.anchor[px], draw.scale_half[px], draw.cal_half[px], z, z, z, draw.k_img[idx])


def split_cp(d_cal: np.ndarray, cal_eval: np.ndarray, test_pred: np.ndarray, alpha: float):
    r = np.sort(np.abs(d_cal[cal_eval]))
    j = int(np.ceil((len(r) + 1) * (1 - alpha))) - 1
    q = r[j] if j < len(r) else np.inf
    return test_pred * np.exp(-q), test_pred * np.exp(q)


def run_cell(tabs, d_gt, obs, sdraws, pred_src, probe_src, s, half_a, alpha, args):
    test = tabs["test"]
    vt, st = eval_view(test, sdraws["test"].eval)
    d_pred, d_probe = obs[pred_src], obs[probe_src]
    probes = {n: split_roles(sdraws[n], d_gt[n], tabs[n].offsets)[1] for n in TABS}
    pdraws = {n: split_roles(sdraws[n], d_pred[n], tabs[n].offsets)[0] for n in TABS}
    d_mix = {n: mix_obs(d_pred[n], d_probe[n], probes[n]) for n in TABS}
    sdraws_obs = {n: with_values(sdraws[n], d_mix[n], tabs[n].offsets) for n in TABS}
    cfg3, fields3, _ = prepare_dist("auto", tabs["head_fit"], d_pred["head_fit"], half_a, alpha, pdraws, tabs, d_pred)
    cfg2, fields2, _ = prepare_dist("auto", tabs["head_fit"], d_mix["head_fit"], half_a, alpha, sdraws_obs, tabs, d_mix)

    methods, inf = {}, {}
    model3, _ = ad.fit_dist_model(tabs["head_fit"], d_pred["head_fit"], s["head_fit"], pdraws["head_fit"], cfg3,
                                  alpha, half_a, use_g=True, single_bin=False, fields=fields3["head_fit"])
    q3 = probe_quantile(model3, d_probe["test"], s["test"], fields3["test"], probes["test"], test.offsets,
                        alpha, args.dual_gamma)
    lo3, hi3 = dual_interval(model3, test.pix["pred"], s["test"], fields3["test"], q3, test.offsets, args.dual_gamma)
    methods["A-dual"] = (lo3[st], hi3[st])
    inf["A-dual"] = float(np.mean(~np.isfinite(q3)))
    ld, hd = a_dist(cfg2, fields2, s, sdraws_obs, tabs, d_mix, half_a, alpha, args)
    methods["A-dist-split"] = (ld, hd)
    methods["A-dual-both"] = (np.minimum(lo3[st], ld), np.maximum(hi3[st], hd))
    inf["A-dual-both"] = inf["A-dual"]
    methods["split-CP"] = split_cp(d_gt["cal"], sdraws["cal"].eval, test.pix["pred"][st].astype(np.float64), alpha)

    out = []
    y = vt.pix["gt"].astype(np.float64)
    for name, (lo, hi) in methods.items():
        m, cov_img = score_row(vt, lo, hi, alpha, args.beta)
        with np.errstate(invalid="ignore"):
            m["median_width_over_gt"] = float(np.median((hi - lo) / y))
        out.append((name, m, cov_img, inf.get(name, 0.0), cfg3.kind, cfg2.kind))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="arkitscenes")
    ap.add_argument("--model", default="unidepthv2-large", help="frozen model table (also gives gt, lidar, positions)")
    ap.add_argument("--anchors", type=Path, required=True, help="export dir with <pattern>/k{k}_seed{s}_draw{d}.npz")
    ap.add_argument("--omni", type=Path, required=True, help="scripts/omnidc_merge.py --out")
    ap.add_argument("--exp-id", default="e19_omnidc")
    ap.add_argument("--patterns", nargs="+", default=["uniform", "cluster"])
    ap.add_argument("--k", type=int, default=200)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--draws", type=int, nargs="+", default=[0])
    ap.add_argument("--omni-sources", nargs="+", default=list(SOURCES), choices=SOURCES)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--dual-gamma", type=float, default=0.5)
    ap.add_argument("--dist-head-images", type=int, default=200)
    ap.add_argument("--max-images", type=int, default=0, help="first N images only (smoke test)")
    args = ap.parse_args()
    args.dist_no_g, args.dist_single_bin = False, False

    base = load_table(args.dataset, args.model)
    base.pix = {k: base.pix[k] for k in ("gt", "pred", "row", "col", "lidar") if k in base.pix}
    full_ids, full_offsets = base.img["sample_id"], base.offsets
    n_img = min(args.max_images, base.n_images) if args.max_images else base.n_images
    cut = int(full_offsets[n_img])
    base = base.subset(np.arange(n_img))
    protocol = PROTOCOLS[args.dataset]
    out = results_dir(args.exp_id, f"{args.dataset}__{args.model}")
    write_spec(out, {**vars(args), "anchors": str(args.anchors), "omni": str(args.omni)})
    rows, cond = [], []

    for pattern in args.patterns:
        for seed in args.seeds:
            for draw_i in args.draws:
                t0 = time.perf_counter()
                stem = f"k{args.k}_seed{seed}_draw{draw_i}"
                with np.load(args.anchors / pattern / f"{stem}.npz") as z:
                    if not (np.array_equal(z["sample_id"], full_ids) and np.array_equal(z["offsets"], full_offsets)):
                        raise ValueError(f"{pattern}/{stem}: mask images differ from the {args.model} table")
                    structured, probe = z["structured"][:cut], z["probe"][:cut]
                centres = {("unidepth", src): base.pix["pred"] for src in SOURCES}
                for src in args.omni_sources:
                    with np.load(args.omni / f"{pattern}__{stem}__{src}.npz") as z:
                        if not np.array_equal(z["offsets"], full_offsets):
                            raise ValueError(f"OMNI-DC {pattern}/{stem}/{src}: table order differs")
                        centres[("omnidc", src)] = z["pred"][:cut]
                ok_img = np.ones(base.n_images, bool)
                for p in centres.values():
                    ok_img &= np.bincount(base.image_of_pixel(), weights=~np.isfinite(p), minlength=base.n_images) == 0
                full = mask_draw(structured, probe, base.offsets, args.k, seed)
                keep = np.flatnonzero(ok_img)
                sub = base.subset(keep)
                full = subset_draw(full, base, keep)
                px_keep = np.repeat(ok_img, base.counts)
                head, cal, test = split_tables(sub, seed)
                pos = {s: i for i, s in enumerate(sub.img["sample_id"])}
                idx = {n: np.array([pos[s] for s in t.img["sample_id"]]) for n, t in (("head", head), ("cal", cal), ("test", test))}
                n_fit = min(len(idx["head"]), args.dist_head_images)
                idx["head_fit"] = np.sort(np.random.default_rng([seed, 7]).choice(idx["head"], n_fit, replace=False))
                half_a = np.random.default_rng([seed, 8]).random(n_fit) < 0.5
                sdraws = {n: subset_draw(full, sub, idx[n]) for n in TABS}

                for (centre, pred_src), pred in centres.items():
                    t_c = PixelTable(pix={**sub.pix, "pred": clamp_pred(pred[px_keep], protocol)},
                                     img=sub.img, offsets=sub.offsets, meta=sub.meta)
                    tabs = {n: t_c.subset(idx[n]) for n in TABS}
                    d_gt = {n: log_err(t) for n, t in tabs.items()}
                    s = {n: np.ones(t.n_pixels) for n, t in tabs.items()}
                    rng_v = np.random.default_rng([seed, 1900])
                    obs = {spec: {n: anchor_values(tabs[n], d_gt[n], spec, rng_v) for n in TABS} for spec in SOURCES}
                    for probe_src in SOURCES:
                        tag = {"pattern": pattern, "seed": seed, "draw": draw_i, "k": args.k, "alpha": args.alpha,
                               "centre": centre, "pred_source": pred_src, "probe_source": probe_src,
                               "n_img": int(ok_img.sum()), "n_test": tabs["test"].n_images}
                        for name, m, cov_img, inf, k3, k2 in run_cell(tabs, d_gt, obs, sdraws, pred_src, probe_src,
                                                                     s, half_a, args.alpha, args):
                            rows.append({**tag, "method": name, **m, "frac_img_local_inf": inf,
                                         "dual_kind": k3, "dist_split_kind": k2})
                            cond.append({**tag, "method": name,
                                         "cond_short_frac": float(np.mean(cov_img < 1 - args.alpha - 0.05)),
                                         "cond_p05": float(np.quantile(cov_img, 0.05))})
                        pd.DataFrame(rows).to_csv(out / "runs.csv", index=False)
                        pd.DataFrame(cond).to_csv(out / "cond.csv", index=False)
                    print(f"  {pattern} {stem} {centre}/{pred_src} done", flush=True)
                print(f"{pattern} {stem}: {time.perf_counter() - t0:.0f}s ({int((~ok_img).sum())} images dropped)", flush=True)

    runs, cnd = pd.DataFrame(rows), pd.DataFrame(cond)
    keys = ["pattern", "centre", "pred_source", "probe_source", "method"]
    cols = ["cov_image_mean", "cov_image_p10", "median_rel_width", "median_width_over_gt", "rel_interval_score",
            "rel_interval_score_robust", "rel_interval_score_imgmed", "frac_img_short", "frac_infinite",
            "frac_uninformative"]
    summ = runs.groupby(keys)[cols].mean().join(cnd.groupby(keys)[["cond_short_frac"]].mean()).reset_index()
    summ.to_csv(out / "summary.csv", index=False)
    (out / "summary.md").write_text(to_markdown(summ) + "\n")
    print(summ.round(4).to_string(index=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
