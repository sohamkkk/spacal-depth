"""E12: does the interval change a safety decision? (src/metrics/safety.py)

Query: the nearest depth inside a forward region of interest (ROI), and per
lane (column strips of the ROI). Every method gives a lower bound
L = pred * exp(-lambda * s) in log space (s = base scale map); the region
bound is the minimum of L over the region.

  point            lambda = 0: the raw prediction
  K1-2s            lower end of the main study's K1 interval (two-sided, whole image)
  V4-2s            lower end of the main study's V4 interval (beta)
  K1-lower         one-sided CRC on ROI pixels: E[fraction of ROI pixels below L] <= alpha
  ROI-event        split CP on max_ROI (log yhat - log y) / s: P(any ROI pixel below L) <= alpha
                   (images exchangeable) -> every region and lane bound holds simultaneously
  ROI-event-scene  the same event with CRC over scenes (valid for an image of a new scene)
  task-scalar      split CP on log min_ROI yhat - log min_ROI y: valid for the whole-ROI
                   query only; applied per lane it has no guarantee
  lane-event       split CP on max over lanes of (log min_lane yhat - log min_lane y): every lane
                   bound (and so the whole-ROI bound) holds simultaneously with prob >= 1 - alpha
  *-scene          ROI-event / task-scalar / lane-event with CRC over scenes; their guarantee is
                   on the scene-averaged rate (columns viol_*_scene)
  A-both           E11's anchored interval (S2-log, k uniform anchors, hull of A-hybrid and
                   A-local) lower end; anchor pixels take their measured depth. Its average and
                   per-image guarantees do not cover the region-minimum event either.
  danger-CRC@D     CRC on the per-image share of truly dangerous ROI pixels (y < D) that the
                   rule "flag if L < D" misses, calibrated on the images that contain danger:
                   E[that share | the image contains danger] <= alpha (column px_cond_miss_img@D).
                   Selecting calibration images by a label event keeps them i.i.d. from the
                   conditional distribution; one lambda per D, applied to every test image.
  danger-CRC-scene@D  the same with CRC over scenes

    python -m experiments.e12_safety --dataset kitti --model dav2-outdoor-large
    python -m experiments.e12_safety --dataset nyu --model unidepthv2-large --n-seeds 3

Outputs (results/e12/<dataset>__<model>/): runs.csv, summary.{csv,md}, spec.json
"""

import argparse
import time

import numpy as np
import pandas as pd

from experiments.common import load_table, results_dir, split_tables, to_markdown, write_spec
from experiments.e11_anchors import crc_units, eval_view, pick_tau
from src.calibrators.anchors import draw_anchors, local_halfwidth
from src.calibrators.crc import InfeasibleRiskError, crc_lambda_exact
from src.calibrators.registry import scene_event_lambda
from src.calibrators.split_cp import conformal_quantile, image_event_lambda
from src.data.splits import exchange_units
from src.metrics.safety import (ROI, THRESHOLDS, danger_scores, group_min, lane_event_scores, lane_ids,
                                region_minima, roi_mask, safety_metrics)
from src.scores.scale_maps import build_scale_model

BASES = ("S2-log", "S5-learned")
PIXEL_METHODS = ("point", "K1-2s", "V4-2s", "K1-lower", "ROI-event", "ROI-event-scene")
REGION_METHODS = ("task-scalar", "task-scalar-scene", "lane-event", "lane-event-scene")


def _lam(fn) -> float:
    try:
        return max(float(fn()), 0.0)
    except InfeasibleRiskError:
        return float("inf")


def log_err(t) -> np.ndarray:
    return np.log(t.pix["gt"].astype(np.float64)) - np.log(t.pix["pred"].astype(np.float64))


def scale_maps(head, cal, test, beta, seed) -> dict[str, dict[str, np.ndarray]]:
    out = {"S2-log": {"cal": np.ones(cal.n_pixels), "test": np.ones(test.n_pixels)}}
    model = build_scale_model("learned", beta=beta, seed=seed).fit(head)
    # bounded away from 0 (as in E11): a near-zero s makes one pixel's normalised score dominate a max
    lo, hi = np.quantile(model.predict(head), [0.01, 0.99])
    out["S5-learned"] = {n: np.clip(model.predict(t), lo, hi) for n, t in (("cal", cal), ("test", test))}
    return out


class AnchoredLower:
    """A-both (S2-log) lower bounds on every test pixel, one anchor draw per seed (E11 recipe)."""

    def __init__(self, head, cal, test, d, k, seed):
        self.head, self.cal, self.test, self.d = head, cal, test, d
        self.rng = np.random.default_rng([seed, 104729, k])
        self.dc = draw_anchors(d["cal"], cal.offsets, k, self.rng)
        self.dt = draw_anchors(d["test"], test.offsets, k, self.rng)
        self.dh = draw_anchors(d["head"], head.offsets, k, self.rng)
        self.vc, self.sc = eval_view(cal, self.dc.eval)
        self.units = exchange_units(self.vc)

    def lower(self, alpha: float) -> np.ndarray:
        cal, test, d, dc, dt = self.cal, self.test, self.d, self.dc, self.dt
        img_c, img_t = cal.image_of_pixel(), test.image_of_pixel()
        tau = pick_tau(self.head, d["head"], np.ones(self.head.n_pixels), self.dh, alpha, self.rng)
        r = np.abs(d["cal"] - dc.shift[img_c]) / (dc.sigma[img_c] + tau)
        lam = crc_units(r[self.sc], self.vc.offsets, self.units, alpha)
        q = local_halfwidth(d["test"], np.ones(test.n_pixels), test.offsets, dt, alpha)
        pred = test.pix["pred"].astype(np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            lo_hybrid = pred * np.exp(dt.shift[img_t] - lam * (dt.sigma[img_t] + tau))
            lo_local = pred * np.exp(dt.shift_half[img_t] - q[img_t])
        lo = np.nan_to_num(np.minimum(lo_hybrid, lo_local), nan=0.0)
        lo[dt.anchor] = test.pix["gt"][dt.anchor]
        return lo


def run_seed(table, seed, args, log):
    head, cal, test = split_tables(table, seed)
    roi = ROI[args.dataset]
    s_all = scale_maps(head, cal, test, args.beta, seed)
    d = {"cal": log_err(cal), "test": log_err(test), "head": log_err(head)}
    anchored = AnchoredLower(head, cal, test, d, args.anchor_k, seed) if args.anchor_k else None
    views, sels = {}, {}
    for name, t in (("cal", cal), ("test", test)):
        m = roi_mask(t, roi)
        views[name], keep = t.select_pixels(m, args.min_roi_px)
        sels[name] = m & np.repeat(keep, t.counts)
    vc, vt = views["cal"], views["test"]
    lanes_c, lanes_t = lane_ids(vc, roi, args.n_lanes), lane_ids(vt, roi, args.n_lanes)
    iop_c = vc.image_of_pixel()
    units_c, units_t = exchange_units(vc), exchange_units(vt)
    pred_t = vt.pix["pred"].astype(np.float64)
    base_tag = {"seed": seed, "n_cal": vc.n_images, "n_test": vt.n_images,
                "frac_cal_kept": vc.n_images / cal.n_images, "frac_test_kept": vt.n_images / test.n_images}

    # task-scalar and lane-event use the prediction only, so they are the same for every base
    o_hat_c = group_min(vc.pix["pred"].astype(np.float64), iop_c, vc.n_images)
    o_c = group_min(vc.pix["gt"].astype(np.float64), iop_c, vc.n_images)
    task_scores = np.log(o_hat_c) - np.log(o_c)
    lane_scores = lane_event_scores(vc, vc.pix["pred"].astype(np.float64), lanes_c, args.n_lanes)
    o_hat_t, o_hat_lane_t = region_minima(vt, pred_t, lanes_t, args.n_lanes)
    region_lams = {"task-scalar": lambda a: conformal_quantile(task_scores, a),
                   "task-scalar-scene": lambda a: scene_event_lambda(task_scores, units_c, a),
                   "lane-event": lambda a: conformal_quantile(lane_scores, a),
                   "lane-event-scene": lambda a: scene_event_lambda(lane_scores, units_c, a)}

    rows = []
    for base in BASES:
        s_c, s_t = s_all[base]["cal"], s_all[base]["test"]
        r2_c = np.abs(d["cal"]) / s_c                       # two-sided, whole image (main study)
        rp_c = (-d["cal"] / s_c)[sels["cal"]]               # one-sided, ROI: > 0 when y < yhat
        s_roi_t = s_t[sels["test"]]
        roi_max_c = np.full(vc.n_images, -np.inf)
        np.maximum.at(roi_max_c, iop_c, rp_c)
        danger = {D: danger_scores(vc, s_c[sels["cal"]], D) for D in args.thresholds}
        for alpha in args.alphas:
            lams = {
                "point": 0.0,
                "K1-2s": _lam(lambda: crc_lambda_exact(r2_c, cal.offsets, alpha)),
                "V4-2s": _lam(lambda: image_event_lambda(r2_c, cal.offsets, alpha, args.beta)),
                "K1-lower": _lam(lambda: crc_lambda_exact(rp_c, vc.offsets, alpha)),
                "ROI-event": _lam(lambda: conformal_quantile(roi_max_c, alpha)),
                "ROI-event-scene": _lam(lambda: scene_event_lambda(roi_max_c, units_c, alpha)),
            }
            for D, (r, off, has) in danger.items():
                lams[f"danger-CRC@{D:g}"] = _lam(lambda: crc_lambda_exact(r, off, alpha))
                lams[f"danger-CRC-scene@{D:g}"] = _lam(lambda: crc_units(r, off, units_c[has], alpha))
            tag = {**base_tag, "base": base, "alpha": alpha}
            for method in PIXEL_METHODS + tuple(m for m in lams if m.startswith("danger-")):
                if method == "point" and base != BASES[0]:
                    continue
                lam = lams[method]
                with np.errstate(over="ignore", invalid="ignore"):
                    lower = pred_t * np.exp(-lam * s_roi_t) if np.isfinite(lam) else np.zeros_like(pred_t)
                b_img, b_lane = region_minima(vt, lower, lanes_t, args.n_lanes)
                m = safety_metrics(vt, b_img, b_lane, lanes_t, args.n_lanes, args.thresholds, lower, units=units_t)
                rows.append({**tag, "method": method, "lambda": lam, **m, **({"base": "-"} if method == "point" else {})})
            if base == BASES[0] and anchored is not None:
                lower = anchored.lower(alpha)[sels["test"]]
                b_img, b_lane = region_minima(vt, lower, lanes_t, args.n_lanes)
                m = safety_metrics(vt, b_img, b_lane, lanes_t, args.n_lanes, args.thresholds, lower, units=units_t)
                rows.append({**tag, "method": f"A-both-k{args.anchor_k}", "lambda": np.nan, **m})
            if base == BASES[0]:
                for method, fn in region_lams.items():
                    lam = _lam(lambda: fn(alpha))
                    f = np.exp(-lam) if np.isfinite(lam) else 0.0
                    m = safety_metrics(vt, o_hat_t * f, o_hat_lane_t * f, lanes_t, args.n_lanes, args.thresholds,
                                       units=units_t)
                    rows.append({**tag, "base": "-", "method": method, "lambda": lam, **m})
    log(f"  seed {seed}: {vc.n_images} cal / {vt.n_images} test images with a usable ROI")
    return rows


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    keys = ["base", "method", "alpha"]
    cols = [c for c in runs.columns if c not in keys + ["seed"] and pd.api.types.is_numeric_dtype(runs[c])]
    g = runs.groupby(keys)
    summ = g[cols].mean()
    for c in ("viol_roi", "viol_lane", "viol_any_px", "viol_roi_scene", "viol_lane_scene", "viol_any_px_scene"):
        if c in runs:
            summ[f"{c}_se"] = g[c].std() / np.sqrt(g.size())
    order = {m: i for i, m in enumerate(PIXEL_METHODS + REGION_METHODS)}
    return summ.reset_index().sort_values(["alpha", "base", "method"], key=lambda c: c.map(order) if c.name == "method" else c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(ROI))
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-seeds", type=int, default=20)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.05, 0.1])
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--n-lanes", type=int, default=3)
    ap.add_argument("--min-roi-px", type=int, default=20, help="images with fewer valid ROI pixels are skipped")
    ap.add_argument("--anchor-k", type=int, default=20, help="anchors per image for A-both (0: skip)")
    ap.add_argument("--thresholds", type=float, nargs="*", default=None, help="decision distances D (m)")
    ap.add_argument("--max-images", type=int, default=0, help="debug: use only the first N images")
    args = ap.parse_args()
    args.thresholds = tuple(args.thresholds or THRESHOLDS[args.dataset])

    table = load_table(args.dataset, args.model)
    table.pix.pop("label", None)
    if args.max_images:
        table = table.subset(np.arange(min(args.max_images, table.n_images)))
    out = results_dir("e12", f"{args.dataset}__{args.model}")
    write_spec(out, {**vars(args), "roi": ROI[args.dataset], "n_images": table.n_images})
    print(f"{args.dataset}/{args.model}: {table.n_images} images; ROI={ROI[args.dataset]} D={args.thresholds}")

    rows = []
    for seed in range(args.n_seeds):
        t0 = time.perf_counter()
        rows += run_seed(table, seed, args, print)
        runs = pd.DataFrame(rows)
        runs.to_csv(out / "runs.csv", index=False)
        print(f"seed {seed} finished in {time.perf_counter() - t0:.0f}s")

    summ = summarize(runs)
    summ.to_csv(out / "summary.csv", index=False)
    (out / "summary.md").write_text(to_markdown(summ))
    show = ["base", "method", "alpha", "viol_roi", "viol_roi_se", "viol_lane", "viol_any_px", "viol_roi_scene",
            "viol_lane_scene", "viol_any_px_scene", "px_viol", "ratio_median"] + [c for c in summ.columns if c.startswith(("miss@", "false_alarm@", "px_cond_miss_img@"))]
    print(summ[[c for c in show if c in summ]].round(4).to_string(index=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
