"""Qualitative A-local vs A-dual maps on ARKitScenes (teaser, Supp. sensor figure): compute stage.

Reproduces one E18 split and budget-split anchor draw (same seeds and draw order as experiments.e18_sensor_dual)
and, on every test image's evaluation pixels (neither structured anchor nor probe), the intervals of
  A-local   within-image calibration that reuses the k - n structured anchors (one random half fixes the shift,
            the other sets the width): the naive method of the teaser
  A-dual    structured anchors predict, the n random probes calibrate (src/calibrators/dual.py)
for each pred_source/probe_source cell in --cells (clean = laser, lidar = iPad). Coverage is of the laser truth.

Writes results/qual_dual/<dataset>__<model>__<pattern>/:
  per_image.csv   coverage of every test image per cell and method (the teaser histogram)
  candidates.csv  frames for the figures, at most one per visit, all with first-cell A-dual near 1 - alpha:
                  kind=teaser (A-local far below), kind=sensor (last-cell A-dual far below), kind=requested
  maps.npz        stride-8 grids of the candidates: row, col, gt, pred, lidar, lidar_conf, structured, probe,
                  eval masks and lo/hi per cell and method
Render with experiments.qual_dual_fig; scripts/export_arkit_frames.py exports the candidates' RGB on the cluster.

    SPACAL_PREPARED_ROOT=prepared_new python -m experiments.qual_dual --model unidepthv2-large --pattern cluster
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from experiments.common import load_table, results_dir, split_tables, write_spec
from experiments.e11_anchors import log_err, prepare_dist
from experiments.e18_sensor_dual import SOURCES
from src.calibrators import anchor_dist as ad
from src.calibrators.anchor_noise import anchor_values
from src.calibrators.anchors import (
    PATTERNS,
    AnchorDraw,
    _anchor_stats,
    anchor_keys,
    draw_split_anchors,
    local_halfwidth,
    rank_by_key,
)
from src.calibrators.dual import dual_interval, probe_quantile, split_roles

MAP_FIELDS = ("row", "col", "gt", "pred", "lidar", "lidar_conf")


def per_image_mean(x: np.ndarray, img: np.ndarray, n_img: int) -> np.ndarray:
    return np.bincount(img, x.astype(np.float64), minlength=n_img) / np.maximum(np.bincount(img, minlength=n_img), 1)


def local_on_structured(test, d: np.ndarray, structured: np.ndarray, k: int, alpha: float, seed: int):
    """A-local on the structured anchors only: random halves (shift / width) among them."""
    n_img = test.n_images
    img = test.image_of_pixel()
    rng = np.random.default_rng([seed, 1900])
    half_rank = rank_by_key(test.offsets, np.where(structured, rng.random(test.n_pixels), np.inf))
    k_s = np.bincount(img[structured], minlength=n_img)
    shift_half = structured & (half_rank < (k_s // 2)[img])
    draw = AnchorDraw(k, structured, shift_half, structured & ~shift_half,
                      *_anchor_stats(d, img, n_img, structured, shift_half), k_s)
    q = local_halfwidth(d, np.ones(test.n_pixels), test.offsets, draw, alpha)
    pred = test.pix["pred"].astype(np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        return pred * np.exp(draw.shift_half[img] - q[img]), pred * np.exp(draw.shift_half[img] + q[img])


def run(args):
    table = load_table(args.dataset, args.model)
    table.pix.pop("label", None)
    seed, k, alpha = args.seed, args.k, args.alpha
    head, cal, test = split_tables(table, seed)
    tabs = {"head": head, "cal": cal, "test": test}
    n_fit = min(head.n_images, args.dist_head_images)
    fit_img = np.zeros(head.n_images, bool)
    fit_img[np.random.default_rng([seed, 7]).choice(head.n_images, n_fit, replace=False)] = True
    tabs["head_fit"] = head.subset(np.flatnonzero(fit_img))
    d_gt = {n: log_err(t) for n, t in tabs.items()}
    d_gt["head_fit"] = d_gt["head"][np.repeat(fit_img, head.counts)]
    half_a = np.random.default_rng([seed, 8]).random(tabs["head_fit"].n_images) < 0.5
    s = {n: np.ones(t.n_pixels) for n, t in tabs.items()}
    obs = {}
    for spec in SOURCES:
        rng_v = np.random.default_rng([seed, 1800 + (0 if spec == "clean" else 1)])
        obs[spec] = {n: anchor_values(tabs[n], d_gt[n], spec, rng_v) for n in tabs}

    rng_d = np.random.default_rng([seed, int(k) + 1, args.draw, 99])

    def draw_one(name):
        t = tabs[name]
        key = None if args.pattern == "uniform" else anchor_keys(args.pattern, t.pix, t.offsets, rng_d)
        return draw_split_anchors(d_gt[name], t.offsets, k, args.n_probes, rng_d, key)

    sdraws = {n: draw_one(n) for n in ("cal", "test", "head_fit")}
    probes = {n: split_roles(sdraws[n], d_gt[n], tabs[n].offsets)[1] for n in sdraws}
    ev = sdraws["test"].eval
    structured = sdraws["test"].anchor & ~probes["test"]
    img = test.image_of_pixel()
    gt = test.pix["gt"].astype(np.float64)

    intervals, per_img = {}, {"sample_id": test.img["sample_id"], "scene_id": test.img["scene_id"],
                              "height": test.img["height"], "width": test.img["width"],
                              "n_eval": np.bincount(img[ev], minlength=test.n_images)}
    for pred_src, probe_src in args.cells:
        t0 = time.perf_counter()
        cell = f"{pred_src}/{probe_src}"
        d_pred, d_probe = obs[pred_src], obs[probe_src]
        pdraws = {n: split_roles(sdraws[n], d_pred[n], tabs[n].offsets)[0] for n in sdraws}
        cfg, fields, _ = prepare_dist("auto", tabs["head_fit"], d_pred["head_fit"], half_a, alpha, pdraws, tabs, d_pred)
        model, _ = ad.fit_dist_model(tabs["head_fit"], d_pred["head_fit"], s["head_fit"], pdraws["head_fit"], cfg,
                                     alpha, half_a, use_g=True, single_bin=False, fields=fields["head_fit"])
        q = probe_quantile(model, d_probe["test"], s["test"], fields["test"], probes["test"], test.offsets, alpha,
                           args.dual_gamma)
        intervals[(cell, "A-dual")] = dual_interval(model, test.pix["pred"], s["test"], fields["test"], q,
                                                    test.offsets, args.dual_gamma)
        intervals[(cell, "A-local")] = local_on_structured(test, d_pred["test"], structured, k, alpha, seed)
        for meth in ("A-local", "A-dual"):
            lo, hi = intervals[(cell, meth)]
            covered = (gt >= lo) & (gt <= hi)
            per_img[f"cov|{cell}|{meth}"] = per_image_mean(covered[ev], img[ev], test.n_images)
        print(f"{cell}: A-local {per_img[f'cov|{cell}|A-local'].mean():.3f}, "
              f"A-dual {per_img[f'cov|{cell}|A-dual'].mean():.3f} ({time.perf_counter() - t0:.0f}s)", flush=True)
    per_img = pd.DataFrame(per_img)

    main = f"{args.cells[0][0]}/{args.cells[0][1]}"
    loc, dua = per_img[f"cov|{main}|A-local"], per_img[f"cov|{main}|A-dual"]
    near = dua.between(1 - alpha - args.dual_window, 1 - alpha + args.dual_window)
    full = per_img.n_eval >= args.min_eval_frac * per_img.n_eval.max()
    teaser = (per_img[near & full & (loc <= args.max_local_cov)].assign(gap=dua - loc, kind="teaser")
              .sort_values("gap", ascending=False).drop_duplicates("scene_id").head(args.n_candidates))
    last = f"cov|{args.cells[-1][0]}/{args.cells[-1][1]}|A-dual"
    sensor = (per_img[near & full].assign(gap=dua - per_img[last], kind="sensor")
              .sort_values("gap", ascending=False).drop_duplicates("scene_id").head(args.n_candidates))
    extra = per_img[per_img.sample_id.isin(args.save_ids)].assign(gap=np.nan, kind="requested")
    cand = pd.concat([teaser, sensor, extra])

    maps = {}
    for i in cand.index.unique():
        sl = slice(test.offsets[i], test.offsets[i + 1])
        sid = test.img["sample_id"][i]
        for f in MAP_FIELDS:
            maps[f"{sid}/{f}"] = test.pix[f][sl]
        maps[f"{sid}/structured"], maps[f"{sid}/probe"], maps[f"{sid}/eval"] = structured[sl], probes["test"][sl], ev[sl]
        for (cell, meth), (lo, hi) in intervals.items():
            maps[f"{sid}/{cell}/{meth}/lo"], maps[f"{sid}/{cell}/{meth}/hi"] = lo[sl].astype(np.float32), hi[sl].astype(np.float32)
    return per_img, cand, maps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="arkitscenes")
    ap.add_argument("--model", default="unidepthv2-large")
    ap.add_argument("--pattern", choices=PATTERNS, default="cluster")
    ap.add_argument("--k", type=int, default=200)
    ap.add_argument("--n-probes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--draw", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--dual-gamma", type=float, default=0.5)
    ap.add_argument("--dist-head-images", type=int, default=200)
    ap.add_argument("--cells", nargs="+", default=["clean/clean", "lidar/clean", "lidar/lidar"],
                    help="pred_source/probe_source; the first one selects the candidate frames")
    ap.add_argument("--max-local-cov", type=float, default=0.3)
    ap.add_argument("--dual-window", type=float, default=0.035,
                    help="candidates need first-cell A-dual coverage within 1 - alpha +- this")
    ap.add_argument("--save-ids", nargs="*", default=[], help="sample_ids whose maps are saved as well")
    ap.add_argument("--min-eval-frac", type=float, default=0.9)
    ap.add_argument("--n-candidates", type=int, default=16)
    args = ap.parse_args()
    args.cells = [tuple(c.split("/")) for c in args.cells]

    out = results_dir("qual_dual", f"{args.dataset}__{args.model}__{args.pattern}")
    write_spec(out, {**vars(args), "cells": [f"{p}/{q}" for p, q in args.cells]})
    per_img, cand, maps = run(args)
    per_img.to_csv(out / "per_image.csv", index=False)
    cand.to_csv(out / "candidates.csv", index=False)
    np.savez_compressed(out / "maps.npz", **maps)
    print(cand[["kind", "sample_id", *[c for c in cand if c.startswith("cov|")], "gap"]].round(3)
          .to_string(index=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
