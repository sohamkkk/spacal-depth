"""Downstream safety query (E12): "how close is the nearest surface in a
forward region of interest (ROI)?" — the primitive behind obstacle-distance
checks and collision tests.

A method returns a per-pixel lower bound L_j <= y_j (metres). The region
bound is min_{j in region} L_j; it is *violated* when it exceeds the true
nearest depth min_{j in region} y_j, i.e. the system would believe the way is
clear further than it is. If every ROI pixel is covered (y_j >= L_j for all
j) then every sub-region bound holds as well, which is why a simultaneous
(whole-ROI) guarantee answers all sub-region queries at once.

ROIs are fixed fractions of the evaluation crop (x-only, so calibration and
test images use the same region). Regions are evaluated on valid-GT lattice
pixels, as every other metric in the study.
  kitti  rows [0, 0.25) of the Garg crop, middle third of the columns: the
         crop starts near the horizon, so this band sees vehicles and
         structures ahead while the road surface in it is >~25 m away
         (camera 1.65 m above the road, f ~ 720 px) [ASSUMPTION: flat road].
  nyu / diode_indoor  the central 60 % x 50 % of the crop (what a robot
         moving forward would hit).
"""

import numpy as np

from src.data.prepared import PixelTable

ROI = {
    "kitti": (0.0, 0.25, 1 / 3, 2 / 3),
    "nyu": (0.2, 0.8, 0.25, 0.75),
    "diode_indoor": (0.2, 0.8, 0.25, 0.75),
}
THRESHOLDS = {"kitti": (10.0, 15.0, 20.0), "nyu": (1.0, 1.5, 2.0, 2.5), "diode_indoor": (1.0, 1.5, 2.0, 2.5)}


def roi_mask(table: PixelTable, roi: tuple[float, float, float, float]) -> np.ndarray:
    r0, r1, c0, c1 = roi
    iop = table.image_of_pixel()
    fr = table.pix["row"] / table.img["height"][iop]
    fc = table.pix["col"] / table.img["width"][iop]
    return (fr >= r0) & (fr < r1) & (fc >= c0) & (fc < c1)


def lane_ids(table: PixelTable, roi: tuple[float, float, float, float], n_lanes: int) -> np.ndarray:
    """Equal-width column strips of the ROI, 0..n_lanes-1 (pixels outside the ROI get a clipped id)."""
    _, _, c0, c1 = roi
    fc = table.pix["col"] / table.img["width"][table.image_of_pixel()]
    return np.clip(((fc - c0) / (c1 - c0) * n_lanes).astype(np.int64), 0, n_lanes - 1)


def group_min(values: np.ndarray, groups: np.ndarray, n_groups: int) -> np.ndarray:
    """Per-group minimum (+inf for empty groups)."""
    out = np.full(n_groups, np.inf)
    np.minimum.at(out, groups, values)
    return out


def region_minima(table: PixelTable, values: np.ndarray, lanes: np.ndarray, n_lanes: int) -> tuple[np.ndarray, np.ndarray]:
    """(per-image min over the whole table's pixels, per-(image, lane) min [n_images, n_lanes])."""
    iop = table.image_of_pixel()
    whole = group_min(values, iop, table.n_images)
    per_lane = group_min(values, iop * n_lanes + lanes, table.n_images * n_lanes).reshape(table.n_images, n_lanes)
    return whole, per_lane


def lane_event_scores(table: PixelTable, pred: np.ndarray, lanes: np.ndarray, n_lanes: int) -> np.ndarray:
    """Per image: max over its non-empty lanes of log(lane-min pred) - log(lane-min y). Split CP on it
    gives one factor exp(-lambda) under which every lane bound of a new image holds with prob >= 1 - alpha;
    the whole-ROI bound (the minimum of the lane bounds) then holds too."""
    _, pred_lane = region_minima(table, pred, lanes, n_lanes)
    _, true_lane = region_minima(table, table.pix["gt"].astype(np.float64), lanes, n_lanes)
    with np.errstate(invalid="ignore"):
        s = np.log(pred_lane) - np.log(true_lane)
    return np.where(np.isfinite(true_lane), s, -np.inf).max(axis=1)


def scene_mean(flags: np.ndarray, units: np.ndarray) -> float:
    """Mean over scenes of each scene's mean flag: the quantity a CRC-over-scenes guarantee bounds."""
    _, inv = np.unique(units, return_inverse=True)
    return float((np.bincount(inv, weights=flags) / np.bincount(inv)).mean())


def danger_scores(table: PixelTable, s: np.ndarray, D: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each truly dangerous pixel (y < D), the lambda below which L = pred * exp(-lambda * s)
    clears it (L >= D): (log pred - log D) / s. Returns (r, offsets, has_danger) over the images
    with at least one dangerous pixel, so CRC on (r, offsets) bounds the per-image share of
    dangerous pixels the rule "flag if L < D" misses, averaged over images that contain danger."""
    dang = table.pix["gt"] < D
    cnt = np.bincount(table.image_of_pixel()[dang], minlength=table.n_images)
    has = cnt > 0
    r = (np.log(table.pix["pred"][dang].astype(np.float64)) - np.log(D)) / s[dang]
    return r, np.concatenate([[0], np.cumsum(cnt[has])]), has


def safety_metrics(
    table: PixelTable,
    bound_img: np.ndarray,
    bound_lane: np.ndarray,
    lanes: np.ndarray,
    n_lanes: int,
    thresholds: tuple[float, ...],
    lower_pix: np.ndarray | None = None,
    min_lane_pixels: int = 5,
    units: np.ndarray | None = None,
) -> dict:
    """bound_img: per-image region bound; bound_lane: [n_images, n_lanes]; lower_pix: per-pixel L (or None
    for methods that only produce region bounds). Rates are over test images (the exchangeable unit);
    with `units` (scene per test image) the viol_* rates are also given averaged over scenes (*_scene)."""
    y = table.pix["gt"].astype(np.float64)
    iop = table.image_of_pixel()
    true_img, true_lane = region_minima(table, y, lanes, n_lanes)
    lane_ok = np.bincount(iop * n_lanes + lanes, minlength=table.n_images * n_lanes).reshape(-1, n_lanes) >= min_lane_pixels
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = bound_img / true_img
    flags = {"viol_roi": bound_img > true_img, "viol_lane": ((bound_lane > true_lane) & lane_ok).any(axis=1)}
    if lower_pix is not None:
        flags["viol_any_px"] = np.bincount(iop, weights=y < lower_pix, minlength=table.n_images) > 0
    m = {k: float(f.mean()) for k, f in flags.items()}
    if units is not None:
        m.update({f"{k}_scene": scene_mean(f, units) for k, f in flags.items()})
    m.update({"ratio_median": float(np.median(ratio)), "ratio_p10": float(np.quantile(ratio, 0.1)),
              "n_images": int(table.n_images)})
    if lower_pix is not None:
        m["px_viol"] = float((np.bincount(iop, weights=y < lower_pix, minlength=table.n_images) / table.counts).mean())
    for D in thresholds:
        danger = true_img < D
        clear = bound_img >= D
        m[f"base_rate@{D:g}"] = float(danger.mean())
        m[f"miss@{D:g}"] = float((danger & clear).sum() / danger.sum()) if danger.any() else float("nan")
        m[f"false_alarm@{D:g}"] = float((~danger & ~clear).sum() / (~danger).sum()) if (~danger).any() else float("nan")
        if lower_pix is not None:
            # per-pixel rule "flag if L < D": the joint rate P(y < D, L >= D) <= P(y < L), the lower-tail
            # miscoverage; the conditional rate (misses / dangerous pixels) has no such bound
            dang, flag = y < D, lower_pix < D
            m[f"px_miss@{D:g}"] = float((np.bincount(iop, weights=dang & ~flag, minlength=table.n_images) / table.counts).mean())
            m[f"px_cond_miss@{D:g}"] = float((dang & ~flag).sum() / dang.sum()) if dang.any() else float("nan")
            # the quantity danger-CRC controls: per-image miss share, averaged over images with danger
            n_dang = np.bincount(iop, weights=dang, minlength=table.n_images)
            n_miss = np.bincount(iop, weights=dang & ~flag, minlength=table.n_images)
            has = n_dang > 0
            m[f"px_cond_miss_img@{D:g}"] = float((n_miss[has] / n_dang[has]).mean()) if has.any() else float("nan")
            m[f"px_false_alarm@{D:g}"] = float((~dang & flag).sum() / (~dang).sum()) if (~dang).any() else float("nan")
    return m
