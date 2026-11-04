"""A-dual: structured anchors predict, random probes calibrate (Gate 1).

On a budget-split draw (draw_split_anchors: k - k_random anchors by the layout, k_random uniform among the
remaining pixels) the two kinds of anchor get different roles:
  prediction anchors  the structured ones; they alone build A-dist's local shift m(x) and scale s(x)
                      (bandwidths, tau, g and cap fitted on head-train with the same structured layout)
  probes              the k_random uniform ones; their values are never used by m or s. Per image,
                      q_i = ceil((n + 1)(1 - alpha))-th smallest |d_j - m(x_j)| / s_g(x_j) over its n probes
Interval: pred * exp(m(x) -/+ q_i s_g(x)) on the pixels that are neither prediction anchors nor probes, with
s_g = med_i (s / med_i) ** gamma (dual_scale; gamma = 0.5, fixed on NYU / Depth Pro, cuts the width tail
of gamma = 1). med_i is over all pixels of the image and s uses no probe value, so s_g is as fixed as s.

Guarantee: given the image and its prediction anchors, m and s are fixed functions of position, and the
probes plus a uniformly random evaluation pixel are exchangeable (the probes are a uniform subset of the
non-prediction pixels). So E[coverage of image i | image, prediction anchors] >= 1 - alpha for every image
and any prediction-anchor layout, without calibration images (the T8 argument with a fixed score); the
expectation is over the probe draw. With n probes, ceil((n + 1)(1 - alpha)) <= n is needed (n >= 9 at
alpha = 0.1), otherwise the interval is infinite.
Compared with A-local-split (same probes, half of them spent on a constant shift) and A-dist-both (hull of
A-dist-split and A-local-split), A-dual spends every probe on calibration and needs no hull.
"""

import numpy as np

from src.calibrators.anchor_dist import DistModel, Fields
from src.calibrators.anchors import AnchorDraw, _anchor_stats, group_conformal_quantile


def split_roles(draw: AnchorDraw, d_obs: np.ndarray, offsets: np.ndarray) -> tuple[AnchorDraw, np.ndarray]:
    """(prediction-anchor draw, probe mask) of a draw_split_anchors draw. The prediction draw's shift and
    spread are recomputed from the structured anchors' values only; its eval mask (~anchor) still contains
    the probes, so evaluation must use the full draw's eval mask."""
    probes = draw.scale_half | draw.cal_half
    structured = draw.anchor & ~probes
    n_img = len(offsets) - 1
    img = np.repeat(np.arange(n_img), np.diff(offsets))
    none = np.zeros_like(structured)
    k_s = np.bincount(img[structured], minlength=n_img)
    pred_draw = AnchorDraw(draw.k, structured, none, none, *_anchor_stats(d_obs, img, n_img, structured, none), k_s)
    return pred_draw, probes


def dual_scale(model: DistModel, fields: Fields, b: np.ndarray, offsets: np.ndarray, gamma: float) -> np.ndarray:
    """s shrunk toward its per-image median: med_i (s / med_i) ** gamma. When the structured anchors put the
    shift off everywhere, the residual is roughly flat, q_i is set by the probes with the smallest s, and the
    full range of s (gamma = 1) multiplies widths elsewhere by up to max s / min s."""
    s = model.scale(fields, b)
    if gamma == 1.0:
        return s
    med = np.repeat([np.median(s[a:z]) if z > a else 1.0 for a, z in zip(offsets[:-1], offsets[1:])], np.diff(offsets))
    return med * (s / med) ** gamma


def probe_quantile(model: DistModel, d: np.ndarray, b: np.ndarray, fields: Fields, probes: np.ndarray,
                   offsets: np.ndarray, alpha: float, gamma: float = 0.5) -> np.ndarray:
    """Per image: the split-conformal quantile of |d - m| / s_gamma over its probes (inf if too few)."""
    img = np.repeat(np.arange(len(offsets) - 1), np.diff(offsets))
    r = np.abs(d - fields.shift) / dual_scale(model, fields, b, offsets, gamma)
    return group_conformal_quantile(r[probes], img[probes], len(offsets) - 1, alpha)


def dual_interval(model: DistModel, pred: np.ndarray, b: np.ndarray, fields: Fields, q_img: np.ndarray,
                  offsets: np.ndarray, gamma: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """[pred exp(m - q_i s_gamma), pred exp(m + q_i s_gamma)] on every pixel ([0, inf] where q_i is infinite)."""
    img = np.repeat(np.arange(len(offsets) - 1), np.diff(offsets))
    q = q_img[img] * dual_scale(model, fields, b, offsets, gamma)
    pred = pred.astype(np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        lo, hi = pred * np.exp(fields.shift - q), pred * np.exp(fields.shift + q)
    inf = ~np.isfinite(q)
    lo[inf], hi[inf] = 0.0, np.inf
    return lo, hi
