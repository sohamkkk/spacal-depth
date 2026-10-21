"""One entry point that turns calibration residuals into per-pixel
thresholds for test pixels, for every calibrator in the study.

  K0      pooled-pixel split CP (naive; pixels treated as exchangeable)
  K1      image-level CRC, exact            -> V1: E[uncovered fraction] <= alpha
  K1-scene  scene-level CRC, exact (scene = split unit) -> V1 per scene: E[scene-mean uncovered fraction] <= alpha
  RCPS    image-level RCPS (Hoeffding, delta) -> V1 with prob. >= 1 - delta
  K2      per-patch max-score split CP       -> V2: P(patch fully covered) >= 1 - alpha
  K3-*    Mondrian CRC on x-only groups      -> V3: per-group V1
  V4      image-event split CP               -> P(image coverage >= 1 - beta) >= 1 - alpha
  V4-scene  scene-level CRC on the image-event loss -> same, for a random image of a NEW scene
  K2-scene  scene-level CRC on the patch-event loss -> V2 for a random image of a NEW scene

V4 and K2 treat images as exchangeable; images of one scene are not, which
shows up as a small systematic shortfall on real data. The -scene variants use
CRC with the scene as unit (crc_lambda_exact_units): the loss of a scene is the
fraction of its images that fail the event, so E[that fraction] <= alpha for a
new scene, i.e. P(event fails for a random image of a new scene) <= alpha.
They need >= 1/alpha - 1 calibration scenes (else infinite).
"""

import time
from dataclasses import dataclass, field

import numpy as np

from src.calibrators.crc import InfeasibleRiskError, crc_lambda_exact, crc_lambda_exact_units, rcps_lambda_hoeffding
from src.calibrators.mondrian import mondrian_crc_lambdas
from src.calibrators.split_cp import (
    image_event_lambda,
    patch_ids,
    patch_lambdas,
    per_image_order_stat,
    per_image_patch_max,
    pooled_pixel_lambda,
)
from src.data.prepared import PixelTable
from src.data.splits import exchange_units
from src.metrics.interval_metrics import patch_event_rate
from src.scores.groups import calibration_groups

CALIBRATORS = ("K0", "K1", "K1-scene", "RCPS", "K2", "K2-scene", "K3-depth", "K3-edge", "V4", "V4-scene")
PATCH_GRID = (4, 4)


@dataclass
class CalibResult:
    lam_pix: np.ndarray            # per test pixel lambda (multiplies s_test)
    lam_summary: float             # scalar lambda, or median over groups/patches
    seconds: float
    extra: dict = field(default_factory=dict)


def _pids(t: PixelTable) -> np.ndarray:
    iop = t.image_of_pixel()
    return patch_ids(t.pix["row"], t.pix["col"], t.img["height"][iop], t.img["width"][iop], PATCH_GRID)


def _scalar(fn) -> float:
    try:
        return fn()
    except InfeasibleRiskError:
        return float("inf")


def scene_event_lambda(score_img: np.ndarray, units: np.ndarray, alpha: float) -> float:
    """CRC over scenes on the binary per-image loss 1{score_i > lambda}
    (one "pixel" per image, so a scene's loss is its fraction of failing images)."""
    score_img = np.asarray(score_img, dtype=np.float64)
    offsets = np.arange(len(score_img) + 1)
    return _scalar(lambda: crc_lambda_exact_units(score_img, offsets, units, alpha))


def scene_patch_lambdas(S_cal: np.ndarray, units: np.ndarray, alpha: float) -> np.ndarray:
    """Per patch scene_event_lambda over the images with a valid pixel in that patch."""
    out = np.full(S_cal.shape[1], np.inf)
    for p in range(S_cal.shape[1]):
        ok = np.isfinite(S_cal[:, p])
        if ok.any():
            out[p] = scene_event_lambda(S_cal[ok, p], units[ok], alpha)
    return out


def calibrate(
    method: str,
    cal: PixelTable,
    r_cal: np.ndarray,
    test: PixelTable,
    r_test: np.ndarray | None,
    alpha: float,
    beta: float = 0.1,
    delta: float = 0.1,
    min_group_pixels: int = 50,
) -> CalibResult:
    t0 = time.perf_counter()
    extra: dict = {}
    if method in ("K0", "K1", "K1-scene", "RCPS", "V4", "V4-scene"):
        lam = {
            "K0": lambda: pooled_pixel_lambda(r_cal, alpha),
            "K1": lambda: _scalar(lambda: crc_lambda_exact(r_cal, cal.offsets, alpha)),
            "K1-scene": lambda: _scalar(lambda: crc_lambda_exact_units(r_cal, cal.offsets, exchange_units(cal), alpha)),
            "RCPS": lambda: _scalar(lambda: rcps_lambda_hoeffding(r_cal, cal.offsets, alpha, delta)),
            "V4": lambda: image_event_lambda(r_cal, cal.offsets, alpha, beta),
            "V4-scene": lambda: scene_event_lambda(per_image_order_stat(r_cal, cal.offsets, beta),
                                                   exchange_units(cal), alpha),
        }[method]()
        return CalibResult(np.full(test.n_pixels, lam), lam, time.perf_counter() - t0)

    if method in ("K2", "K2-scene"):
        n_p = PATCH_GRID[0] * PATCH_GRID[1]
        S_cal = per_image_patch_max(r_cal, cal.image_of_pixel(), _pids(cal), cal.n_images, n_p)
        lam_p = patch_lambdas(S_cal, alpha) if method == "K2" else scene_patch_lambdas(S_cal, exchange_units(cal), alpha)
        pid_test = _pids(test)
        secs = time.perf_counter() - t0
        if r_test is not None:
            S_test = per_image_patch_max(r_test, test.image_of_pixel(), pid_test, test.n_images, n_p)
            extra["patch_event_rate"] = patch_event_rate(S_test, lam_p)
            if "scene_id" in test.img:
                extra["patch_event_rate_scene"] = patch_event_rate(S_test, lam_p, exchange_units(test))
        return CalibResult(lam_p[pid_test], float(np.median(lam_p)), secs, extra)

    if method.startswith("K3-"):
        kind = {"K3-depth": "pred_depth", "K3-edge": "edge"}[method]
        g_cal, names = calibration_groups(cal, kind)
        g_test, _ = calibration_groups(test, kind)
        lam_g = mondrian_crc_lambdas(
            r_cal, cal.image_of_pixel(), g_cal, len(names), cal.n_images, alpha, min_group_pixels
        )
        extra["lambda_groups"] = dict(zip(names, map(float, lam_g)))
        return CalibResult(lam_g[g_test], float(np.median(lam_g)), time.perf_counter() - t0, extra)

    raise ValueError(f"unknown calibrator {method!r}")
