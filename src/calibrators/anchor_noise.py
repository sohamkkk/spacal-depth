"""Anchor values that differ from the ground truth (E16): a noisy sensor, or real LiDAR returns.

An anchor's value is the log residual its sensor reports, d_obs = log z_sensor - log yhat; coverage is always
measured against the ground truth. Specs:
  clean         d_obs = d (the GT value, as in E11-E15)
  gauss:<s>     d + N(0, s^2): relative noise of about s * 100 %
  outlier:<p>   with probability p a gross error, d + sign * U(log 1.25, log 2) (a 25-100 % wrong reading)
  bias:<b>      d + b: a sensor whose scale is off by about b * 100 %
  lidar         KITTI only: the single raw LiDAR scan at the pixel (table field `lidar`, 0 = no return,
                added by scripts/add_kitti_lidar.py); anchors only where a return exists

Cross-image methods stay valid when calibration anchors come from the same sensor, because their scores use
the true residual at evaluation pixels and anchor values only as features. A-local's within-image quantile is
taken over anchor values, so it covers the sensor's reading, not the ground truth.
"""

import numpy as np

from src.data.prepared import PixelTable

KINDS = ("clean", "gauss", "outlier", "bias", "lidar")
OUTLIER_RANGE = (np.log(1.25), np.log(2.0))


def parse(spec: str) -> tuple[str, float]:
    kind, _, val = spec.partition(":")
    if kind not in KINDS:
        raise ValueError(f"anchor source must be one of {KINDS} (with :value for gauss/outlier/bias), got {spec!r}")
    if kind in ("clean", "lidar"):
        if val:
            raise ValueError(f"{kind} takes no value, got {spec!r}")
        return kind, 0.0
    if not val:
        raise ValueError(f"{kind} needs a value, e.g. {kind}:0.05")
    return kind, float(val)


def available(t: PixelTable, spec: str) -> np.ndarray | None:
    """Pixels that can carry an anchor under this source (None: all)."""
    if parse(spec)[0] != "lidar":
        return None
    if not t.has("lidar"):
        raise ValueError("table has no `lidar` field; run scripts/add_kitti_lidar.py first")
    return t.pix["lidar"] > 0


def anchor_values(t: PixelTable, d: np.ndarray, spec: str, rng: np.random.Generator) -> np.ndarray:
    """Per-pixel value an anchor at that pixel would report (NaN where lidar has no return)."""
    kind, v = parse(spec)
    n = len(d)
    if kind == "clean":
        return d
    if kind == "gauss":
        return d + rng.normal(0.0, v, n)
    if kind == "outlier":
        hit = rng.random(n) < v
        return d + hit * rng.choice([-1.0, 1.0], n) * rng.uniform(*OUTLIER_RANGE, n)
    if kind == "bias":
        return d + v
    z = t.pix["lidar"].astype(np.float64)
    out = np.full(n, np.nan)
    ok = z > 0
    out[ok] = np.log(z[ok]) - np.log(t.pix["pred"][ok].astype(np.float64))
    return out
