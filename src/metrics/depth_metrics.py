"""Standard monocular depth metrics (section 2.4 of the build guide).

All metrics are computed only over `mask` (valid ground-truth pixels) —
ground truth is never dense (Kinect holes, sparse LiDAR), so unmasked
pixels would silently corrupt every number. Predictions are clamped to
`min_depth` so ratios and logs stay finite. Dataset-level numbers follow the
standard protocol: compute per image, then average over images.
"""

import numpy as np

MIN_DEPTH = 1e-3


def _valid(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray | None):
    if mask is None:
        mask = np.isfinite(gt) & (gt > 0)
    p, g = pred[mask], gt[mask]
    if p.size == 0:
        raise ValueError("empty validity mask")
    return np.maximum(p, MIN_DEPTH), g


def abs_rel(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray | None = None) -> float:
    p, g = _valid(pred, gt, mask)
    return float(np.mean(np.abs(p - g) / g))


def rmse(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray | None = None) -> float:
    p, g = _valid(pred, gt, mask)
    return float(np.sqrt(np.mean((p - g) ** 2)))


def delta_k(pred: np.ndarray, gt: np.ndarray, k: int = 1, mask: np.ndarray | None = None) -> float:
    p, g = _valid(pred, gt, mask)
    ratio = np.maximum(p / g, g / p)
    return float(np.mean(ratio < 1.25**k))


def silog(pred: np.ndarray, gt: np.ndarray, lam: float = 0.85, mask: np.ndarray | None = None) -> float:
    """Eigen et al. form with lambda=0.85, unscaled (BTS/ZoeDepth multiply by 100)."""
    p, g = _valid(pred, gt, mask)
    d = np.log(p) - np.log(g)
    return float(np.sqrt(max(np.mean(d**2) - lam * np.mean(d) ** 2, 0.0)))


def all_metrics(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray | None = None) -> dict:
    return {
        "AbsRel": abs_rel(pred, gt, mask),
        "RMSE": rmse(pred, gt, mask),
        "delta1": delta_k(pred, gt, 1, mask),
        "delta2": delta_k(pred, gt, 2, mask),
        "delta3": delta_k(pred, gt, 3, mask),
        "SILog": silog(pred, gt, mask=mask),
    }


def dataset_metrics(per_image: list[dict]) -> dict:
    """Average per-image metric dicts (standard protocol)."""
    if not per_image:
        raise ValueError("no images")
    return {k: float(np.mean([m[k] for m in per_image])) for k in per_image[0]}
