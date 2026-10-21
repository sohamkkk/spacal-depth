"""Interval quality on a test PixelTable. Every image-level statistic weights
images equally (the image is the exchangeable unit); pooled-pixel
coverage is reported only to expose the pooled-vs-per-image gap.
"""

import numpy as np

from src.data.prepared import PixelTable
from src.data.splits import exchange_units


def _scene_mean(table: PixelTable, cov_img: np.ndarray) -> float:
    """Coverage averaged per exchangeable unit first (what K1-scene guarantees)."""
    if "scene_id" not in table.img:
        return float("nan")
    _, inv = np.unique(exchange_units(table), return_inverse=True)
    return float((np.bincount(inv, weights=cov_img) / np.bincount(inv)).mean())


def _per_image_mean(x: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    return np.add.reduceat(x.astype(np.float64), offsets[:-1]) / np.diff(offsets)


def interval_score(lo, hi, y, alpha):
    """Winkler / Gneiting-Raftery interval score (lower is better)."""
    return (hi - lo) + (2.0 / alpha) * (np.maximum(lo - y, 0.0) + np.maximum(y - hi, 0.0))


def evaluate_intervals(
    table: PixelTable,
    lo: np.ndarray,
    hi: np.ndarray,
    alpha: float,
    beta: float = 0.1,
    groups: dict[str, tuple[np.ndarray, list[str]]] | None = None,
    min_group_pixels: int = 50,
) -> tuple[dict, list[dict]]:
    """Returns (scalar metrics, per-group rows)."""
    y = table.pix["gt"].astype(np.float64)
    pred = table.pix["pred"].astype(np.float64)
    off = table.offsets
    covered = (y >= lo) & (y <= hi)
    cov_img = _per_image_mean(covered, off)

    finite = np.isfinite(hi)
    width = np.where(finite, hi - lo, np.inf)
    with np.errstate(invalid="ignore"):
        rel_width = width / pred
        is_m = interval_score(lo, np.where(finite, hi, np.inf), y, alpha)
    m = {
        "cov_pixel": float(covered.mean()),
        "cov_image_mean": float(cov_img.mean()),
        "cov_scene_mean": _scene_mean(table, cov_img),
        "cov_image_sd": float(cov_img.std(ddof=1)) if len(cov_img) > 1 else 0.0,
        "cov_image_p10": float(np.quantile(cov_img, 0.1)),
        "frac_img_cov_below_1ma_m10": float(np.mean(cov_img < 1 - alpha - 0.10)),
        "v4_rate": float(np.mean(cov_img >= 1 - beta)),
        "v4_rate_scene": _scene_mean(table, (cov_img >= 1 - beta).astype(np.float64)),
        "mpiw": float(_per_image_mean(width, off).mean()),
        "rel_width": float(_per_image_mean(rel_width, off).mean()),
        "median_rel_width": float(np.median(rel_width)),
        "interval_score": float(_per_image_mean(is_m, off).mean()),
        "rel_interval_score": float(_per_image_mean(is_m / y, off).mean()),
        "frac_infinite": float(1 - finite.mean()),
        "n_test_images": int(table.n_images),
        "n_test_pixels": int(table.n_pixels),
    }

    rows = []
    if groups:
        iop = table.image_of_pixel()
        worst = 0.0
        for gname, (labels, names) in groups.items():
            for g, label_name in enumerate(names):
                in_g = labels == g
                counts = np.bincount(iop[in_g], minlength=table.n_images)
                keep = counts >= min_group_pixels
                if not keep.any():
                    continue
                hits = np.bincount(iop[in_g], weights=covered[in_g], minlength=table.n_images)
                cov_g = hits[keep] / counts[keep]
                rows.append({
                    "group_family": gname,
                    "group": label_name,
                    "cov_image_mean": float(cov_g.mean()),
                    "cov_pixel": float(covered[in_g].mean()),
                    "n_images": int(keep.sum()),
                    "rel_width": float(np.median(rel_width[in_g])),
                })
                if gname in ("pred_depth", "edge"):
                    worst = max(worst, (1 - alpha) - float(cov_g.mean()))
        m["worst_group_gap"] = worst
    return m, rows


def patch_event_rate(S_test: np.ndarray, lam_patch: np.ndarray, units: np.ndarray | None = None) -> float:
    """Fraction of (image, patch) pairs with every valid pixel covered. With
    `units` (unit of each image): per patch, averaged over units first (what
    K2-scene guarantees), then over patches."""
    valid = np.isfinite(S_test)
    hit = S_test <= lam_patch[None, :]
    if units is None:
        return float(hit[valid].mean())
    _, inv = np.unique(units, return_inverse=True)
    rates = []
    for p in range(S_test.shape[1]):
        ok = valid[:, p]
        if ok.any():
            u = inv[ok]
            rates.append((np.bincount(u, weights=hit[ok, p].astype(np.float64)) / np.maximum(np.bincount(u), 1))[np.unique(u)].mean())
    return float(np.mean(rates))
