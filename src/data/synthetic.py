"""Synthetic caches in the exact on-disk layout of scripts/cache_predictions.py,
for smoke tests and laptop dry runs without data or a GPU.

Error model (log space): log pred - log gt = g_i + u_ij * z_ij, where
g_i ~ N(0, sigma_g) is a per-image global scale error (shared by all
pixels), u_ij a smooth heteroscedastic magnitude, z_ij a spatially
correlated unit field. Auxiliary signals (flip, conf, grad) are noisy
functions of u — informative but imperfect, like the real ones.
"""

from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter


def _grf(rng, h, w, ell):
    z = gaussian_filter(rng.standard_normal((h, w)), ell, mode="wrap")
    return z / (z.std() + 1e-12)


def write_synthetic_cache(
    root: Path,
    dataset: str = "nyu",
    model_tags: tuple[str, ...] = ("dav2-indoor-large", "dav2-indoor-base", "dav2-indoor-small"),
    n_scenes: int = 40,
    imgs_per_scene: int = 4,
    h: int = 48,
    w: int = 64,
    sigma_g: float = 0.1,
    seed: int = 0,
) -> None:
    rng = np.random.default_rng(seed)
    root = Path(root) / dataset
    (root / "gt").mkdir(parents=True, exist_ok=True)
    for t in model_tags:
        (root / t).mkdir(parents=True, exist_ok=True)
    max_d = 10.0 if "indoor" in model_tags[0] or dataset in ("nyu", "diode_indoor") else 80.0
    yy = np.linspace(0, 1, h)[:, None]
    for s in range(n_scenes):
        scene_base = 0.2 * max_d * (1 + rng.random()) * (0.5 + yy) * np.ones((1, w))
        for k in range(imgs_per_scene):
            sid = f"{dataset}_{s:03d}_{k:02d}"
            gt = np.clip(scene_base * np.exp(0.2 * _grf(rng, h, w, 6)), 0.05, 0.95 * max_d).astype(np.float32)
            mask = rng.random((h, w)) > 0.1
            labels = rng.integers(0, 900, (h, w)).astype(np.uint16)
            extra = {}
            if k == 0:
                gray = (255 * gt / gt.max()).astype(np.uint8)
                extra["rgb"] = np.stack([gray] * 3, axis=-1)
            np.savez_compressed(
                root / "gt" / f"{sid}.npz", gt=gt, mask=mask, labels=labels, scene_id=f"scene_{s:03d}",
                sample_id=sid, official_test=bool(s % 2), subset="", **extra,
            )
            g = rng.normal(0, sigma_g)
            u = 0.03 + 0.12 * (gaussian_filter(rng.random((h, w)), 4) > 0.5) + 0.05 * yy
            for j, tag in enumerate(model_tags):
                e = g + u * _grf(rng, h, w, 2) + 0.02 * j * rng.standard_normal((h, w))
                pred = (gt * np.exp(e)).astype(np.float32)
                payload = {"pred": pred, "t_pred": np.float32(0.01)}
                if j == 0:
                    payload["pred_flip"] = (pred * np.exp(u * 0.7 * rng.standard_normal((h, w)))).astype(np.float16)
                    payload["grad_unc"] = (u * np.exp(0.3 * rng.standard_normal((h, w)))).astype(np.float16)
                    payload["conf"] = (u * 0.8 + 0.01 * rng.random((h, w))).astype(np.float16)
                np.savez_compressed(root / tag / f"{sid}.npz", **payload)
