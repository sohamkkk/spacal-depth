"""Uncalibrated ("raw") uncertainty baselines, read as Gaussian in log depth:
interval = yhat * exp(+-z_{1-alpha/2} sigma(x)). These are what a user gets
from the uncertainty signal alone; the conformal methods are compared to
them (contribution C4: raw vs conformalized).

UniDepthV2's confidence has no raw baseline: it is a within-image ranking
with no absolute scale (median ~0.55-0.68 vs median |log err| ~0.05-0.06,
pooled Spearman with |log err| +0.14 NYU, -0.24 KITTI; experiments/diag_tables.py),
so it is used only inside conformal scores (S4c-conf, S6-fact-conf).
"""

import numpy as np
from scipy.stats import norm

from src.data.prepared import PixelTable

RAW_SIGMA = {
    "raw-flip": lambda t: t.pix["dflip"] / np.sqrt(2.0),  # |d1 - d2| of two noisy draws ~ sqrt(2) sigma (in mean)
    "raw-ens": lambda t: t.pix["ens_std"],
}
RAW_NEEDS = {"raw-flip": "dflip", "raw-ens": "ens_std"}


def available_raw(table: PixelTable) -> list[str]:
    return [k for k, f in RAW_NEEDS.items() if table.has(f)]


def raw_interval(pred: np.ndarray, sigma: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    z = norm.ppf(1 - alpha / 2)
    q = z * np.maximum(sigma.astype(np.float64), 0.0)
    pred = pred.astype(np.float64)
    return pred * np.exp(-q), pred * np.exp(q)
