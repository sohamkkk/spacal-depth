"""Image-level Conformal Risk Control (section 4.4 / 6.4 of the build guide).

Exchangeable unit is the IMAGE, not the pixel (section 5.1) — each
calibration image contributes exactly one loss curve
L_i(lambda) = fraction of its valid pixels with normalised residual > lambda,
so pooling never treats correlated pixels as independent samples.

`crc_lambda_exact` solves the CRC condition exactly (no grid) from the
pooled residuals weighted by 1 / (valid pixels of their image):
    sum_i L_i(lambda) = sum_j w_j 1{r_j > lambda}  <=  budget.
CRC (Angelopoulos et al., ICLR 2024): budget = (n + 1) alpha - B.
RCPS with a Hoeffding bound (Bates et al.; used by im2im-uq):
    budget = n (alpha - sqrt(log(1/delta) / (2n))).
"""

import warnings

import numpy as np


class InfeasibleRiskError(ValueError):
    """No finite lambda satisfies the risk bound (alpha too small for n)."""


def per_image_losses(resid_norm_list: list[np.ndarray], lambdas: np.ndarray) -> np.ndarray:
    """L[i, j] = fraction of valid pixels in image i with normalised
    residual |y - yhat| / s greater than lambdas[j] (non-increasing in lambda)."""
    L = np.empty((len(resid_norm_list), len(lambdas)))
    for i, r in enumerate(resid_norm_list):
        if len(r) == 0:
            raise ValueError(f"image {i} has no valid pixels; filter it out before calibration")
        r = np.sort(r)
        L[i] = 1.0 - np.searchsorted(r, lambdas, side="right") / len(r)
    return L


def crc_lambda(L: np.ndarray, lambdas: np.ndarray, alpha: float, B: float = 1.0) -> float:
    """Smallest lambda on the grid with (sum_i L_i(lambda) + B) / (n + 1) <= alpha.
    lambdas must be ascending. Raises InfeasibleRiskError if no grid value works."""
    n = L.shape[0]
    risk = (L.sum(axis=0) + B) / (n + 1)
    ok = np.where(risk <= alpha)[0]
    if len(ok) == 0:
        raise InfeasibleRiskError(
            f"no lambda in [{lambdas[0]:.3g}, {lambdas[-1]:.3g}] reaches risk {alpha} with n={n}; "
            "extend the grid or use more calibration images"
        )
    return float(lambdas[ok[0]])


def calibrate(resid_norm_list: list[np.ndarray], alpha: float, lambda_max: float = 50.0, n_grid: int = 2000) -> float:
    """Convenience wrapper: dense log-spaced grid + crc_lambda."""
    lambdas = np.concatenate([[0.0], np.geomspace(1e-4, lambda_max, n_grid)])
    L = per_image_losses(resid_norm_list, lambdas)
    return crc_lambda(L, lambdas, alpha)


def _min_lambda_under_budget(r: np.ndarray, w: np.ndarray, budget: float) -> float:
    """Smallest lambda with sum_j w_j 1{r_j > lambda} <= budget (lambda in {0} U r)."""
    if budget < 0:
        raise InfeasibleRiskError(f"risk budget {budget:.4g} < 0: alpha too small for this many images")
    if w[r > 0].sum() <= budget:
        return 0.0
    order = np.argsort(r, kind="stable")
    rs, ws = r[order], w[order]
    # weight strictly above rs[k] (ties included) as a suffix sum, so it is exactly 0
    # at the maximum; total - cumsum leaves rounding noise that breaks a zero budget
    last_of_tie = np.searchsorted(rs, rs, side="right") - 1
    above = np.append(np.cumsum(ws[::-1])[::-1][1:], 0.0)[last_of_tie]
    k = np.argmax(above <= budget + 1e-12)
    return float(rs[k])


def _image_weights(offsets: np.ndarray) -> tuple[np.ndarray, int]:
    counts = np.diff(offsets)
    if np.any(counts == 0):
        raise ValueError("an image has no valid pixels; filter it out before calibration")
    return np.repeat(1.0 / counts, counts), len(counts)


def crc_lambda_exact(r: np.ndarray, offsets: np.ndarray, alpha: float, B: float = 1.0) -> float:
    """Exact CRC lambda. `r` holds pooled normalised residuals of the
    calibration images; image i owns r[offsets[i]:offsets[i+1]]."""
    w, n = _image_weights(offsets)
    return _min_lambda_under_budget(r, w, (n + 1) * alpha - B)


def crc_lambda_exact_units(r: np.ndarray, offsets: np.ndarray, unit_of_image: np.ndarray, alpha: float,
                           B: float = 1.0) -> float:
    """Exact CRC with a coarser exchangeable unit (scene): the loss of a unit
    is the mean uncovered fraction of its images, the budget counts units.
    Guarantees E[scene-averaged miscoverage] <= alpha for a new scene."""
    w_img, _ = _image_weights(offsets)
    _, inv, cnt = np.unique(np.asarray(unit_of_image), return_inverse=True, return_counts=True)
    w = w_img * np.repeat(1.0 / cnt[inv], np.diff(offsets))
    return _min_lambda_under_budget(r, w, (len(cnt) + 1) * alpha - B)


def rcps_lambda_hoeffding(r: np.ndarray, offsets: np.ndarray, alpha: float, delta: float = 0.1) -> float:
    """High-probability version: P(E[L(lambda_hat)] > alpha) <= delta (Hoeffding UCB)."""
    w, n = _image_weights(offsets)
    return _min_lambda_under_budget(r, w, n * (alpha - np.sqrt(np.log(1 / delta) / (2 * n))))


def crc_lambda_or_inf(r: np.ndarray, offsets: np.ndarray, alpha: float, B: float = 1.0) -> float:
    """Like crc_lambda_exact but returns inf (with a warning) when infeasible —
    for Mondrian groups that can be tiny. Callers must report inf intervals."""
    try:
        return crc_lambda_exact(r, offsets, alpha, B)
    except InfeasibleRiskError as e:
        warnings.warn(str(e))
        return float("inf")
