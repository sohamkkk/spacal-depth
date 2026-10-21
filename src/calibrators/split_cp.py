"""Split-conformal calibrators over exchangeable images (sections 4.1, 5.2).

- `conformal_quantile`: the ceil((n+1)(1-alpha))-th smallest score.
- K0 `pooled_pixel_lambda`: the deliberately naive baseline that treats every
  pixel as an exchangeable sample (n = #pixels).
- V4 `image_event_lambda`: P(coverage of a new image >= 1 - beta) >= 1 - alpha,
  using one score per image = the order statistic that makes exactly
  ceil(m (1 - beta)) of its m pixels covered.
- V2 `patch_lambdas`: per fixed patch (a function of pixel position only),
  P(every valid pixel of that patch is covered) >= 1 - alpha, from one
  max-score per image per patch.
"""

import numpy as np


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample-valid threshold; inf if n is too small for alpha."""
    scores = np.asarray(scores, dtype=np.float64)
    scores = scores[np.isfinite(scores)]
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if n == 0 or k > n:
        return float("inf")
    return float(np.partition(scores, k - 1)[k - 1])


def pooled_pixel_lambda(r: np.ndarray, alpha: float) -> float:
    """K0: pixels as if i.i.d. — no valid image-level statement."""
    return conformal_quantile(r, alpha)


def per_image_order_stat(r: np.ndarray, offsets: np.ndarray, beta: float) -> np.ndarray:
    """score_i such that lambda >= score_i  <=>  image i has >= (1 - beta) coverage."""
    out = np.empty(len(offsets) - 1)
    for i in range(len(out)):
        ri = r[offsets[i] : offsets[i + 1]]
        if len(ri) == 0:
            raise ValueError(f"image {i} has no valid pixels")
        k = int(np.ceil(len(ri) * (1 - beta)))
        k = min(max(k, 1), len(ri))
        out[i] = np.partition(ri, k - 1)[k - 1]
    return out


def image_event_lambda(r: np.ndarray, offsets: np.ndarray, alpha: float, beta: float) -> float:
    return conformal_quantile(per_image_order_stat(r, offsets, beta), alpha)


def patch_ids(rows: np.ndarray, cols: np.ndarray, height: np.ndarray, width: np.ndarray, grid: tuple[int, int]) -> np.ndarray:
    """Fixed grid cell index of each pixel (depends on position only)."""
    gr, gc = grid
    pr = np.minimum(rows.astype(np.int64) * gr // height, gr - 1)
    pc = np.minimum(cols.astype(np.int64) * gc // width, gc - 1)
    return pr * gc + pc


def per_image_patch_max(r: np.ndarray, image_of_pixel: np.ndarray, pid: np.ndarray, n_images: int, n_patches: int) -> np.ndarray:
    """S[i, p] = max residual of image i in patch p (-inf if no valid pixel)."""
    S = np.full(n_images * n_patches, -np.inf)
    np.maximum.at(S, image_of_pixel * n_patches + pid, r)
    return S.reshape(n_images, n_patches)


def patch_lambdas(S_cal: np.ndarray, alpha: float) -> np.ndarray:
    """One threshold per patch; images without valid pixels in a patch are
    excluded from that patch (same filter applies at test time)."""
    return np.array([conformal_quantile(S_cal[:, p][np.isfinite(S_cal[:, p])], alpha) for p in range(S_cal.shape[1])])
