"""K3: Mondrian (group-conditional) CRC (section 4.3, V3).

Groups are per-pixel labels computed from x only (predicted-depth bins,
predicted-depth edges). For group g, image i's loss is the uncovered
fraction of its group-g pixels; images with fewer than `min_pixels` such
pixels are excluded — the same deterministic filter applies to test
images, so exchangeability among the retained images is preserved and
E[L_g] <= alpha holds for test images that pass the filter.
"""

import numpy as np

from src.calibrators.crc import crc_lambda_or_inf


def group_offsets(image_of_pixel: np.ndarray, in_group: np.ndarray, n_images: int, min_pixels: int):
    """Residual selector + CSR offsets over images that keep >= min_pixels group pixels."""
    counts = np.bincount(image_of_pixel[in_group], minlength=n_images)
    keep_img = counts >= min_pixels
    sel = in_group & keep_img[image_of_pixel]
    kept_counts = counts[keep_img]
    offsets = np.concatenate([[0], np.cumsum(kept_counts)])
    return sel, offsets, keep_img


def mondrian_crc_lambdas(
    r: np.ndarray,
    image_of_pixel: np.ndarray,
    groups: np.ndarray,
    n_groups: int,
    n_images: int,
    alpha: float,
    min_pixels: int = 50,
) -> np.ndarray:
    """lambda_g for g in range(n_groups); inf if a group is too small for alpha.
    Pixels must be sorted by image (CSR order) so selections stay contiguous."""
    lams = np.empty(n_groups)
    for g in range(n_groups):
        sel, offsets, _ = group_offsets(image_of_pixel, groups == g, n_images, min_pixels)
        if len(offsets) < 2:
            lams[g] = np.inf
            continue
        lams[g] = crc_lambda_or_inf(r[sel], offsets, alpha)
    return lams
