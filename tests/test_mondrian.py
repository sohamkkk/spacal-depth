import warnings

import numpy as np

from src.calibrators.mondrian import group_offsets, mondrian_crc_lambdas


def _make(rng, n_img, m=60):
    """Two groups per image; group 1 is 3x noisier, so one marginal lambda
    over-covers group 0 and under-covers group 1."""
    groups = np.tile((np.arange(m) % 2), n_img)
    scale = np.where(groups == 1, 3.0, 1.0)
    r = np.abs(rng.normal(size=n_img * m)) * scale
    iop = np.repeat(np.arange(n_img), m)
    return r, iop, groups


def test_group_offsets_filters_small_groups():
    iop = np.array([0, 0, 0, 1, 2, 2])
    in_g = np.array([True, True, False, True, True, True])
    sel, off, keep = group_offsets(iop, in_g, 3, min_pixels=2)
    assert keep.tolist() == [True, False, True]
    assert off.tolist() == [0, 2, 4]
    assert sel.tolist() == [True, True, False, False, True, True]


def test_mondrian_per_group_risk():
    rng = np.random.default_rng(0)
    alpha, reps = 0.1, 150
    loss = {0: [], 1: []}
    for _ in range(reps):
        r, iop, g = _make(rng, 101)
        n_cal_px = 100 * 60
        lams = mondrian_crc_lambdas(r[:n_cal_px], iop[:n_cal_px], g[:n_cal_px], 2, 100, alpha, min_pixels=10)
        for k in (0, 1):
            test = r[n_cal_px:][g[n_cal_px:] == k]
            loss[k].append(np.mean(test > lams[k]))
    for k in (0, 1):
        x = np.array(loss[k])
        assert x.mean() <= alpha + 3 * x.std(ddof=1) / np.sqrt(reps)


def test_tiny_group_gives_inf():
    iop = np.repeat(np.arange(3), 10)
    g = np.zeros(30, dtype=int)
    g[0] = 1  # group 1 exists in one image with one pixel
    r = np.random.default_rng(1).random(30)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lams = mondrian_crc_lambdas(r, iop, g, 2, 3, alpha=0.1, min_pixels=1)
    assert np.isinf(lams[1])
