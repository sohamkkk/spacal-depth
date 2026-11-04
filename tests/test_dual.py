import numpy as np
import pytest

from src.calibrators.anchor_dist import DistModel, FieldConfig, anchor_fields
from src.calibrators.anchors import anchor_keys, draw_split_anchors
from src.calibrators.dual import dual_interval, probe_quantile, split_roles
from tests.test_anchors_dist import _images


def _coverage_over_probe_draws(t, d, key, k, k_random, alpha, n_rep, seed, gamma):
    """Per image, coverage of the evaluation pixels averaged over n_rep probe draws (fixed structured anchors)."""
    model = DistModel(FieldConfig(kind="kernel", h=0.1, spread="local"), tau=0.05, sigma_fill=0.1, edges=())
    b = np.ones(t.n_pixels)
    img = t.image_of_pixel()
    rng = np.random.default_rng(seed)
    cov, fields0 = np.zeros((n_rep, t.n_images)), None
    for rep in range(n_rep):
        draw = draw_split_anchors(d, t.offsets, k, k_random, rng, key)
        pdraw, probes = split_roles(draw, d, t.offsets)
        if fields0 is None:
            fields0 = anchor_fields(t, d, pdraw, model.cfg)
            structured0 = pdraw.anchor
        assert np.array_equal(pdraw.anchor, structured0)  # the prediction anchors stay fixed
        q = probe_quantile(model, d, b, fields0, probes, t.offsets, alpha, gamma)
        lo, hi = dual_interval(model, t.pix["pred"], b, fields0, q, t.offsets, gamma)
        y = t.pix["gt"].astype(np.float64)
        ev = draw.eval
        hit = (y >= lo) & (y <= hi)
        cov[rep] = np.bincount(img[ev], weights=hit[ev], minlength=t.n_images) / np.bincount(img[ev], minlength=t.n_images)
    return cov.mean(axis=0), cov.std(axis=0) / np.sqrt(n_rep)


@pytest.mark.parametrize("gamma", [1.0, 0.5, 0.0])
def test_per_image_validity_under_clustered_prediction_anchors(gamma):
    """With one compact cluster of prediction anchors the shift is wrong far from it, yet every image's
    coverage averaged over probe draws is >= 1 - alpha, and at most 1 - alpha + 1 / (n + 1), for any
    scale shrinkage gamma."""
    rng = np.random.default_rng(0)
    t, d = _images(rng, 4, smooth=0.6)
    key = anchor_keys("cluster", t.pix, t.offsets, rng)
    alpha, n_probe = 0.1, 20
    mean, se = _coverage_over_probe_draws(t, d, key, 60, n_probe, alpha, n_rep=1500, seed=1, gamma=gamma)
    assert np.all(mean >= 1 - alpha - 4 * se), mean
    assert np.all(mean <= 1 - alpha + 1 / (n_probe + 1) + 4 * se), mean


def test_probe_values_do_not_enter_the_fields():
    rng = np.random.default_rng(2)
    t, d = _images(rng, 3, smooth=0.5)
    key = anchor_keys("rows", t.pix, t.offsets, rng)
    draw = draw_split_anchors(d, t.offsets, 50, 20, rng, key)
    pdraw, probes = split_roles(draw, d, t.offsets)
    d2 = d.copy()
    d2[probes] += 5.0
    pdraw2, _ = split_roles(draw, d2, t.offsets)
    cfg = FieldConfig(kind="kernel", h=0.1, spread="local")
    f1, f2 = anchor_fields(t, d, pdraw, cfg), anchor_fields(t, d2, pdraw2, cfg)
    assert np.array_equal(f1.shift, f2.shift) and np.array_equal(f1.sigma, f2.sigma, equal_nan=True)
    assert np.array_equal(pdraw.shift, pdraw2.shift)
    assert not (pdraw.anchor & probes).any() and np.array_equal(pdraw.anchor | probes, draw.anchor)


def test_too_few_probes_give_infinite_intervals():
    rng = np.random.default_rng(3)
    t, d = _images(rng, 2)
    draw = draw_split_anchors(d, t.offsets, 30, 8, rng, anchor_keys("cluster", t.pix, t.offsets, rng))
    pdraw, probes = split_roles(draw, d, t.offsets)
    model = DistModel(FieldConfig(kind="knn"), tau=0.05, sigma_fill=0.1, edges=())
    f = anchor_fields(t, d, pdraw, model.cfg)
    q = probe_quantile(model, d, np.ones(t.n_pixels), f, probes, t.offsets, 0.1)  # ceil(9 * 0.9) = 9 > 8
    assert np.all(np.isinf(q))
    lo, hi = dual_interval(model, t.pix["pred"], np.ones(t.n_pixels), f, q, t.offsets)
    assert np.all(lo == 0) and np.all(np.isinf(hi))
