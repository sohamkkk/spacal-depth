import numpy as np
from sklearn.ensemble import RandomForestRegressor

from src.calibrators.anchors import draw_anchors
from src.calibrators.lscp import lscp_offsets, qrf_weights, weighted_quantiles
from src.data.prepared import PixelTable


def _field_images(rng, n, h=40, w=60):
    """Stationary log residual: global error + smooth sinusoid + pixel noise."""
    rr, cc = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    phase = rng.uniform(0, 2 * np.pi, (n, 1, 1))
    d = (rng.normal(0, 0.2, (n, 1, 1)) + 0.3 * np.sin(2 * np.pi * (rr / h + cc / w) + phase)
         + rng.normal(0, 0.05, (n, h, w)))
    y = np.exp(rng.uniform(0, 2, (n, h, w)))
    pix = {"gt": y.ravel().astype(np.float32), "pred": (y * np.exp(-d)).ravel().astype(np.float32),
           "row": np.tile(rr.ravel(), n).astype(np.uint16), "col": np.tile(cc.ravel(), n).astype(np.uint16)}
    t = PixelTable(pix=pix, img={"height": np.full(n, h), "width": np.full(n, w)},
                   offsets=np.arange(n + 1) * h * w)
    return t, np.log(t.pix["gt"].astype(float)) - np.log(t.pix["pred"].astype(float))


def test_qrf_weights_are_distributions_and_quantiles_follow_the_cdf():
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=(60, 3)), rng.normal(size=60)
    forest = RandomForestRegressor(20, min_samples_leaf=5, random_state=0).fit(x, y)
    w = qrf_weights(forest, x, rng.normal(size=(15, 3)))
    assert np.allclose(w.sum(axis=1), 1) and (w >= 0).all()

    ys = np.array([3.0, 1.0, 2.0, 4.0])
    uniform = np.full((1, 4), 0.25)
    q = weighted_quantiles(ys, uniform, np.array([0.0, 0.25, 0.26, 0.5, 1.0]))[:, 0]
    assert list(q) == [1.0, 1.0, 2.0, 2.0, 4.0]
    first_zero = np.array([[0.5, 0.0, 0.5, 0.0]])            # weights follow ys: y = 1 has none
    assert weighted_quantiles(ys, first_zero, np.array([0.0]))[0, 0] == 2.0


def test_lscp_covers_near_target_on_a_stationary_field_with_many_anchors():
    rng = np.random.default_rng(1)
    t, d = _field_images(rng, 20)
    dr = draw_anchors(d, t.offsets, 150, rng)
    alpha = 0.1
    lo, hi = lscp_offsets(t, d, dr, [alpha], seed=0)[alpha]
    ev = dr.eval
    assert lo.shape == hi.shape == (ev.sum(),) and (lo <= hi).all()
    cov = ((d[ev] >= lo) & (d[ev] <= hi)).mean()
    assert 0.8 <= cov <= 0.97, cov


def test_lscp_is_trivial_with_too_few_anchors():
    rng = np.random.default_rng(2)
    t, d = _field_images(rng, 3)
    dr = draw_anchors(d, t.offsets, 5, rng)
    lo, hi = lscp_offsets(t, d, dr, [0.1])[0.1]
    assert np.isneginf(lo).all() and np.isposinf(hi).all()
