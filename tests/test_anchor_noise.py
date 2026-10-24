import numpy as np
import pytest

from src.calibrators.anchor_noise import anchor_values, available, parse
from src.calibrators.anchors import draw_anchors, with_values
from src.data.prepared import PixelTable


def _table(rng, n=20, m=60):
    gt = np.exp(rng.uniform(0, 2, n * m)).astype(np.float32)
    lidar = np.where(rng.random(n * m) < 0.3, gt * np.exp(rng.normal(0, 0.02, n * m)), 0).astype(np.float32)
    lidar[:m] = np.where(np.arange(m) < 4, gt[:m], 0)        # image 0: only 4 returns
    pix = {"gt": gt, "pred": gt * np.float32(1.1), "lidar": lidar,
           "row": np.tile(np.arange(m), n).astype(np.uint16), "col": np.zeros(n * m, np.uint16)}
    t = PixelTable(pix=pix, img={"height": np.full(n, m), "width": np.ones(n)}, offsets=np.arange(n + 1) * m)
    return t, np.log(gt.astype(float)) - np.log(t.pix["pred"].astype(float))


def test_parse_rejects_bad_specs():
    assert parse("gauss:0.05") == ("gauss", 0.05)
    assert parse("lidar") == ("lidar", 0.0)
    for bad in ("gauss", "lidar:1", "laser:0.1"):
        with pytest.raises(ValueError):
            parse(bad)


def test_values_per_source():
    rng = np.random.default_rng(0)
    t, d = _table(rng)
    assert np.array_equal(anchor_values(t, d, "clean", rng), d)
    assert np.allclose(anchor_values(t, d, "bias:0.05", rng) - d, 0.05)
    assert abs(np.std(anchor_values(t, d, "gauss:0.05", rng) - d) - 0.05) < 0.005
    e = np.abs(anchor_values(t, d, "outlier:0.1", rng) - d)
    hit = e > 0
    assert abs(hit.mean() - 0.1) < 0.03 and e[hit].min() >= np.log(1.25) and e[hit].max() <= np.log(2.0)
    v = anchor_values(t, d, "lidar", rng)
    ok = available(t, "lidar")
    assert np.isnan(v[~ok]).all() and np.abs(v[ok] - d[ok]).max() < 0.2


def test_lidar_anchors_only_on_returns_and_capped():
    rng = np.random.default_rng(1)
    t, d = _table(rng)
    ok = available(t, "lidar")
    dr = draw_anchors(d, t.offsets, 10, rng, avail=ok)
    assert not (dr.anchor & ~ok).any()
    assert dr.k_img[0] == 4 and (dr.k_img[1:] == 10).all()
    assert np.array_equal(np.bincount(t.image_of_pixel()[dr.anchor], minlength=t.n_images), dr.k_img)


def test_with_values_keeps_positions_and_recomputes_shift():
    rng = np.random.default_rng(2)
    t, d = _table(rng)
    dr = draw_anchors(d, t.offsets, 10, rng)
    moved = with_values(dr, d + 0.3, t.offsets)
    assert np.array_equal(moved.anchor, dr.anchor) and np.array_equal(moved.cal_half, dr.cal_half)
    assert np.allclose(moved.shift, dr.shift + 0.3) and np.allclose(moved.sigma, dr.sigma)
    same = with_values(dr, d, t.offsets)
    assert np.array_equal(same.shift_half, dr.shift_half)
