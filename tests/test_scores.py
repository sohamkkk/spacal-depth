import numpy as np
import pytest

from src.data.prepared import PixelTable, build_table
from src.data.synthetic import write_synthetic_cache
from src.metrics.interval_metrics import evaluate_intervals, interval_score
from src.scores.scale_maps import (
    SCORES,
    HeuristicScale,
    available_scores,
    build_scale_model,
    interval,
    residual,
)


@pytest.fixture(scope="module")
def table(tmp_path_factory):
    root = tmp_path_factory.mktemp("cache")
    write_synthetic_cache(root, "nyu", n_scenes=30, imgs_per_scene=3, h=40, w=48)
    return build_table("nyu", "dav2-indoor-large", ("dav2-indoor-base", "dav2-indoor-small"),
                       stride=2, min_valid=50, cache_root=root)


@pytest.mark.parametrize("space", ["lin", "log"])
def test_residual_interval_duality(table, space):
    """y in interval(pred, q)  <=>  residual <= q  (what makes CRC on r valid for the interval)."""
    rng = np.random.default_rng(0)
    q = rng.uniform(0, 1.5, table.n_pixels)
    lo, hi = interval(table.pix["pred"], q, space)
    y = table.pix["gt"].astype(np.float64)
    covered = (y >= lo) & (y <= hi)
    r = residual(table, space)
    if space == "lin":  # the lower clip at 0 can only add coverage, never remove it
        assert np.all(covered[r <= q])
    else:
        assert np.array_equal(covered, r <= q * (1 + 1e-12)) or np.mean(covered == (r <= q)) > 0.9999


def test_every_score_fits_and_is_positive(table):
    head = table.subset(np.arange(45))
    other = table.subset(np.arange(45, table.n_images))
    for name in available_scores(table):
        model = build_scale_model(SCORES[name].scale, seed=0).fit(head)
        s = model.predict(other)
        assert s.shape == (other.n_pixels,), name
        assert np.all(np.isfinite(s)) and np.all(s > 0), name


def test_heuristic_floor_from_grid_and_cap(table):
    m = HeuristicScale("dflip").fit(table)
    assert m.floor_frac in HeuristicScale.FLOOR_GRID
    assert m.transform in HeuristicScale.TRANSFORMS
    s = m.predict(table)
    assert s.max() <= HeuristicScale.TRANSFORMS[m.transform](m.cap) + m.floor_frac + 1e-12


def _heavy_tail_table(seed=0, n_img=120, m=200):
    """Log-normal residuals whose scale grows only mildly with a field that
    spans ~5 orders of magnitude (like gradient magnitudes)."""
    rng = np.random.default_rng(seed)
    n = n_img * m
    field = np.exp(2.5 * rng.standard_normal(n))
    sigma = 0.1 * (1 + 0.3 * np.log1p(field))
    pred = np.full(n, 3.0)
    gt = pred * np.exp(sigma * rng.standard_normal(n))
    return PixelTable(
        pix={"pred": pred.astype(np.float32), "gt": gt.astype(np.float32), "grad_unc": field.astype(np.float32),
             "row": np.tile(np.arange(m), n_img).astype(np.uint16), "col": np.zeros(n, np.uint16)},
        img={"height": np.full(n_img, m), "width": np.ones(n_img)},
        offsets=np.arange(n_img + 1) * m,
        meta={"dataset": "nyu"},
    )


def test_heuristic_compresses_heavy_tails():
    head, test = _heavy_tail_table(0), _heavy_tail_table(1)
    m = HeuristicScale("grad_unc").fit(head)
    assert m.transform != "id"
    from src.calibrators.crc import crc_lambda_exact
    lam = crc_lambda_exact(residual(head, "log") / m.predict(head), head.offsets, 0.1)
    lo, hi = interval(test.pix["pred"], lam * m.predict(test), "log")
    assert np.max((hi - lo) / test.pix["pred"]) < 50


def test_factorized_learns_image_factor():
    """If per-image error magnitude is visible in an image feature, a(x) must track it."""
    rng = np.random.default_rng(0)
    n_img, m = 200, 100
    level = rng.uniform(0.02, 0.4, n_img)  # true per-image error scale
    pred = np.full(n_img * m, 3.0)
    gt = pred * np.exp(np.repeat(level, m) * rng.standard_normal(n_img * m))
    t = PixelTable(
        pix={"pred": pred.astype(np.float32), "gt": gt.astype(np.float32),
             "row": np.tile(np.arange(m), n_img).astype(np.uint16), "col": np.zeros(n_img * m, np.uint16),
             "conf": np.ones(n_img * m, np.float32)},
        img={"med_conf": level * np.exp(0.05 * rng.standard_normal(n_img)), "med_logpred": np.zeros(n_img),
             "height": np.full(n_img, m), "width": np.ones(n_img)},
        offsets=np.arange(n_img + 1) * m,
        meta={"dataset": "nyu"},
    )
    model = build_scale_model("factorized:heur:conf").fit(t)
    a = model.image_factor(t)
    assert np.corrcoef(np.log(a), np.log(level))[0, 1] > 0.9

    # shifted images (image feature 100x outside the head range) must not extrapolate s
    shifted = PixelTable(pix=t.pix, img={**t.img, "med_conf": t.img["med_conf"] * 100}, offsets=t.offsets, meta=t.meta)
    s = model.predict(shifted)
    assert np.all(np.isfinite(s)) and s.max() <= model.s_max <= residual(t, "log").max()


def test_interval_metrics_basic():
    t = PixelTable(
        pix={"pred": np.array([1.0, 1.0, 2.0, 2.0]), "gt": np.array([1.0, 1.5, 2.0, 3.0]),
             "edge": np.zeros(4, bool), "row": np.zeros(4, np.uint16), "col": np.zeros(4, np.uint16)},
        img={"height": np.array([1, 1]), "width": np.array([1, 1])},
        offsets=np.array([0, 2, 4]),
        meta={"dataset": "nyu"},
    )
    lo, hi = np.array([0.9, 0.9, 1.9, 1.9]), np.array([1.1, 1.1, 2.1, 3.5])
    m, _ = evaluate_intervals(t, lo, hi, alpha=0.1, beta=0.4)
    assert m["cov_pixel"] == 0.75
    assert m["cov_image_mean"] == 0.75  # (0.5 + 1.0) / 2
    assert m["v4_rate"] == 0.5
    assert np.isclose(interval_score(np.array([0.9]), np.array([1.1]), np.array([1.5]), 0.1)[0], 0.2 + 20 * 0.4)
