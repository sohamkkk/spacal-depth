import numpy as np

from src.calibrators.split_cp import (
    conformal_quantile,
    image_event_lambda,
    patch_ids,
    patch_lambdas,
    per_image_order_stat,
    per_image_patch_max,
    pooled_pixel_lambda,
)


def _sim_images(rng, n, m=40, sigma_g=0.5):
    return [np.abs(rng.normal(rng.normal(0, sigma_g), 1, size=m)) for _ in range(n)]


def _csr(images):
    return np.concatenate(images), np.concatenate([[0], np.cumsum([len(x) for x in images])])


def test_conformal_quantile_rank_and_small_n():
    s = np.arange(1, 20, dtype=float)  # n = 19
    assert conformal_quantile(s, 0.1) == 18.0  # ceil(20 * 0.9) = 18th smallest
    assert conformal_quantile(s[:5], 0.1) == np.inf  # ceil(6 * 0.9) = 6 > 5
    assert conformal_quantile(np.array([1.0, np.inf, 2.0]), 0.5) == 2.0  # infinite scores ignored


def test_order_stat_gives_exact_image_coverage():
    rng = np.random.default_rng(0)
    imgs = [rng.random(37) for _ in range(5)]
    r, off = _csr(imgs)
    beta = 0.1
    stat = per_image_order_stat(r, off, beta)
    for x, s in zip(imgs, stat):
        assert np.mean(x <= s) >= 1 - beta
        assert np.mean(x < s) < 1 - beta  # any smaller lambda fails the event


def test_v4_image_event_guarantee():
    """P(coverage of a new image >= 1 - beta) >= 1 - alpha."""
    rng = np.random.default_rng(1)
    alpha, beta, reps = 0.1, 0.1, 500
    hits = []
    for _ in range(reps):
        imgs = _sim_images(rng, 81)
        r, off = _csr(imgs[:-1])
        lam = image_event_lambda(r, off, alpha, beta)
        hits.append(np.mean(imgs[-1] <= lam) >= 1 - beta)
    rate = np.mean(hits)
    se = np.sqrt(alpha * (1 - alpha) / reps)
    assert rate >= 1 - alpha - 3 * se


def test_pooled_pixel_misses_image_event_under_global_shift():
    rng = np.random.default_rng(2)
    imgs = _sim_images(rng, 400, sigma_g=1.0)
    r, off = _csr(imgs[:200])
    lam0 = pooled_pixel_lambda(r, 0.1)
    lam4 = image_event_lambda(r, off, 0.1, 0.1)
    rate0 = np.mean([np.mean(x <= lam0) >= 0.9 for x in imgs[200:]])
    rate4 = np.mean([np.mean(x <= lam4) >= 0.9 for x in imgs[200:]])
    assert lam4 > lam0 and rate4 > rate0 + 0.1


def test_patch_ids_cover_grid():
    rows, cols = np.meshgrid(np.arange(8), np.arange(12), indexing="ij")
    pid = patch_ids(rows.ravel(), cols.ravel(), np.full(96, 8), np.full(96, 12), (2, 3))
    assert set(pid) == set(range(6))
    assert np.all(np.bincount(pid) == 16)


def test_patch_max_and_validity():
    rng = np.random.default_rng(3)
    n_img, n_p, m = 300, 4, 50
    alpha = 0.1
    iop = np.repeat(np.arange(n_img), m)
    pid = np.tile(np.repeat(np.arange(n_p), m // n_p + 1)[:m], n_img)
    r = np.abs(rng.normal(np.repeat(rng.normal(0, 0.5, n_img), m), 1))
    S = per_image_patch_max(r, iop, pid, n_img, n_p)
    assert np.allclose(S[0, 0], r[:m][pid[:m] == 0].max())
    lam = patch_lambdas(S[:150], alpha)
    rate = np.mean(S[150:] <= lam[None, :])
    assert rate >= 1 - alpha - 3 * np.sqrt(alpha * (1 - alpha) / (150 * n_p)) * 2


def test_scene_event_equals_split_cp_with_one_image_per_scene():
    from src.calibrators.registry import scene_event_lambda

    s = np.random.default_rng(4).random(50)
    for alpha in (0.05, 0.1, 0.2):
        assert scene_event_lambda(s, np.arange(50), alpha) == conformal_quantile(s, alpha)
    assert scene_event_lambda(s[:5], np.arange(5), 0.1) == np.inf


def _clustered_scenes(rng, n_scenes):
    """Big scenes are easy, small scenes hard (as in NYU): image-level calibration
    is dominated by the big scenes and under-covers a random image of a new scene."""
    sizes = rng.integers(1, 30, n_scenes)
    hard = 1.0 + 3.0 / sizes
    scores = rng.exponential(np.repeat(hard, sizes) * np.repeat(rng.lognormal(0, 0.3, n_scenes), sizes))
    return scores, np.repeat(np.arange(n_scenes), sizes)


def test_scene_event_guarantee_for_new_scene():
    from src.calibrators.registry import scene_event_lambda

    rng = np.random.default_rng(5)
    alpha, reps = 0.1, 3000
    hit_scene, hit_img = [], []
    for _ in range(reps):
        s, u = _clustered_scenes(rng, 31)
        cal, new = u < 30, u == 30
        new_score = rng.choice(s[new])
        hit_scene.append(new_score <= scene_event_lambda(s[cal], u[cal], alpha))
        hit_img.append(new_score <= conformal_quantile(s[cal], alpha))
    se = np.sqrt(alpha * (1 - alpha) / reps)
    assert np.mean(hit_scene) >= 1 - alpha - 3 * se
    assert np.mean(hit_img) < np.mean(hit_scene)


def test_patch_event_rate_scene_weighting():
    from src.metrics.interval_metrics import patch_event_rate

    # scene a: 3 images all hit; scene b: 1 image, miss. image-level 3/4, scene-level 1/2
    S = np.array([[0.1], [0.2], [0.3], [5.0]])
    lam = np.array([1.0])
    assert patch_event_rate(S, lam) == 0.75
    assert patch_event_rate(S, lam, np.array(["a", "a", "a", "b"])) == 0.5
