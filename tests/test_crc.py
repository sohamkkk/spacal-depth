import numpy as np
import pytest

from src.calibrators.crc import (
    InfeasibleRiskError,
    calibrate,
    crc_lambda,
    crc_lambda_exact,
    crc_lambda_exact_units,
    crc_lambda_or_inf,
    per_image_losses,
    rcps_lambda_hoeffding,
)


def _csr(images):
    return np.concatenate(images), np.concatenate([[0], np.cumsum([len(x) for x in images])])


def test_losses_are_non_increasing_in_lambda():
    rng = np.random.default_rng(0)
    resids = [rng.uniform(0, 5, size=100) for _ in range(10)]
    lambdas = np.linspace(0, 10, 50)
    L = per_image_losses(resids, lambdas)
    assert np.all(np.diff(L, axis=1) <= 1e-12)


def test_zero_residual_needs_zero_lambda():
    resids = [np.zeros(50) for _ in range(20)]
    lambdas = np.array([0.0, 1.0, 2.0])
    assert crc_lambda(per_image_losses(resids, lambdas), lambdas, alpha=0.5) == 0.0
    r, off = _csr(resids)
    assert crc_lambda_exact(r, off, alpha=0.5) == 0.0


def test_infeasible_alpha_raises_instead_of_silent_fallback():
    resids = [np.ones(10) for _ in range(3)]
    lambdas = np.array([0.0, 0.5, 1.0])
    with pytest.raises(InfeasibleRiskError):
        crc_lambda(per_image_losses(resids, lambdas), lambdas, alpha=1e-9)
    r, off = _csr(resids)
    with pytest.raises(InfeasibleRiskError):
        crc_lambda_exact(r, off, alpha=1e-9)
    with pytest.warns(UserWarning):
        assert crc_lambda_or_inf(r, off, alpha=1e-9) == np.inf


def test_exact_matches_dense_grid():
    rng = np.random.default_rng(1)
    imgs = [np.abs(rng.normal(0, rng.uniform(0.5, 2), size=rng.integers(20, 80))) for _ in range(60)]
    r, off = _csr(imgs)
    for alpha in (0.05, 0.1, 0.2):
        exact = crc_lambda_exact(r, off, alpha)
        grid = calibrate(imgs, alpha, lambda_max=20, n_grid=20000)
        assert exact <= grid + 1e-12  # exact is the infimum; grid can only overshoot
        assert grid - exact < 5e-3 * max(exact, 1)
        # exact lambda satisfies the CRC inequality; anything just below does not
        L = lambda lam: np.mean([np.mean(x > lam) for x in imgs]) * len(imgs)
        assert (L(exact) + 1) / (len(imgs) + 1) <= alpha + 1e-12
        assert (L(exact - 1e-9) + 1) / (len(imgs) + 1) > alpha


def test_exact_handles_ties():
    imgs = [np.array([1.0, 1.0, 2.0, 2.0])] * 30
    r, off = _csr(imgs)
    lam = crc_lambda_exact(r, off, alpha=0.4)
    assert lam in (1.0, 2.0)
    assert (np.mean(imgs[0] > lam) * 30 + 1) / 31 <= 0.4


@pytest.mark.parametrize("n,alpha", [(4, 0.2), (9, 0.1), (19, 0.05)])
def test_zero_budget_returns_max_residual(n, alpha):
    """(n + 1) alpha - B == 0: only the largest residual is feasible, even with
    many pixels whose weights do not sum to exactly 1 in floating point."""
    rng = np.random.default_rng(n)
    imgs = [rng.exponential(1.0, rng.integers(5000, 30000)) for _ in range(n)]
    r, off = _csr(imgs)
    assert crc_lambda_exact(r, off, alpha) == r.max()
    units = np.repeat(np.arange(n), 3)
    r3, off3 = _csr([x for x in imgs for _ in range(3)])
    assert crc_lambda_exact_units(r3, off3, units, alpha) == r.max()


def test_rcps_is_more_conservative_than_crc():
    rng = np.random.default_rng(2)
    imgs = [np.abs(rng.normal(size=40)) for _ in range(500)]
    r, off = _csr(imgs)
    assert rcps_lambda_hoeffding(r, off, 0.1, delta=0.1) >= crc_lambda_exact(r, off, 0.1)


def test_crc_guarantee_holds_with_correlated_images():
    """E[uncovered fraction of a new image] <= alpha, with a per-image global
    shift (pixels within an image are NOT independent). One-sided check with
    a Monte Carlo standard-error tolerance."""
    rng = np.random.default_rng(42)
    alpha, reps, n_cal = 0.1, 400, 100
    losses = []
    for _ in range(reps):
        imgs = [np.abs(rng.normal(rng.normal(0, 1), 1, size=30)) for _ in range(n_cal + 1)]
        r, off = _csr(imgs[:-1])
        lam = crc_lambda_exact(r, off, alpha)
        losses.append(np.mean(imgs[-1] > lam))
    losses = np.array(losses)
    se = losses.std(ddof=1) / np.sqrt(reps)
    assert losses.mean() <= alpha + 3 * se
    assert losses.mean() >= alpha - 0.05  # and not absurdly conservative


def test_scene_crc_equals_image_crc_with_one_image_per_scene():
    rng = np.random.default_rng(3)
    r, off = _csr([rng.exponential(1.0, rng.integers(20, 60)) for _ in range(80)])
    units = np.arange(80)
    assert crc_lambda_exact_units(r, off, units, 0.1) == crc_lambda_exact(r, off, 0.1)


def test_scene_crc_valid_when_images_share_scene_error():
    """Scenes with a shared error scale and 1-12 images each: K1-scene must
    hold E[scene-averaged coverage] >= 1 - alpha on new scenes."""
    rng = np.random.default_rng(0)
    alpha, covs = 0.1, []
    for _ in range(300):
        def scenes(n):
            imgs, units = [], []
            for s in range(n):
                scale = np.exp(0.8 * rng.standard_normal())
                for _ in range(rng.integers(1, 13)):
                    imgs.append(np.abs(scale * rng.standard_normal(30)))
                    units.append(s)
            return imgs, np.array(units)
        cal, cal_units = scenes(40)
        test, test_units = scenes(40)
        r, off = _csr(cal)
        lam = crc_lambda_exact_units(r, off, cal_units, alpha)
        cov_img = np.array([np.mean(x <= lam) for x in test])
        per_scene = np.bincount(test_units, weights=cov_img) / np.bincount(test_units)
        covs.append(per_scene.mean())
    assert np.mean(covs) >= 1 - alpha - 0.005
