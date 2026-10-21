import numpy as np

from src.calibrators.anchors import (
    N_ROWS,
    NEAR_FRAC,
    anchor_keys,
    draw_anchors,
    group_conformal_quantile,
    group_loo_spread,
    group_median,
    local_halfwidth,
    nearest_anchor_distance,
    within_image_rank,
)
from src.calibrators.crc import crc_lambda_exact


def _groups(rng, sizes):
    g = np.repeat(np.arange(len(sizes)), sizes)
    return rng.normal(size=len(g)), g


def test_group_median_and_loo_spread_match_brute_force():
    rng = np.random.default_rng(0)
    sizes = [1, 2, 3, 4, 7, 10]
    x, g = _groups(rng, sizes)
    perm = rng.permutation(len(x))  # inputs need not be grouped
    med = group_median(x[perm], g[perm], len(sizes))
    spread = group_loo_spread(x[perm], g[perm], len(sizes))
    for i in range(len(sizes)):
        xi = x[g == i]
        assert np.isclose(med[i], np.median(xi))
        if len(xi) < 3:
            assert np.isnan(spread[i])
        else:
            loo = [abs(xi[j] - np.median(np.delete(xi, j))) for j in range(len(xi))]
            assert np.isclose(spread[i], np.median(loo))


def test_group_conformal_quantile_rank_and_infinity():
    scores = np.array([3.0, 1.0, 2.0] + list(range(1, 21)), dtype=float)
    g = np.array([0, 0, 0] + [1] * 20)
    q = group_conformal_quantile(scores, g, 2, alpha=0.1)
    assert np.isinf(q[0])                  # ceil(4 * 0.9) = 4 > 3 anchors
    assert q[1] == 19.0                    # ceil(21 * 0.9) = 19th smallest of 1..20


def test_within_image_rank_is_a_permutation_per_image():
    offsets = np.array([0, 5, 5 + 8, 5 + 8 + 3])
    rank = within_image_rank(offsets, np.random.default_rng(1))
    for i in range(3):
        assert sorted(rank[offsets[i]:offsets[i + 1]]) == list(range(offsets[i + 1] - offsets[i]))


def test_local_coverage_holds_per_image():
    """A-local is valid conditionally on the image: averaged over anchor draws,
    every image (here: very different, skewed, heteroscedastic) is covered >= 1 - alpha."""
    rng = np.random.default_rng(2)
    n_px, alpha, k = 4000, 0.1, 40
    images = [rng.normal(0.3, 0.05, n_px),                      # pure global scale error
              rng.standard_t(2, n_px) * 0.2,                    # heavy tails
              np.abs(rng.normal(0, 1, n_px)) ** 2 * 0.1 - 0.4]  # skewed, biased
    d = np.concatenate(images)
    offsets = np.arange(len(images) + 1) * n_px
    s = np.exp(rng.normal(0, 0.3, len(d)))                      # fixed, anchor-independent scale
    img = np.repeat(np.arange(len(images)), n_px)
    cov = np.zeros(len(images))
    n_draws = 400
    for _ in range(n_draws):
        draw = draw_anchors(d, offsets, k, rng)
        q = local_halfwidth(d, s, offsets, draw, alpha)
        hit = np.abs(d - draw.shift_half[img]) <= q[img] * s
        ev = draw.eval
        cov += np.bincount(img[ev], weights=hit[ev], minlength=len(images)) / np.bincount(img[ev])
    cov /= n_draws
    assert (cov >= 1 - alpha - 0.01).all(), cov


def test_draw_keeps_half_the_pixels_for_evaluation():
    d = np.zeros(10 + 300)
    offsets = np.array([0, 10, 310])
    draw = draw_anchors(d, offsets, 50, np.random.default_rng(3))
    assert list(draw.k_img) == [5, 50]
    assert draw.anchor[:10].sum() == 5 and draw.anchor[10:].sum() == 50
    assert not (draw.scale_half & draw.cal_half).any()
    assert (draw.scale_half | draw.cal_half).sum() == draw.anchor.sum()


def test_nearest_anchor_distance_matches_brute_force():
    rng = np.random.default_rng(6)
    pix, d, offsets = _grid_images(3, 10, 15, rng)
    anchor = rng.random(len(d)) < 0.05
    anchor[offsets[2]:] = False                      # image 2 has no anchors
    diag = np.full(3, np.hypot(10, 15))
    dist = nearest_anchor_distance(pix["row"], pix["col"], offsets, anchor, diag)
    for i in range(2):
        sl = slice(offsets[i], offsets[i + 1])
        r, c, a = pix["row"][sl].astype(float), pix["col"][sl].astype(float), anchor[sl]
        brute = np.hypot(r[:, None] - r[a][None], c[:, None] - c[a][None]).min(1) / diag[i]
        assert np.allclose(dist[sl], brute)
    assert np.isinf(dist[offsets[2]:]).all()


def _grid_images(n_img, h, w, rng):
    """n_img full h x w lattices with a smooth per-image error field (left-right ramp)."""
    rr, cc = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    pix = {"row": np.tile(rr.ravel(), n_img).astype(np.uint16), "col": np.tile(cc.ravel(), n_img).astype(np.uint16)}
    pix["pred"] = rng.uniform(1, 10, len(pix["row"]))
    pix["edge"] = rng.random(len(pix["row"])) < 0.2
    slope = rng.normal(0, 0.5, n_img)
    d = np.repeat(rng.normal(0, 0.3, n_img), h * w) + np.repeat(slope, h * w) * (pix["col"] / w - 0.5)
    d = d + rng.normal(0, 0.05, len(d))
    return pix, d, np.arange(n_img + 1) * h * w


def test_structured_patterns_place_anchors_as_specified():
    rng = np.random.default_rng(4)
    pix, d, offsets = _grid_images(5, 20, 30, rng)
    k = 60
    for pattern in ("rows", "cluster", "edges", "near"):
        draw = draw_anchors(d, offsets, k, rng, anchor_keys(pattern, pix, offsets, rng))
        assert list(draw.k_img) == [k] * 5
        assert (draw.scale_half.reshape(5, -1).sum(1) == k // 2).all()
        assert not (draw.scale_half & draw.cal_half).any() and ((draw.scale_half | draw.cal_half) == draw.anchor).all()
        for i in range(5):
            sl = slice(offsets[i], offsets[i + 1])
            a = draw.anchor[sl]
            if pattern == "rows":
                assert len(np.unique(pix["row"][sl][a])) <= N_ROWS
            elif pattern == "edges":
                assert pix["edge"][sl][a].all()
            elif pattern == "near":
                cut = np.sort(pix["pred"][sl])[int(NEAR_FRAC * a.size) - 1]
                assert (pix["pred"][sl][a] <= cut).all()
            else:
                r, c = pix["row"][sl].astype(float), pix["col"][sl].astype(float)
                # anchors form a disc: some centre pixel is closer to every anchor than to any non-anchor
                dist = np.hypot(r[:, None] - r[None], c[:, None] - c[None])
                assert (dist[:, a].max(1) <= dist[:, ~a].min(1) + 1e-9).any()


def test_rescale_crc_stays_valid_and_local_fails_under_clustered_anchors():
    """Images are exchangeable under any per-image anchor rule, so cross-image CRC on the
    anchor-shifted score keeps E[miscoverage] <= alpha; A-local loses its per-image guarantee
    when anchors are spatially clustered and the error has a spatial trend."""
    rng = np.random.default_rng(5)
    alpha, k, h, w, n_img, reps = 0.1, 40, 12, 20, 40, 300
    miss_crc, miss_local = [], []
    for _ in range(reps):
        pix, d, offsets = _grid_images(n_img, h, w, rng)
        draw = draw_anchors(d, offsets, k, rng, anchor_keys("cluster", pix, offsets, rng))
        img = np.repeat(np.arange(n_img), h * w)
        r = np.abs(d - draw.shift[img])
        ev = draw.eval
        cal_img = np.arange(n_img) < n_img // 2
        m_cal = ev & cal_img[img]
        off_cal = np.r_[0, np.cumsum(np.bincount(img[m_cal], minlength=n_img)[cal_img])]
        lam = crc_lambda_exact(r[m_cal], off_cal, alpha)
        m_test = ev & ~cal_img[img]
        per_img = np.bincount(img[m_test], weights=r[m_test] > lam, minlength=n_img)[~cal_img] / (h * w - k)
        miss_crc.append(per_img.mean())
        q = local_halfwidth(d, np.ones_like(d), offsets, draw, alpha)
        hit = np.abs(d - draw.shift_half[img]) <= q[img]
        miss_local.append(1 - hit[ev].mean())
    miss_crc, miss_local = np.array(miss_crc), np.array(miss_local)
    assert miss_crc.mean() <= alpha + 3 * miss_crc.std() / np.sqrt(reps)
    assert miss_local.mean() > alpha + 3 * miss_local.std() / np.sqrt(reps)
