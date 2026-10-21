import numpy as np

from src.calibrators.anchor_dist import (DIST_EDGES, KINDS, DistModel, FieldConfig, anchor_fields, calibrate,
                                         interval, merge_sparse_bins, min_units_for, select_model)
from src.calibrators.anchors import anchor_keys, draw_anchors, draw_split_anchors, local_halfwidth, rank_by_key
from src.calibrators.crc import crc_lambda_exact_units
from src.data.prepared import PixelTable
from src.data.splits import exchange_units


def _images(rng, n, h=32, w=48, smooth=0.0):
    """Log residual d = global error + smooth spatial field (amplitude `smooth`) + pixel noise."""
    rr, cc = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    y = np.exp(rng.uniform(0, 2, (n, h, w)))
    phase = rng.uniform(0, 2 * np.pi, (n, 1, 1))
    fr, fc = rng.uniform(0.5, 1.5, (2, n, 1, 1))
    field = smooth * rng.uniform(0.5, 1.0, (n, 1, 1)) * np.sin(2 * np.pi * (fr * rr / h + fc * cc / w) + phase)
    d = rng.normal(0, 0.2, (n, 1, 1)) + field + rng.normal(0, 0.05, (n, h, w))
    pix = {"gt": y.ravel().astype(np.float32), "pred": (y * np.exp(-d)).ravel().astype(np.float32),
           "row": np.tile(rr.ravel(), n).astype(np.uint16), "col": np.tile(cc.ravel(), n).astype(np.uint16)}
    img = {"height": np.full(n, h), "width": np.full(n, w), "scene_id": np.arange(n).astype(str)}
    t = PixelTable(pix=pix, img=img, offsets=np.arange(n + 1) * h * w)
    return t, np.log(t.pix["gt"].astype(float)) - np.log(t.pix["pred"].astype(float))


def test_median_global_single_bin_equals_a_hybrid():
    """M0 (image median shift) + the image's LOO spread + one bin + no g is exactly A-hybrid."""
    rng = np.random.default_rng(0)
    (cal, d_c), (test, d_t) = _images(rng, 40), _images(rng, 30)
    dc, dt = draw_anchors(d_c, cal.offsets, 9, rng), draw_anchors(d_t, test.offsets, 9, rng)
    alpha, tau = 0.1, 0.03
    model = DistModel(FieldConfig(kind="median", spread="global"), tau, sigma_fill=1.0, edges=())
    lams = calibrate(model, cal, d_c, np.ones(cal.n_pixels), dc, alpha, units=exchange_units(cal), min_pixels=1)
    ft = anchor_fields(test, d_t, dt, model.cfg)
    lo, hi = interval(model, test, np.ones(test.n_pixels), lams, ft)

    img_c, img_t = cal.image_of_pixel(), test.image_of_pixel()
    ev = dc.eval
    r = (np.abs(d_c - dc.shift[img_c]) / (dc.sigma[img_c] + tau))[ev]
    off = np.concatenate([[0], np.cumsum(np.bincount(img_c[ev], minlength=cal.n_images))])
    lam = crc_lambda_exact_units(r, off, exchange_units(cal), alpha)
    q = lam * (dt.sigma[img_t] + tau)
    pred = test.pix["pred"].astype(float)
    assert np.isclose(lams[0], lam)
    assert np.allclose(lo, pred * np.exp(dt.shift[img_t] - q)) and np.allclose(hi, pred * np.exp(dt.shift[img_t] + q))


def _bin_coverage(t, y, lo, hi, delta, ev, min_px=10):
    """Per fixed distance bin: mean over images (with >= min_px evaluation pixels there) of the covered share."""
    nb = len(DIST_EDGES) + 1
    key = t.image_of_pixel()[ev] * nb + np.digitize(delta[ev], DIST_EDGES)
    cnt = np.bincount(key, minlength=t.n_images * nb).reshape(-1, nb)
    hit = np.bincount(key, weights=((y >= lo) & (y <= hi))[ev], minlength=t.n_images * nb).reshape(-1, nb)
    ok = cnt >= min_px
    return np.array([(hit[ok[:, b], b] / cnt[ok[:, b], b]).mean() if ok[:, b].any() else np.nan for b in range(nb)])


def test_a_dist_keeps_per_bin_coverage_under_clustered_anchors_while_a_local_fails():
    rng = np.random.default_rng(1)
    alpha, k, reps = 0.1, 20, 30
    per_bin, local_cov = [], []
    for _ in range(reps):
        (head, d_h), (cal, d_c), (test, d_t) = (_images(rng, n, smooth=0.4) for n in (60, 60, 60))
        draws = [draw_anchors(d, t.offsets, k, rng, anchor_keys("cluster", t.pix, t.offsets, rng))
                 for t, d in ((head, d_h), (cal, d_c), (test, d_t))]
        half_a = rng.random(head.n_images) < 0.5
        model = select_model(head, d_h, np.ones(head.n_pixels), draws[0], "kernel", alpha, half_a)
        model.edges = DIST_EDGES                         # fixed bins so reps are comparable
        lams = calibrate(model, cal, d_c, np.ones(cal.n_pixels), draws[1], alpha, units=exchange_units(cal))
        ft = anchor_fields(test, d_t, draws[2], model.cfg)
        lo, hi = interval(model, test, np.ones(test.n_pixels), lams, ft)
        y = test.pix["gt"].astype(float)
        per_bin.append(_bin_coverage(test, y, lo, hi, ft.delta, draws[2].eval))

        q = local_halfwidth(d_t, np.ones(test.n_pixels), test.offsets, draws[2], alpha)[test.image_of_pixel()]
        c = draws[2].shift_half[test.image_of_pixel()]
        pred = test.pix["pred"].astype(float)
        cov = (y >= pred * np.exp(c - q)) & (y <= pred * np.exp(c + q))
        local_cov.append(cov[draws[2].eval].mean())
    per_bin, local_cov = np.array(per_bin), np.array(local_cov)
    for b in range(per_bin.shape[1]):
        v = per_bin[:, b][np.isfinite(per_bin[:, b])]
        if len(v) >= reps // 2:
            assert v.mean() >= 1 - alpha - 3 * v.std() / np.sqrt(len(v)), (b, v.mean())
    assert local_cov.mean() < 1 - alpha - 3 * local_cov.std() / np.sqrt(reps)


def test_sparse_far_bins_merge_into_nearer_neighbour():
    """Scenes 0-29 have pixels in bins 0-1, only scenes 0-4 reach bins 2-3 and none bin 4: bins 2-4 must
    join bin 1, and nothing merges when every bin is dense."""
    n_img, px = 30, 40
    img = np.repeat(np.arange(n_img), px)
    bins = np.tile(np.repeat([0, 1], px // 2), n_img)
    bins[(img < 5) & (np.arange(len(img)) % px >= 30)] = 2
    bins[(img < 5) & (np.arange(len(img)) % px >= 35)] = 3
    units = np.arange(n_img)
    assert min_units_for(0.1) == 19
    assert list(merge_sparse_bins(bins, img, units, 5, min_pixels=5, min_units=19)) == [0, 1, 1, 1, 1]
    dense = np.tile(np.repeat(np.arange(5), px // 5), n_img)
    assert list(merge_sparse_bins(dense, img, units, 5, min_pixels=5, min_units=19)) == [0, 1, 2, 3, 4]
    # two images per scene: 15 scenes < 19 in every bin, so everything ends in one group
    assert len(set(merge_sparse_bins(dense, img, units // 2, 5, min_pixels=5, min_units=19))) == 1


def test_calibrate_gives_finite_lambdas_with_few_scenes_far_from_anchors():
    rng = np.random.default_rng(3)
    cal, d_c = _images(rng, 40, smooth=0.4)
    dc = draw_anchors(d_c, cal.offsets, 20, rng, anchor_keys("cluster", cal.pix, cal.offsets, rng))
    model = DistModel(FieldConfig(kind="knn"), tau=0.05, sigma_fill=0.1)
    u = exchange_units(cal)
    f = anchor_fields(cal, d_c, dc, model.cfg)
    # last edge so that exactly 5 scenes have >= 10 evaluation pixels beyond it: 5 < 1/alpha - 1 is infeasible
    sl = [slice(cal.offsets[i], cal.offsets[i + 1]) for i in range(cal.n_images)]
    tenth = np.sort([np.sort(f.delta[s][dc.eval[s]])[-10] for s in sl])
    model.edges = (*DIST_EDGES, float(tenth[-6] + tenth[-5]) / 2)
    assert np.isinf(calibrate(model, cal, d_c, np.ones(cal.n_pixels), dc, 0.1, units=u, fields=f,
                              min_units=1)).any()
    lams = calibrate(model, cal, d_c, np.ones(cal.n_pixels), dc, 0.1, units=u, fields=f)
    assert np.isfinite(lams).all()


def test_auto_picks_the_kind_with_the_best_head_score():
    rng = np.random.default_rng(4)
    head, d_h = _images(rng, 40, smooth=0.4)
    dh = draw_anchors(d_h, head.offsets, 20, rng)
    half_a = rng.random(head.n_images) < 0.5
    per_kind = {k: select_model(head, d_h, np.ones(head.n_pixels), dh, k, 0.1, half_a) for k in KINDS}
    auto = select_model(head, d_h, np.ones(head.n_pixels), dh, "auto", 0.1, half_a)
    best = min(per_kind, key=lambda k: per_kind[k].head_score)
    assert auto.cfg == per_kind[best].cfg and auto.head_score == per_kind[best].head_score


def test_split_draw_structure_and_a_local_guarantee_under_clustered_anchors():
    """k - k_r anchors follow the rule; the A-local halves come from the k_r random ones, so A-local keeps
    its within-image coverage (theory: 10/11 for 10 calibration anchors) even with clustered anchors."""
    rng = np.random.default_rng(2)
    alpha, k, k_r, reps = 0.1, 40, 20, 30
    covs = []
    for rep in range(reps):
        test, d = _images(rng, 40, smooth=0.4)
        key = anchor_keys("cluster", test.pix, test.offsets, rng)
        dr = draw_split_anchors(d, test.offsets, k, k_r, rng, key)
        img = test.image_of_pixel()
        rand = dr.scale_half | dr.cal_half
        assert not (dr.scale_half & dr.cal_half).any()
        assert (np.bincount(img[dr.anchor], minlength=test.n_images) == k).all()
        assert (np.bincount(img[rand], minlength=test.n_images) == k_r).all()
        assert (np.bincount(img[dr.scale_half], minlength=test.n_images) == k_r // 2).all()
        if rep == 0:
            assert (dr.anchor & ~rand == (rank_by_key(test.offsets, key) < k - k_r)).all()
        q = local_halfwidth(d, np.ones(test.n_pixels), test.offsets, dr, alpha)[img]
        c = dr.shift_half[img]
        y, pred = test.pix["gt"].astype(float), test.pix["pred"].astype(float)
        cov = (y >= pred * np.exp(c - q)) & (y <= pred * np.exp(c + q))
        covs.append(cov[dr.eval].mean())
    covs = np.array(covs)
    assert covs.mean() >= 10 / 11 - 3 * covs.std() / np.sqrt(reps)
