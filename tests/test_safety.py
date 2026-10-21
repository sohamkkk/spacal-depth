import numpy as np

from src.calibrators.crc import crc_lambda_exact
from src.calibrators.split_cp import conformal_quantile
from src.data.prepared import PixelTable
from src.metrics.safety import (danger_scores, lane_event_scores, lane_ids, region_minima, roi_mask,
                                safety_metrics)


def _table(gt_list, pred_list, h=4, w=6):
    """Images on a full h x w grid; gt/pred given per image as (h, w) arrays."""
    rr, cc = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    n = len(gt_list)
    pix = {
        "gt": np.concatenate([g.ravel() for g in gt_list]).astype(np.float32),
        "pred": np.concatenate([p.ravel() for p in pred_list]).astype(np.float32),
        "row": np.tile(rr.ravel(), n).astype(np.uint16),
        "col": np.tile(cc.ravel(), n).astype(np.uint16),
    }
    img = {"height": np.full(n, h), "width": np.full(n, w), "scene_id": np.arange(n).astype(str)}
    return PixelTable(pix=pix, img=img, offsets=np.arange(n + 1) * h * w)


def test_roi_lanes_and_minima():
    gt = np.arange(24, dtype=float).reshape(4, 6) + 1
    t = _table([gt], [gt])
    roi = (0.5, 1.0, 0.0, 1.0)                       # bottom two rows
    m = roi_mask(t, roi)
    assert m.reshape(4, 6)[2:].all() and not m.reshape(4, 6)[:2].any()
    v, keep = t.select_pixels(m)
    assert keep.all() and v.n_pixels == 12
    lanes = lane_ids(v, roi, 3)                      # columns {0,1}, {2,3}, {4,5}
    assert list(lanes[:6]) == [0, 0, 1, 1, 2, 2]
    whole, per_lane = region_minima(v, v.pix["gt"].astype(float), lanes, 3)
    assert whole[0] == 13 and list(per_lane[0]) == [13, 15, 17]


def test_select_pixels_drops_sparse_images():
    t = _table([np.ones((4, 6))] * 2, [np.ones((4, 6))] * 2)
    mask = np.zeros(t.n_pixels, bool)
    mask[:10] = True                                 # 10 px of image 0, none of image 1
    v, keep = t.select_pixels(mask, min_count=5)
    assert list(keep) == [True, False] and v.n_images == 1 and v.n_pixels == 10


def test_roi_event_bound_is_simultaneous_and_k1_lower_is_not():
    """Split CP on the per-image max one-sided score covers every ROI pixel of a new image
    with prob >= 1 - alpha; the pixel-average CRC bound does not control that event."""
    rng = np.random.default_rng(0)
    alpha, n_cal, n_test, reps, h, w = 0.1, 60, 60, 300, 4, 6
    any_event, any_k1 = [], []
    for _ in range(reps):
        def images(n):
            g = rng.normal(0, 0.2, n)[:, None, None]
            d = g + rng.normal(0, 0.1, (n, h, w))    # log y - log yhat
            y = np.exp(rng.uniform(0, 2, (n, h, w)))
            return _table(list(y), list(y * np.exp(-d)))
        cal, test = images(n_cal), images(n_test)
        rp = np.log(cal.pix["pred"].astype(float)) - np.log(cal.pix["gt"].astype(float))
        roi_max = np.maximum.reduceat(rp, cal.offsets[:-1])
        lam_event = conformal_quantile(roi_max, alpha)
        lam_k1 = crc_lambda_exact(rp, cal.offsets, alpha)
        lanes = lane_ids(test, (0, 1, 0, 1), 3)
        pred = test.pix["pred"].astype(float)
        for lam, store in ((lam_event, any_event), (lam_k1, any_k1)):
            lower = pred * np.exp(-lam)
            b_img, b_lane = region_minima(test, lower, lanes, 3)
            store.append(safety_metrics(test, b_img, b_lane, lanes, 3, (), lower)["viol_any_px"])
    any_event, any_k1 = np.array(any_event), np.array(any_k1)
    assert any_event.mean() <= alpha + 3 * any_event.std() / np.sqrt(reps)
    assert any_k1.mean() > alpha + 3 * any_k1.std() / np.sqrt(reps)


def test_scene_averaged_violation_rates():
    """Image 0 of scene a is violated; scenes {a: 2 images, b: 1}: image rate 1/3, scene rate (1/2 + 0)/2."""
    gt = [np.full((4, 6), 2.0)] * 3
    t = _table(gt, gt)
    lanes = lane_ids(t, (0, 1, 0, 1), 3)
    lower = t.pix["gt"].astype(float) * 0.5
    lower[: 24] *= 3                                 # image 0 bound above the truth everywhere
    b_img, b_lane = region_minima(t, lower, lanes, 3)
    m = safety_metrics(t, b_img, b_lane, lanes, 3, (), lower, units=np.array(["a", "a", "b"]))
    for k in ("viol_roi", "viol_lane", "viol_any_px"):
        assert np.isclose(m[k], 1 / 3) and np.isclose(m[f"{k}_scene"], 0.25)


def test_lane_event_covers_every_lane_and_task_scalar_does_not():
    """Split CP on the worst lane's log ratio controls P(any lane bound violated); the whole-ROI
    task-scalar factor does not when lanes err independently."""
    rng = np.random.default_rng(2)
    alpha, n_cal, n_test, reps, h, w = 0.1, 80, 80, 300, 4, 6
    lane_rate, task_rate = [], []
    for _ in range(reps):
        def images(n):
            d = rng.normal(0, 0.1, (n, 1, 1)) + np.repeat(rng.normal(0, 0.2, (n, 1, 3)), 2, axis=2)
            y = np.exp(rng.uniform(0, 2, (n, h, w)))
            return _table(list(y), list(y * np.exp(-d)))
        cal, test = images(n_cal), images(n_test)
        lanes_c, lanes_t = lane_ids(cal, (0, 1, 0, 1), 3), lane_ids(test, (0, 1, 0, 1), 3)
        pred_c = cal.pix["pred"].astype(float)
        o_hat, _ = region_minima(cal, pred_c, lanes_c, 3)
        o, _ = region_minima(cal, cal.pix["gt"].astype(float), lanes_c, 3)
        b_img, b_lane = region_minima(test, test.pix["pred"].astype(float), lanes_t, 3)
        for scores, store in ((lane_event_scores(cal, pred_c, lanes_c, 3), lane_rate), (np.log(o_hat) - np.log(o), task_rate)):
            f = np.exp(-max(conformal_quantile(scores, alpha), 0))
            store.append(safety_metrics(test, b_img * f, b_lane * f, lanes_t, 3, ())["viol_lane"])
    lane_rate, task_rate = np.array(lane_rate), np.array(task_rate)
    assert lane_rate.mean() <= alpha + 3 * lane_rate.std() / np.sqrt(reps)
    assert task_rate.mean() > alpha + 3 * task_rate.std() / np.sqrt(reps)


def test_danger_crc_bounds_the_conditional_miss_share():
    """CRC on danger_scores keeps E[missed share of dangerous pixels | image has danger] <= alpha,
    while the point estimate (and the all-pixel CRC bound) leave it far above alpha."""
    rng = np.random.default_rng(1)
    alpha, D, n_cal, n_test, reps, h, w = 0.1, 1.5, 80, 80, 200, 4, 6
    miss_dcrc, miss_point, miss_k1 = [], [], []
    for _ in range(reps):
        def images(n):
            near = rng.random(n) < 0.4                # only some images contain anything near
            y = np.exp(rng.uniform(0.5, 2.0, (n, h, w)))
            y[near, 0, :2] = rng.uniform(1.0, 1.45, (near.sum(), 2))
            # near surfaces are over-predicted, so a pixel-average bound can leave them past D
            d = rng.normal(0, 0.15, (n, 1, 1)) + rng.normal(0, 0.1, (n, h, w)) - 0.4 * (y < D)
            return _table(list(y), list(y * np.exp(-d)))
        cal, test = images(n_cal), images(n_test)
        r, off, has = danger_scores(cal, np.ones(cal.n_pixels), D)
        assert has.sum() == len(off) - 1 and (np.diff(off) > 0).all()
        rp = np.log(cal.pix["pred"].astype(float)) - np.log(cal.pix["gt"].astype(float))
        lanes = lane_ids(test, (0, 1, 0, 1), 3)
        pred = test.pix["pred"].astype(float)
        for lam, store in ((max(crc_lambda_exact(r, off, alpha), 0), miss_dcrc), (0.0, miss_point),
                           (max(crc_lambda_exact(rp, cal.offsets, alpha), 0), miss_k1)):
            lower = pred * np.exp(-lam)
            b_img, b_lane = region_minima(test, lower, lanes, 3)
            store.append(safety_metrics(test, b_img, b_lane, lanes, 3, (D,), lower)[f"px_cond_miss_img@{D:g}"])
    miss_dcrc, miss_point, miss_k1 = map(np.array, (miss_dcrc, miss_point, miss_k1))
    assert miss_dcrc.mean() <= alpha + 3 * miss_dcrc.std() / np.sqrt(reps)
    for other in (miss_point, miss_k1):
        assert other.mean() > alpha + 3 * other.std() / np.sqrt(reps)
