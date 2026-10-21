import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.metrics.depth_metrics import abs_rel, rmse, delta_k, silog, all_metrics


def test_perfect_prediction_is_zero_error():
    gt = np.array([1.0, 2.0, 3.0, 4.0])
    pred = gt.copy()
    assert abs_rel(pred, gt) == 0.0
    assert rmse(pred, gt) == 0.0
    assert delta_k(pred, gt, 1) == 1.0
    assert abs(silog(pred, gt)) < 1e-9


def test_abs_rel_known_value():
    gt = np.array([10.0, 10.0])
    pred = np.array([11.0, 9.0])  # 10% off both directions
    assert abs(abs_rel(pred, gt) - 0.1) < 1e-9


def test_mask_excludes_invalid_pixels():
    gt = np.array([10.0, -1.0, 10.0])  # -1.0 is an invalid/hole pixel
    pred = np.array([10.0, 999.0, 10.0])  # garbage prediction at the hole
    mask = np.array([True, False, True])
    assert abs_rel(pred, gt, mask) == 0.0  # garbage pixel must not leak in


def test_all_metrics_keys():
    gt = np.full(5, 2.0)
    pred = np.full(5, 2.0)
    m = all_metrics(pred, gt)
    assert set(m.keys()) == {"AbsRel", "RMSE", "delta1", "delta2", "delta3", "SILog"}


if __name__ == "__main__":
    test_perfect_prediction_is_zero_error()
    test_abs_rel_known_value()
    test_mask_excludes_invalid_pixels()
    test_all_metrics_keys()
    print("metrics: all checks passed")
