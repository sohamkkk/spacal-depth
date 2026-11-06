import numpy as np
import pytest

torch = pytest.importorskip("torch")
m3d = pytest.importorskip("src.models.metric3d")


class FakeHub(torch.nn.Module):
    """Canonical depth 100 everywhere (metric depth = focal seen / 10, below MAX_DEPTH), confidence 0.75."""

    def inference(self, data):
        x = data["input"]
        assert tuple(x.shape[-2:]) == m3d.INPUT_SIZE
        ones = torch.ones(x.shape[0], 1, *x.shape[-2:])
        return 100.0 * ones, 0.75 * ones, {}


def test_metric_scale_and_shapes(monkeypatch):
    monkeypatch.setattr(m3d, "load_hub_model", lambda entry: FakeHub())
    model = m3d.Metric3DModel("large", device="cpu")
    rgb = np.zeros((352, 1216, 3), np.uint8)
    out = model.predict(rgb, focal=721.5)
    scale = min(m3d.INPUT_SIZE[0] / 352, m3d.INPUT_SIZE[1] / 1216)
    assert out["depth"].shape == (352, 1216) and out["conf"].shape == (352, 1216)
    np.testing.assert_allclose(out["depth"], 721.5 * scale / 10, rtol=1e-5)
    np.testing.assert_allclose(out["conf"], 0.25, rtol=1e-5)
    flip = model.predict_flip(rgb, focal=721.5)
    np.testing.assert_allclose(flip["depth"], out["depth"])
    with pytest.raises(ValueError):
        model.predict(rgb)
