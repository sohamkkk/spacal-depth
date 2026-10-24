"""Common interface for frozen metric depth models (section 9.1).

`predict(rgb)` returns {"depth": (H, W) float32 metres, optional "conf": (H, W)}
at the input resolution; models with `needs_focal` take `predict(rgb, focal=...)`
(pixels, for the image as given). Flip test-time augmentation is model-agnostic
and lives here; models may add `gradient_uncertainty`.
"""

import time
from contextlib import contextmanager

import numpy as np
import torch


class DepthModel:
    tag: str = ""
    model_id: str = ""
    has_conf: bool = False
    has_grad_uncertainty: bool = False
    needs_focal: bool = False

    def predict(self, rgb: np.ndarray) -> dict[str, np.ndarray]:
        raise NotImplementedError

    def predict_flip(self, rgb: np.ndarray, **kw) -> dict[str, np.ndarray]:
        out = self.predict(np.ascontiguousarray(rgb[:, ::-1]), **kw)
        return {k: np.ascontiguousarray(v[:, ::-1]) for k, v in out.items()}

    def gradient_uncertainty(self, rgb: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def info(self) -> dict:
        return {"tag": self.tag, "model_id": self.model_id, "has_conf": self.has_conf}


@contextmanager
def cuda_timer(device: str, store: dict, key: str):
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    yield
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    store[key] = time.perf_counter() - t0


def freeze(module: torch.nn.Module) -> torch.nn.Module:
    module.eval()
    for p in module.parameters():
        p.requires_grad_(False)
    return module
