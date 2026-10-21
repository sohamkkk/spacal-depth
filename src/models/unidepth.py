"""Frozen UniDepthV2 (section 9.1, second base model; Piccinelli et al. 2025).

Trained on 23 datasets, none of them NYU / KITTI / DIODE. Runs without camera
intrinsics (fully monocular, same information as DA-V2). `infer` returns
"depth" (B,1,H,W) and "confidence" (B,1,H,W) at input resolution. Per the
authors, the confidence is an estimated scale-invariant log error: a
within-image ranking with no absolute meaning — i.e. a spatial *shape*, not
an image-level difficulty.
"""

import warnings

import numpy as np
import torch

from src.config import DEVICE
from src.models.base import DepthModel, freeze

CHECKPOINTS = {"small": "lpiccinelli/unidepth-v2-vits14", "large": "lpiccinelli/unidepth-v2-vitl14"}


class UniDepthV2Model(DepthModel):
    has_conf = True

    def __init__(self, size: str = "large", device: str | None = None):
        from unidepth.models import UniDepthV2

        self.device = device or DEVICE
        self.model_id = CHECKPOINTS[size]
        self.tag = f"unidepthv2-{size}"
        self.model = freeze(UniDepthV2.from_pretrained(self.model_id)).to(self.device)
        self.model.interpolation_mode = "bilinear"

    @torch.inference_mode()
    def predict(self, rgb: np.ndarray) -> dict[str, np.ndarray]:
        h, w = rgb.shape[:2]
        x = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).to(self.device)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = self.model.infer(x)
        depth = out["depth"][0, 0].float().cpu().numpy()
        conf = out["confidence"][0, 0].float().cpu().numpy()
        if depth.shape != (h, w) or conf.shape != (h, w):
            raise RuntimeError(f"UniDepth output {depth.shape}/{conf.shape} != input {(h, w)}")
        return {"depth": depth, "conf": conf}
