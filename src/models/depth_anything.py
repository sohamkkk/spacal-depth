"""Frozen Depth Anything V2 metric checkpoints (section 9.1, primary base model).

Indoor checkpoints were fine-tuned on Hypersim, outdoor on Virtual KITTI 2,
so NYU / KITTI / DIODE are fully unseen. Needs transformers >= 4.45 (metric
head). The raw network output is at the processor resolution (short side 518,
multiple of 14); we upsample with bilinear, align_corners=True, matching the
official DA-V2 `infer_image`.

`gradient_uncertainty` is a GrUMoDepth-style re-implementation (Hornauer &
Belagiannis, ECCV 2022): gradients of a label-free flip-consistency loss
w.r.t. the decoder (neck fusion) feature maps, channel-averaged |grad|,
upsampled and summed over fusion layers.
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from src.config import DEVICE
from src.models.base import DepthModel, freeze

CHECKPOINTS = {
    (domain, size): f"depth-anything/Depth-Anything-V2-Metric-{domain.capitalize()}-{size.capitalize()}-hf"
    for domain in ("indoor", "outdoor")
    for size in ("small", "base", "large")
}


class DepthAnythingV2Metric(DepthModel):
    has_grad_uncertainty = True

    def __init__(self, domain: str = "indoor", size: str = "large", device: str | None = None):
        if (domain, size) not in CHECKPOINTS:
            raise ValueError(f"(domain, size) must be one of {list(CHECKPOINTS)}")
        self.device = device or DEVICE
        self.model_id = CHECKPOINTS[(domain, size)]
        self.tag = f"dav2-{domain}-{size}"
        self.processor = AutoImageProcessor.from_pretrained(self.model_id)
        self.model = freeze(AutoModelForDepthEstimation.from_pretrained(self.model_id)).to(self.device)
        if getattr(self.model.config, "depth_estimation_type", None) != "metric":
            raise RuntimeError(f"{self.model_id} did not load a metric head; upgrade transformers to >= 4.45")

    def _pixel_values(self, rgb: np.ndarray) -> torch.Tensor:
        inputs = self.processor(images=Image.fromarray(rgb), return_tensors="pt")
        return inputs["pixel_values"].to(self.device)

    @staticmethod
    def _upsample(d: torch.Tensor, h: int, w: int) -> torch.Tensor:
        return F.interpolate(d[:, None], size=(h, w), mode="bilinear", align_corners=True)[:, 0]

    @torch.inference_mode()
    def predict(self, rgb: np.ndarray) -> dict[str, np.ndarray]:
        h, w = rgb.shape[:2]
        d = self.model(pixel_values=self._pixel_values(rgb)).predicted_depth
        depth = self._upsample(d, h, w)[0].float().cpu().numpy()
        if depth.shape != (h, w):
            raise RuntimeError(f"prediction shape {depth.shape} != input {(h, w)}")
        return {"depth": depth}

    @torch.autocast("cuda", enabled=False)  # fp16 would overflow the summed squared loss / its gradients
    def gradient_uncertainty(self, rgb: np.ndarray) -> np.ndarray:
        h, w = rgb.shape[:2]
        pv = self._pixel_values(rgb).float()
        with torch.no_grad():
            d_flip = self.model(pixel_values=torch.flip(pv, dims=[-1])).predicted_depth.flip(-1)
        fused: list[torch.Tensor] = []

        def keep(_module, _inputs, outputs):
            for t in outputs:
                t.retain_grad()
            fused.extend(outputs)

        handle = self.model.neck.register_forward_hook(keep)
        try:
            with torch.enable_grad():
                pv = pv.clone().requires_grad_(True)
                d = self.model(pixel_values=pv).predicted_depth
                ((d - d_flip) ** 2).sum().backward()
        finally:
            handle.remove()
        if not fused or any(t.grad is None for t in fused):
            raise RuntimeError("could not capture neck feature gradients")
        unc = sum(self._upsample(t.grad.abs().mean(1), h, w) for t in fused)
        return unc[0].float().cpu().numpy()
