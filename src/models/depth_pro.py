"""Frozen Depth Pro (section 9.1, third base model; Bochkovskii et al., ICLR 2025,
arXiv 2410.02073), via Hugging Face transformers (`apple/DepthPro-hf`).

NYUv2 and KITTI are evaluation-only in the paper's dataset table (Tab. 15);
DIODE is not listed, so all three of our datasets are unseen. Runs without
camera intrinsics: the focal length comes from its own field-of-view head,
the same information DA-V2 and UniDepthV2 get.

The network outputs a canonical inverse depth at 1536 x 1536;
`post_process_depth_estimation` turns it into metric depth with the predicted
FOV and resizes to the target size. Without the FOV head that function still
returns an inverted map, but at the wrong scale, so a missing FOV is an error.
Needs transformers >= 4.49.0 (first release with DepthPro).
"""

import numpy as np
import torch
from PIL import Image

from src.config import DEVICE
from src.models.base import DepthModel, freeze

CHECKPOINT = "apple/DepthPro-hf"


class DepthProModel(DepthModel):
    def __init__(self, device: str | None = None):
        try:
            from transformers import AutoImageProcessor, DepthProForDepthEstimation
        except ImportError as e:
            raise ImportError("this transformers release has no DepthPro; upgrade transformers") from e
        self.device = device or DEVICE
        self.model_id = CHECKPOINT
        self.tag = "depthpro"
        self.processor = AutoImageProcessor.from_pretrained(CHECKPOINT)
        self.model = freeze(DepthProForDepthEstimation.from_pretrained(CHECKPOINT, use_fov_model=True)).to(self.device)

    @torch.inference_mode()
    def predict(self, rgb: np.ndarray) -> dict[str, np.ndarray]:
        h, w = rgb.shape[:2]
        inputs = self.processor(images=Image.fromarray(rgb), return_tensors="pt").to(self.device)
        out = self.model(**inputs)
        if out.field_of_view is None:
            raise RuntimeError("Depth Pro returned no field of view; metric depth would be mis-scaled")
        post = self.processor.post_process_depth_estimation(out, target_sizes=[(h, w)])[0]
        depth = post["predicted_depth"].float().cpu().numpy()
        if depth.shape != (h, w):
            raise RuntimeError(f"prediction shape {depth.shape} != input {(h, w)}")
        return {"depth": depth}

    def info(self) -> dict:
        return {**super().info(), "use_fov_model": True}
