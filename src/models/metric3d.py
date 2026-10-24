"""Frozen Metric3D v2 (fourth base model; Hu et al., TPAMI 2024, arXiv 2404.15506 `[VERIFY: venue]`),
via torch.hub (`yvanyin/metric3d`, entry `metric3d_vit_{small,large,giant2}`).

Unlike the other three models it needs the camera's focal length: the network predicts depth for a
canonical camera with focal 1000 px, and metric depth is that times focal_used / 1000, where focal_used
is the focal length of the image the network actually saw (after the keep-ratio resize). Preprocessing
follows the repo's hubconf.py: keep-ratio resize to fit (616, 1064), pad with the ImageNet mean,
normalise, un-pad, bilinear upsample to the input size, clamp to [0, 300] m.

`confidence` comes from the RAFT depth head, trained towards 1 - |relative error| `[VERIFY]`; we
store conf = max(1 - confidence, 0), an error-like map like UniDepthV2's (higher = less reliable).
Its training data `[VERIFY]` exclude NYU, KITTI, DIODE and ETH3D, which the paper uses for zero-shot
evaluation. Needs mmengine (mmcv is stubbed when absent) and the repo's own `mono` package, fetched
by torch.hub.
"""

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.config import DEVICE
from src.models.base import DepthModel, freeze

HUB_REPO = "yvanyin/metric3d"
ENTRIES = {"small": "metric3d_vit_small", "large": "metric3d_vit_large", "giant2": "metric3d_vit_giant2"}
INPUT_SIZE = (616, 1064)
MEAN = (123.675, 116.28, 103.53)
STD = (58.395, 57.12, 57.375)
CANONICAL_FOCAL = 1000.0
MAX_DEPTH = 300.0


def install_mmcv_stub() -> None:
    """The repo's mono/utils/comm.py hard-imports mmcv.utils.collect_env (unused at inference), and
    mmcv has no wheels for recent torch / Python. Register a minimal mmcv.utils when mmcv is missing;
    every other mmcv import in the repo (Config, DictAction, get_git_hash) is in a try/except that
    falls back to mmengine when the name is absent from the stub."""
    if "mmcv" in sys.modules or importlib.util.find_spec("mmcv") is not None:
        return
    mmcv = types.ModuleType("mmcv")
    utils = types.ModuleType("mmcv.utils")
    utils.collect_env = lambda: {}
    mmcv.utils = utils
    sys.modules["mmcv"] = mmcv
    sys.modules["mmcv.utils"] = utils


install_mmcv_stub()


def local_repo() -> Path:
    """Where torch.hub keeps the downloaded repo (under $TORCH_HOME/hub)."""
    return Path(torch.hub.get_dir()) / "yvanyin_metric3d_main"


def load_hub_model(entry: str) -> torch.nn.Module:
    """From the local hub cache when present (compute nodes have no internet), else from GitHub."""
    repo = local_repo()
    if repo.is_dir():
        return torch.hub.load(str(repo), entry, source="local", pretrain=True)
    return torch.hub.load(HUB_REPO, entry, pretrain=True, trust_repo=True)


class Metric3DModel(DepthModel):
    has_conf = True
    needs_focal = True

    def __init__(self, size: str = "large", device: str | None = None):
        self.device = device or DEVICE
        self.model_id = f"{HUB_REPO}:{ENTRIES[size]}"
        self.tag = f"metric3d-{size}"
        self.model = freeze(load_hub_model(ENTRIES[size])).to(self.device)
        self.mean = torch.tensor(MEAN, device=self.device)[:, None, None]
        self.std = torch.tensor(STD, device=self.device)[:, None, None]

    @torch.inference_mode()
    def predict(self, rgb: np.ndarray, focal: float | None = None) -> dict[str, np.ndarray]:
        if focal is None or not np.isfinite(focal) or focal <= 0:
            raise ValueError(f"Metric3D needs the focal length in pixels, got {focal!r}")
        h, w = rgb.shape[:2]
        scale = min(INPUT_SIZE[0] / h, INPUT_SIZE[1] / w)
        rh, rw = int(h * scale), int(w * scale)
        x = torch.from_numpy(np.ascontiguousarray(rgb)).to(self.device).permute(2, 0, 1).float()
        x = F.interpolate(x[None], size=(rh, rw), mode="bilinear", align_corners=False)[0]
        x = (x - self.mean) / self.std
        ph, pw = INPUT_SIZE[0] - rh, INPUT_SIZE[1] - rw
        top, left = ph // 2, pw // 2
        x = F.pad(x, (left, pw - left, top, ph - top), value=0.0)  # = padding with MEAN before normalising

        out = self.model.inference({"input": x[None]})
        depth, conf = out[0], out[1]

        def restore(m: torch.Tensor) -> torch.Tensor:
            m = m.reshape(INPUT_SIZE)[top : top + rh, left : left + rw].float()
            return F.interpolate(m[None, None], size=(h, w), mode="bilinear", align_corners=False)[0, 0]

        depth = torch.clamp(restore(depth) * (focal * scale / CANONICAL_FOCAL), 0, MAX_DEPTH)
        conf = torch.clamp(1.0 - restore(conf), min=0.0)
        return {"depth": depth.cpu().numpy(), "conf": conf.cpu().numpy()}

    def info(self) -> dict:
        return {**super().info(), "input_size": INPUT_SIZE, "conf": "max(1 - confidence, 0)"}
