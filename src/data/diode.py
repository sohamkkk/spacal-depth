"""DIODE validation set (distribution shift, E5): official `val.tar.gz`,
layout `val/{indoors,outdoor}/scene_XXXXX/scan_XXXXX/<id>.png` with
`<id>_depth.npy` (H, W, 1) and `<id>_depth_mask.npy` (H, W). Scene id is the
`scene_XXXXX` folder, which is the exchangeable unit.
"""

from pathlib import Path

import numpy as np
from PIL import Image

from src.config import DATA_ROOT

_FOLDER = {"indoor": "indoors", "outdoor": "outdoor"}
FOCAL = 886.81  # fx at 1024 x 768, shared by every DIODE image `[VERIFY: DIODE intrinsics]`


class DIODE:
    def __init__(self, split: str = "indoor", root=None):
        if split not in _FOLDER:
            raise ValueError(f"split must be one of {list(_FOLDER)}")
        self.name = f"diode_{split}"
        base = Path(root or (DATA_ROOT / "diode")) / "val" / _FOLDER[split]
        self.rgb_paths = sorted(base.glob("scene_*/scan_*/*.png"))
        if not self.rgb_paths:
            raise FileNotFoundError(f"no DIODE images under {base}")

    def __len__(self) -> int:
        return len(self.rgb_paths)

    def scene_id(self, idx: int) -> str:
        return self.rgb_paths[idx].parent.parent.name

    def all_scene_ids(self) -> list[str]:
        return [self.scene_id(i) for i in range(len(self))]

    def __getitem__(self, idx: int) -> dict:
        p = self.rgb_paths[idx]
        rgb = np.array(Image.open(p).convert("RGB"))
        gt = np.load(p.with_name(p.stem + "_depth.npy")).squeeze(-1).astype(np.float32)
        mask = np.load(p.with_name(p.stem + "_depth_mask.npy")).astype(bool)
        return {
            "rgb": rgb,
            "gt": gt,
            "extra_mask": mask,
            "scene_id": self.scene_id(idx),
            "sample_id": f"{self.name}_{p.stem}",
            "meta": {"index": idx, "official_test": True, "focal": FOCAL},
        }
