"""KITTI loader (section 8) for the Marigold evaluation archives:
`kitti_eigen_split_test.tar` (Eigen test, 652 frames with improved GT) and
`kitti_sampled_val_800.tar` (800 frames from Eigen-train drives, unseen by
zero-shot models — enlarges the calibration pool). Split lists have lines
`<rgb_rel> <gt_rel|None> <focal>`.

GT is sparse LiDAR (uint16 / 256 = metres, 0 = invalid). Exchangeable unit
is the DRIVE, parsed from the path. Images get the KITTI benchmark crop
(352 x 1216) before inference; the Garg crop is applied later by the protocol.
"""

import re
from pathlib import Path

import numpy as np
from PIL import Image

from src.config import DATA_ROOT
from src.data.protocols import kitti_benchmark_crop

_SEQ_PATTERN = re.compile(r"\d{4}_\d{2}_\d{2}_drive_\d{4}_sync")

SUBSETS = {
    "eigen_test": ("eigen_test", "eigen_test_files_with_gt.txt"),
    "val800": ("val800", "eigen_val_from_train_800.txt"),
}


def sequence_id(path) -> str:
    m = _SEQ_PATTERN.search(str(path))
    if not m:
        raise ValueError(f"could not parse a KITTI drive sequence id from {path}")
    return m.group(0)


def _resolve(root: Path, rel: str) -> Path:
    """The tars may or may not contain a top-level folder; try both."""
    candidates = [root / rel] + [d / rel for d in sorted(root.iterdir()) if d.is_dir()]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"{rel} not found under {root} (tried {len(candidates)} locations)")


def read_split_list(path: Path) -> list[tuple[str, str, float]]:
    rows = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 2 or parts[1] == "None":
            continue
        focal = float(parts[2]) if len(parts) > 2 else float("nan")
        rows.append((parts[0], parts[1], focal))
    return rows


class KITTI:
    name = "kitti"

    def __init__(self, subsets=("eigen_test", "val800"), root=None):
        root = Path(root or (DATA_ROOT / "kitti"))
        self.items = []
        for subset in subsets:
            folder, list_name = SUBSETS[subset]
            sub_root = root / folder
            for rgb_rel, gt_rel, focal in read_split_list(root / "splits" / list_name):
                self.items.append((subset, sub_root, rgb_rel, gt_rel, focal))
        if not self.items:
            raise FileNotFoundError(f"no KITTI items found under {root}")

    def __len__(self) -> int:
        return len(self.items)

    def scene_id(self, idx: int) -> str:
        return sequence_id(self.items[idx][2])

    def all_scene_ids(self) -> list[str]:
        return [self.scene_id(i) for i in range(len(self))]

    def __getitem__(self, idx: int) -> dict:
        subset, sub_root, rgb_rel, gt_rel, focal = self.items[idx]
        rgb = np.array(Image.open(_resolve(sub_root, rgb_rel)).convert("RGB"))
        gt = np.array(Image.open(_resolve(sub_root, gt_rel)), dtype=np.float32) / 256.0
        if rgb.shape[:2] != gt.shape:
            raise ValueError(f"rgb {rgb.shape} vs gt {gt.shape} mismatch for {rgb_rel}")
        frame = Path(rgb_rel).stem
        return {
            "rgb": np.ascontiguousarray(kitti_benchmark_crop(rgb)),
            "gt": np.ascontiguousarray(kitti_benchmark_crop(gt)),
            "scene_id": sequence_id(rgb_rel),
            "sample_id": f"kitti_{subset}_{sequence_id(rgb_rel)}_{frame}",
            "meta": {"index": idx, "subset": subset, "official_test": subset == "eigen_test", "focal": focal},
        }
