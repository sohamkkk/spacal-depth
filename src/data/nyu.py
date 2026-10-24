"""NYU Depth v2 labelled subset (section 8): nyu_depth_v2_labeled.mat (HDF5 /
MATLAB v7.3, 1449 images, 464 scenes) + splits.mat (official 795/654).

Ground truth is `rawDepths` (holes = 0), not the colorization-inpainted
`depths`: inpainted values are fabricated exactly on mirrors/windows, which
would contaminate every coverage number. Zero-shot models (DA-V2 metric,
UniDepthV2) never saw NYU, so all 1449 images are usable for calibration;
`official_test` marks the 654 Eigen test images for E0 comparability.
"""

import h5py
import numpy as np
from scipy.io import loadmat

from src.config import DATA_ROOT
from src.data.protocols import NYU_GLASS_IDS as GLASS_IDS  # noqa: F401  (re-exported)
from src.data.protocols import NYU_MIRROR_IDS as MIRROR_IDS  # noqa: F401
from src.data.protocols import NYU_WINDOW_IDS as WINDOW_IDS  # noqa: F401

REQUIRED_KEYS = ("images", "rawDepths", "labels", "scenes")
FOCAL = 518.8579  # fx of the RGB camera (NYU toolbox camera_params.m), for models that need intrinsics


def decode_matlab_string(f: h5py.File, ref) -> str:
    return "".join(chr(int(c)) for c in np.array(f[ref]).ravel())


class NYUDepthV2:
    name = "nyu"

    def __init__(self, root=None, expected_test: int = 654):
        root = root or (DATA_ROOT / "nyu")
        self.mat_path = root / "nyu_depth_v2_labeled.mat"
        self._file = h5py.File(self.mat_path, "r")
        missing = [k for k in REQUIRED_KEYS if k not in self._file]
        if missing:
            raise KeyError(f"keys {missing} missing from {self.mat_path}; found {list(self._file.keys())}")
        n = self._file["images"].shape[0]
        scene_refs = self._file["scenes"]
        if scene_refs.shape != (1, n):
            raise ValueError(f"expected scenes shape (1, {n}), got {scene_refs.shape}")
        self._scenes = [decode_matlab_string(self._file, r) for r in scene_refs[0]]
        splits = loadmat(root / "splits.mat")
        self._test = set(int(i) - 1 for i in splits["testNdxs"].ravel())  # 1-based in the file
        if len(self._test) != expected_test:
            raise ValueError(f"expected {expected_test} official test indices, got {len(self._test)}")

    def __len__(self) -> int:
        return len(self._scenes)

    def scene_id(self, idx: int) -> str:
        return self._scenes[idx]

    def all_scene_ids(self) -> list[str]:
        return list(self._scenes)

    def __getitem__(self, idx: int) -> dict:
        rgb = np.ascontiguousarray(self._file["images"][idx].T)  # (3,640,480) -> (480,640,3)
        gt = np.ascontiguousarray(self._file["rawDepths"][idx].T).astype(np.float32)
        labels = np.ascontiguousarray(self._file["labels"][idx].T).astype(np.uint16)
        return {
            "rgb": rgb,
            "gt": gt,
            "labels": labels,
            "scene_id": self._scenes[idx],
            "sample_id": f"nyu_{idx:05d}",
            "meta": {"index": idx, "official_test": idx in self._test, "focal": FOCAL},
        }
