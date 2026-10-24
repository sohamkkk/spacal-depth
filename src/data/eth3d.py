"""ETH3D high-resolution multi-view training set (Schöps et al., CVPR 2017) as a fifth, zero-shot dataset:
13 scenes (7 indoor, 6 outdoor), 454 DSLR images at 6048 x 4032 with laser-scan depth.

The rendered depth maps match the ORIGINAL (distorted) images, not ETH3D's pre-undistorted ones
(eth3d.net/documentation, "Training data"), so we use the original JPGs and their THIN_PRISM_FISHEYE
calibration. scripts/prepare_eth3d.py streams the Marigold evaluation copy (same files, distorted JPGs)
and stores everything reduced by DOWNSAMPLE, ~4 GB instead of ~49 GB:
  <scene>/rgb/<stem>.png                      box-filtered RGB
  <scene>/depth/<stem>.npy                    float32, inf = no depth (see downsample_depth)
  <scene>/calibration/{cameras,images}.txt    COLMAP text format, full-resolution intrinsics

Metric3D gets fx / DOWNSAMPLE as its focal length. For the fisheye model that is exact at the image centre
and only approximate towards the borders (radial distortion up to ~100 px at full resolution).
The scene is the exchangeable unit; with 13 scenes, scene-level calibration is infeasible, as on DIODE.
"""

from pathlib import Path

import numpy as np
from PIL import Image

from src.config import DATA_ROOT

DOWNSAMPLE = 4
INDOOR = ("delivery_area", "kicker", "office", "pipes", "relief", "relief_2", "terrains")
OUTDOOR = ("courtyard", "electro", "facade", "meadow", "playground", "terrace")
CAMERA_MODELS = ("THIN_PRISM_FISHEYE", "PINHOLE")  # both list fx first among the parameters


def read_colmap_focals(calib_dir: Path) -> dict[str, float]:
    """Image stem -> fx (full resolution) from COLMAP cameras.txt / images.txt."""
    fx = {}
    for line in (calib_dir / "cameras.txt").read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        if parts[1] not in CAMERA_MODELS:
            raise ValueError(f"{calib_dir}: camera {parts[0]} is {parts[1]}, expected one of {CAMERA_MODELS}")
        fx[parts[0]] = float(parts[4])
    lines = [ln for ln in (calib_dir / "images.txt").read_text().splitlines() if not ln.startswith("#")]
    focals = {}
    for line in lines[::2]:  # each image has a pose line followed by a 2D-points line
        parts = line.split()
        if len(parts) >= 10:
            focals[Path(parts[9]).stem] = fx[parts[8]]
    return focals


def downsample_rgb(rgb: np.ndarray, f: int = DOWNSAMPLE) -> np.ndarray:
    h, w = rgb.shape[0] // f, rgb.shape[1] // f
    return np.asarray(Image.fromarray(rgb).resize((w, h), Image.BOX))


def downsample_depth(depth: np.ndarray, f: int = DOWNSAMPLE) -> np.ndarray:
    """Lower median of the finite depths in each f x f block (inf if none). The rendered depth is sparse
    at full resolution, and the lower median is always one of the measured values, so depths on the two
    sides of an edge are never averaged."""
    h, w = depth.shape[0] // f, depth.shape[1] // f
    blocks = depth[: h * f, : w * f].reshape(h, f, w, f).transpose(0, 2, 1, 3).reshape(h, w, f * f)
    blocks = np.where(np.isfinite(blocks) & (blocks > 0), blocks, np.inf)
    n = np.isfinite(blocks).sum(-1)
    picked = np.take_along_axis(np.sort(blocks, -1), np.maximum(n - 1, 0)[..., None] // 2, -1)[..., 0]
    return np.where(n > 0, picked, np.inf).astype(np.float32)


class ETH3D:
    name = "eth3d"

    def __init__(self, root=None, scenes=INDOOR + OUTDOOR):
        root = Path(root or (DATA_ROOT / "eth3d"))
        self.items = []
        for scene in scenes:
            depth_dir = root / scene / "depth"
            if not depth_dir.is_dir():
                raise FileNotFoundError(f"ETH3D scene {scene}: no {depth_dir} (run scripts/download_data.sh eth3d)")
            focals = read_colmap_focals(root / scene / "calibration")
            for d in sorted(depth_dir.glob("*.npy")):
                rgb = root / scene / "rgb" / f"{d.stem}.png"
                if not rgb.exists():
                    raise FileNotFoundError(f"no image for depth map {d}")
                self.items.append((scene, rgb, d, focals[d.stem]))
        if not self.items:
            raise FileNotFoundError(f"no ETH3D images under {root}")

    def __len__(self) -> int:
        return len(self.items)

    def scene_id(self, idx: int) -> str:
        return self.items[idx][0]

    def all_scene_ids(self) -> list[str]:
        return [it[0] for it in self.items]

    def __getitem__(self, idx: int) -> dict:
        scene, rgb_path, depth_path, focal = self.items[idx]
        rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
        gt = np.load(depth_path)
        if gt.shape != rgb.shape[:2]:
            raise ValueError(f"{depth_path}: depth {gt.shape} vs image {rgb.shape[:2]}")
        return {
            "rgb": np.ascontiguousarray(rgb),
            "gt": gt,
            "scene_id": scene,
            "sample_id": f"eth3d_{scene}_{rgb_path.stem}",
            "meta": {"index": idx, "official_test": True, "focal": focal / DOWNSAMPLE,
                     "domain": "indoor" if scene in INDOOR else "outdoor"},
        }
