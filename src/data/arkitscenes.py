"""ARKitScenes depth-upsampling validation split (Baruch et al., NeurIPS 2021 Datasets): 287 videos from
107 visits (rooms), 5,594 usable frames, with an independent real sensor next to the ground truth:
  highres_depth  1920 x 1440, rendered from the Faro laser scan of the room (ground truth)
  lowres_depth   256 x 192, the iPad Pro LiDAR depth as delivered by ARKit (the sensor), plus
  confidence     256 x 192, ARKit's 0 / 1 / 2 confidence for lowres_depth.

Read straight from the zips that scripts/fetch_arkitscenes.py stores under <root>:
  zips/Validation/<video>.zip, intrinsics/Validation/<video>.zip, upsampling_train_val_splits.csv, metadata.csv

Everything is reduced 3x to 480 x 640 (the VGA grid, so the raw release's vga_wide intrinsics apply directly):
RGB box-filtered, ground truth by the lower median of each 3 x 3 block (as for ETH3D), and the sensor read
at the lowres pixel nearest to each grid pixel centre (field `lidar`, metres, 0 = no value; `lidar_conf`).
Frames are then rotated upright from the metadata's sky_direction (ARKitScenes' own convention: Left ->
90 degrees clockwise, Right -> counter-clockwise, Down -> 180), so 102 videos become 640 x 480 portrait.
The visit is the exchangeable unit (videos of one visit film the same room).
"""

import csv
import io
import re
import zipfile
from bisect import bisect_left
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from src.config import DATA_ROOT
from src.data.eth3d import downsample_depth, downsample_rgb

DOWNSAMPLE = 3
SPLIT = "Validation"
ROT90_K = {"Up": 0, "Left": -1, "Right": 1, "Down": 2}  # np.rot90 k: negative = clockwise
_STEM = re.compile(r"(\d+)_(\d+\.\d+)\.png$")
_PINCAM = re.compile(r"_(\d+\.\d+)\.pincam$")


def sensor_on_grid(low: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour read of a low-resolution map at the pixel centres of an h x w grid."""
    h, w = shape
    r = np.minimum(((np.arange(h) + 0.5) * low.shape[0] / h).astype(int), low.shape[0] - 1)
    c = np.minimum(((np.arange(w) + 0.5) * low.shape[1] / w).astype(int), low.shape[1] - 1)
    return low[np.ix_(r, c)]


def read_png(zf: zipfile.ZipFile, name: str) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(zf.read(name))))


def nearest_focal(zf: zipfile.ZipFile, ts: float) -> tuple[float, float]:
    """(fx, nearest-timestamp gap in s) from vga_wide_intrinsics/*.pincam ("640 480 fx fy cx cy")."""
    stamps, names = _pincam_index(zf)
    i = bisect_left(stamps, ts)
    j = min((k for k in (i - 1, i) if 0 <= k < len(stamps)), key=lambda k: abs(stamps[k] - ts))
    w, h, fx = (float(v) for v in zf.read(names[j]).decode().split()[:3])
    if (w, h) != (640.0, 480.0):
        raise ValueError(f"{names[j]}: expected a 640 x 480 camera, got {w:g} x {h:g}")
    return fx, abs(stamps[j] - ts)


@lru_cache(maxsize=4)
def _pincam_index(zf: zipfile.ZipFile) -> tuple[list[float], list[str]]:
    pairs = sorted((float(m.group(1)), n) for n in zf.namelist() if (m := _PINCAM.search(n)))
    if not pairs:
        raise FileNotFoundError(f"no .pincam files in {zf.filename}")
    return [p[0] for p in pairs], [p[1] for p in pairs]


class ARKitScenes:
    name = "arkitscenes"
    MAX_FOCAL_GAP = 0.1  # s; intrinsics are sampled at ~30 Hz

    def __init__(self, root=None, split: str = SPLIT):
        self.root = Path(root or (DATA_ROOT / "arkitscenes"))
        zdir, idir = self.root / "zips" / split, self.root / "intrinsics" / split
        if not zdir.is_dir():
            raise FileNotFoundError(f"no {zdir} (run scripts/fetch_arkitscenes.py --dst {self.root})")
        visits = {r["video_id"]: r["visit_id"] for r in csv.DictReader(open(self.root / "upsampling_train_val_splits.csv"))}
        sky = {r["video_id"]: r["sky_direction"] for r in csv.DictReader(open(self.root / "metadata.csv"))}
        self.items = []
        for zp in sorted(zdir.glob("*.zip")):
            vid = zp.stem
            if not (idir / f"{vid}.zip").exists():
                raise FileNotFoundError(f"no intrinsics for video {vid} in {idir}")
            if sky.get(vid) not in ROT90_K:
                raise ValueError(f"video {vid}: unknown sky_direction {sky.get(vid)!r}")
            with zipfile.ZipFile(zp) as zf:
                names = set(zf.namelist())
            stems = sorted(m.group(0)[:-4] for n in names if "/lowres_depth/" in n and (m := _STEM.search(n)))
            # video 48458647 ships without confidence maps; its 5 frames are dropped (5,594 frames remain)
            self.items += [(vid, s) for s in stems
                           if all(f"{vid}/{f}/{s}.png" in names for f in ("wide", "highres_depth", "confidence"))]
        if not self.items:
            raise FileNotFoundError(f"no ARKitScenes frames under {zdir}")
        self.visits, self.sky, self.zdir, self.idir = visits, sky, zdir, idir

    def __len__(self) -> int:
        return len(self.items)

    def scene_id(self, idx: int) -> str:
        return self.visits[self.items[idx][0]]

    def sample_id(self, idx: int) -> str:
        return f"arkitscenes_{self.items[idx][1]}"

    def all_scene_ids(self) -> list[str]:
        return [self.visits[vid] for vid, _ in self.items]

    @lru_cache(maxsize=2)
    def _zips(self, vid: str) -> tuple[zipfile.ZipFile, zipfile.ZipFile]:
        return zipfile.ZipFile(self.zdir / f"{vid}.zip"), zipfile.ZipFile(self.idir / f"{vid}.zip")

    def __getitem__(self, idx: int) -> dict:
        vid, stem = self.items[idx]
        zf, zi = self._zips(vid)
        rgb = read_png(zf, f"{vid}/wide/{stem}.png")[..., :3]
        hi = read_png(zf, f"{vid}/highres_depth/{stem}.png").astype(np.float32) / 1000.0
        low = read_png(zf, f"{vid}/lowres_depth/{stem}.png").astype(np.float32) / 1000.0
        conf = read_png(zf, f"{vid}/confidence/{stem}.png").astype(np.uint8)
        if hi.shape != rgb.shape[:2]:
            raise ValueError(f"{vid}/{stem}: depth {hi.shape} vs image {rgb.shape[:2]}")
        rgb, gt = downsample_rgb(rgb, DOWNSAMPLE), downsample_depth(hi, DOWNSAMPLE)
        lidar, lidar_conf = sensor_on_grid(low, gt.shape), sensor_on_grid(conf, gt.shape)
        fx, gap = nearest_focal(zi, float(stem.split("_")[1]))
        if gap > self.MAX_FOCAL_GAP:
            raise ValueError(f"{vid}/{stem}: nearest intrinsics are {gap:.3f} s away")
        k = ROT90_K[self.sky[vid]]
        rgb, gt, lidar, lidar_conf = (np.ascontiguousarray(np.rot90(a, k)) for a in (rgb, gt, lidar, lidar_conf))
        return {
            "rgb": rgb,
            "gt": gt,
            "lidar": lidar,
            "lidar_conf": lidar_conf,
            "scene_id": self.visits[vid],
            "sample_id": f"arkitscenes_{stem}",
            "meta": {"index": idx, "official_test": True, "focal": fx, "video": vid, "sky": self.sky[vid]},
        }
