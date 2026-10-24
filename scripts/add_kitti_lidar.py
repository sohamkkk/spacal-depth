"""Add the single raw LiDAR scan of every KITTI frame to the prepared KITTI tables (field `lidar`, metres,
0 = no return), for E16's real-sensor anchors. CPU only; the tables' other fields are unchanged.

The source is the KITTI depth-completion release `data_depth_velodyne.zip` (projected raw scans, 5 GB; members
`{train,val}/<drive>/proj_depth/velodyne_raw/image_0{2,3}/<frame>.png`, uint16 / 256 = metres). It is read in
place. The ground truth our tables use is the depth-prediction annotation, which accumulates 11 scans and
removes outliers, so a raw scan differs from it by real sensor effects: occlusion bleed where the LiDAR sees
past an edge the camera does not, moving objects, and range noise.

Each raw map gets the same crops as the GT (KITTI benchmark crop, then the Garg crop) and is read at the table's
pixels. Table pixels only exist where the annotated GT is valid, so returns on the lattice without annotated GT
cannot become anchors; their share is reported as `frac_lattice_returns_without_gt`.

    python -m scripts.add_kitti_lidar --zip data/kitti/data_depth_velodyne.zip
"""

import argparse
import io
import re
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from experiments.common import table_path
from src.config import PREPARED_ROOT
from src.data.kitti import sequence_id
from src.data.prepared import PixelTable
from src.data.protocols import PROTOCOLS, crop, kitti_benchmark_crop

_MEMBER = re.compile(r"(?:^|/)(?:train|val)/(\d{4}_\d{2}_\d{2}_drive_\d{4}_sync)/proj_depth/velodyne_raw/"
                     r"(image_0[23])/(\d{10})\.png$")
AGREE_TOL = 0.05  # median |log raw - log GT| above this means the wrong frame or camera


def index_zip(zf: zipfile.ZipFile) -> dict[tuple[str, str, str], str]:
    out = {}
    for name in zf.namelist():
        if m := _MEMBER.search(name):
            out[(m.group(1), m.group(2), m.group(3))] = name
    if not out:
        raise ValueError("no velodyne_raw PNGs found in the zip; is it data_depth_velodyne.zip?")
    return out


def read_raw(zf: zipfile.ZipFile, member: str) -> np.ndarray:
    raw = np.array(Image.open(io.BytesIO(zf.read(member))), dtype=np.float32) / 256.0
    proto = PROTOCOLS["kitti"]
    raw = crop(kitti_benchmark_crop(raw), proto)
    raw[(raw <= proto.min_depth) | (raw >= proto.max_depth)] = 0.0
    return raw


def frame_key(sample_id: str) -> tuple[str, str]:
    drive = sequence_id(sample_id)
    frame = sample_id.rsplit("_", 1)[1]
    if not re.fullmatch(r"\d{10}", frame):
        raise ValueError(f"cannot parse a frame number from {sample_id!r}")
    return drive, frame


def add_lidar(t: PixelTable, zf: zipfile.ZipFile, index: dict) -> tuple[np.ndarray, dict]:
    stride = int(t.meta.get("stride", 1))
    lidar = np.zeros(t.n_pixels, np.float32)
    cams, missing, n_lat, n_lat_in_table = {}, [], 0, 0
    for i, sid in enumerate(t.img["sample_id"].astype(str)):
        drive, frame = frame_key(sid)
        sl = slice(t.offsets[i], t.offsets[i + 1])
        r, c = t.pix["row"][sl].astype(np.int64), t.pix["col"][sl].astype(np.int64)
        gt = t.pix["gt"][sl].astype(np.float64)
        chosen = None
        for cam in ("image_02", "image_03"):
            if (drive, cam, frame) not in index:
                continue
            raw = read_raw(zf, index[(drive, cam, frame)])
            if raw.shape[0] <= r.max() or raw.shape[1] <= c.max():
                raise ValueError(f"{sid}: raw map {raw.shape} smaller than the table's pixel grid")
            v = raw[r, c]
            ok = v > 0
            if ok.sum() >= 20 and np.median(np.abs(np.log(v[ok]) - np.log(gt[ok]))) < AGREE_TOL:
                chosen = (cam, raw, v)
                break
        if chosen is None:
            missing.append(sid)
            continue
        cam, raw, v = chosen
        cams[cam] = cams.get(cam, 0) + 1
        lidar[sl] = v
        n_lat += int((raw[::stride, ::stride] > 0).sum())
        n_lat_in_table += int((v > 0).sum())
    has = lidar > 0
    e = np.abs(np.log(lidar[has]) - np.log(t.pix["gt"][has].astype(np.float64)))
    per_img = np.bincount(t.image_of_pixel()[has], minlength=t.n_images)
    stats = {
        "source": "KITTI data_depth_velodyne.zip (single raw scan)",
        "n_images": t.n_images, "n_missing": len(missing), "missing_examples": missing[:5], "cameras": cams,
        "frac_table_pixels_with_return": float(has.mean()),
        "returns_per_image_median": float(np.median(per_img)), "returns_per_image_min": int(per_img.min()),
        "frac_lattice_returns_without_gt": float(1 - n_lat_in_table / max(n_lat, 1)),
        "err_median_abs_log": float(np.median(e)), "err_frac_gt_10pct": float(np.mean(e > np.log(1.1))),
        "err_frac_gt_25pct": float(np.mean(e > np.log(1.25))),
    }
    return lidar, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", type=Path, required=True, help="data_depth_velodyne.zip")
    ap.add_argument("--models", nargs="*", help="model tags (default: every prepared kitti table)")
    ap.add_argument("--max-missing-frac", type=float, default=0.02)
    args = ap.parse_args()

    paths = [table_path("kitti", m) for m in args.models] if args.models else sorted(PREPARED_ROOT.glob("kitti__*.npz"))
    if not paths:
        raise FileNotFoundError(f"no kitti tables under {PREPARED_ROOT}")
    with zipfile.ZipFile(args.zip) as zf:
        index = index_zip(zf)
        print(f"{len(index)} raw scans indexed in {args.zip}")
        for path in paths:
            t = PixelTable.load(path)
            lidar, stats = add_lidar(t, zf, index)
            print(path.name, stats)
            if stats["n_missing"] > args.max_missing_frac * t.n_images:
                raise RuntimeError(f"{path.name}: {stats['n_missing']} images without a matching raw scan; "
                                   f"table left unchanged")
            t.pix["lidar"] = lidar
            t.meta["lidar"] = stats
            tmp = path.with_name(path.stem + ".tmp.npz")
            t.save(tmp)
            tmp.replace(path)


if __name__ == "__main__":
    main()
