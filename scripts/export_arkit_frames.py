"""Export the qualitative-figure frames of ARKitScenes (the candidates of experiments.qual_dual) at 480 x 640.

    python -m scripts.export_arkit_frames --candidates candidates.csv [--ids arkitscenes_<video>_<ts> ...] --out qual_frames

Per sample_id, in the same upright orientation as the prepared tables: <id>_rgb.png, <id>_gt.png and
<id>_lidar.png (uint16 mm, 0 = no value; laser ground truth and iPad LiDAR on the 480 x 640 grid).
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image

from src.data.arkitscenes import ARKitScenes


def to_mm(depth: np.ndarray) -> np.ndarray:
    return np.clip(np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0) * 1000.0 + 0.5, 0, 65535).astype(np.uint16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", nargs="*", default=[], help="candidates.csv files (column sample_id)")
    ap.add_argument("--ids", nargs="*", default=[])
    ap.add_argument("--root", default=None, help="ARKitScenes root (default: DATA_ROOT/arkitscenes)")
    ap.add_argument("--out", default="qual_frames")
    args = ap.parse_args()

    want = set(args.ids)
    for path in args.candidates:
        want |= {r["sample_id"] for r in csv.DictReader(open(path))}
    if not want:
        raise SystemExit("no sample ids given")
    ds = ARKitScenes(args.root)
    index = {ds.sample_id(i): i for i in range(len(ds))}
    missing = sorted(want - index.keys())
    if missing:
        raise SystemExit(f"not in the dataset: {missing}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for sid in sorted(want):
        s = ds[index[sid]]
        Image.fromarray(s["rgb"]).save(out / f"{sid}_rgb.png")
        Image.fromarray(to_mm(s["gt"])).save(out / f"{sid}_gt.png")
        Image.fromarray(to_mm(s["lidar"])).save(out / f"{sid}_lidar.png")
        print(sid, s["rgb"].shape, flush=True)
    print(f"wrote {len(want)} frames to {out}")


if __name__ == "__main__":
    main()
