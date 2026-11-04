"""Resumable download of the ARKitScenes depth-upsampling zips (one zip per video) and their intrinsics.

    python scripts/fetch_arkitscenes.py --dst /content/drive/MyDrive/spacal/data/arkitscenes [--split Validation]

Each zip is fetched to a local temp file, size-checked against Content-Length and then moved into
<dst>/zips/<split>/ (and <dst>/intrinsics/<split>/), so an interrupted run leaves no partial zip in <dst>
and a rerun skips finished videos.
Upsampling zip: <video_id>/{wide,highres_depth,lowres_depth,confidence}/<video_id>_<timestamp>.png
(depths uint16 mm; highres_depth is rendered from the Faro laser scan, lowres_depth is the iPad LiDAR).
The upsampling release has no intrinsics, so the raw release's vga_wide_intrinsics.zip is fetched as well:
vga_wide_intrinsics/<video_id>_<timestamp>.pincam = "640 480 fx fy cx cy" for the same wide camera at
640 x 480, at ~30 Hz (the nearest one is within ~20 ms of an upsampling frame).
"""
import argparse
import csv
import io
import os
import shutil
import subprocess
import time
import urllib.request

BASE = "https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1"
SPLITS_CSV = ("https://raw.githubusercontent.com/apple/ARKitScenes/main/"
              "depth_upsampling/upsampling_train_val_splits.csv")


def video_ids(split):
    with urllib.request.urlopen(SPLITS_CSV, timeout=60) as r:
        rows = csv.DictReader(io.StringIO(r.read().decode()))
        return sorted(row["video_id"] for row in rows if row["fold"] == split)


def remote_size(url):
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers["Content-Length"])


def fetch(url, out, tmp_dir, retries=3):
    size = remote_size(url)
    tmp = os.path.join(tmp_dir, os.path.basename(out) + ".part")
    for attempt in range(retries):
        subprocess.run(["curl", "-sL", "--fail", "--retry", "3", "-o", tmp, url], check=False)
        if os.path.exists(tmp) and os.path.getsize(tmp) == size:
            shutil.move(tmp, out)
            return size
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"failed after {retries} attempts: {url}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dst", required=True)
    ap.add_argument("--split", default="Validation", choices=["Training", "Validation"])
    ap.add_argument("--tmp", default="/tmp/arkit_dl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0, help="with --n-shards: fetch every n-th video (parallel runs)")
    ap.add_argument("--n-shards", type=int, default=1)
    args = ap.parse_args()

    zdir = os.path.join(args.dst, "zips", args.split)
    os.makedirs(zdir, exist_ok=True)
    os.makedirs(args.tmp, exist_ok=True)
    meta = {"upsampling_train_val_splits.csv": SPLITS_CSV, "metadata.csv": f"{BASE}/raw/metadata.csv"}
    if args.split == "Validation":
        meta["val_attributes.csv"] = f"{BASE}/upsampling/Validation/val_attributes.csv"
    for name, url in meta.items():
        if not os.path.exists(os.path.join(args.dst, name)):
            urllib.request.urlretrieve(url, os.path.join(args.dst, name))

    idir = os.path.join(args.dst, "intrinsics", args.split)
    os.makedirs(idir, exist_ok=True)
    vids = video_ids(args.split)
    if args.limit:
        vids = vids[:args.limit]
    vids = vids[args.shard::args.n_shards]
    t0, done_bytes = time.time(), 0
    for i, vid in enumerate(vids, 1):
        jobs = [(f"{BASE}/upsampling/{args.split}/{vid}.zip", os.path.join(zdir, f"{vid}.zip")),
                (f"{BASE}/raw/{args.split}/{vid}/vga_wide_intrinsics.zip", os.path.join(idir, f"{vid}.zip"))]
        todo = [(url, out) for url, out in jobs if not os.path.exists(out)]
        for url, out in todo:
            done_bytes += fetch(url, out, args.tmp)
        if todo:
            rate = done_bytes / max(time.time() - t0, 1e-6) / 1e6
            print(f"[{i}/{len(vids)}] {vid} ok  {rate:.1f} MB/s", flush=True)
    n = len([f for f in os.listdir(zdir) if f.endswith(".zip")])
    n_i = len([f for f in os.listdir(idir) if f.endswith(".zip")])
    print(f"DONE {n}/{len(vids)} zips in {zdir}, {n_i}/{len(vids)} intrinsics in {idir}", flush=True)


if __name__ == "__main__":
    main()
