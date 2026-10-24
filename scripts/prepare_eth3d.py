"""Stream ETH3D (Marigold's evaluation copy: original DSLR JPGs + rendered depth, one 49 GB tar) and keep
only the DOWNSAMPLE-reduced images and depth maps, ~4 GB, in the layout src/data/eth3d.py reads.
Nothing large touches the disk. Resumable: the byte offset of the next tar member is saved after each
member, and a restart continues from there with an HTTP Range request.

    python -m scripts.prepare_eth3d                       # into $SPACAL_DATA_ROOT/eth3d
    python -m scripts.prepare_eth3d --tar /path/eth3d.tar # from a local copy
"""

import argparse
import http.client
import io
import json
import tarfile
import time
import urllib.request
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image

from src.config import DATA_ROOT
from src.data.eth3d import DOWNSAMPLE, INDOOR, OUTDOOR, downsample_depth, downsample_rgb

URL = "https://share.phys.ethz.ch/~pf/bingkedata/marigold/evaluation_dataset/eth3d/eth3d.tar"
N_IMAGES = 454


def route(name: str) -> tuple[str, str, str] | None:
    """Tar member -> (kind, scene, file name), or None to skip. Member paths:
    rgb/<scene>/dslr_calibration_jpg/{cameras,images}.txt, rgb/<scene>/images/dslr_images/<name>.JPG,
    depth/<scene>_dslr_depth/<scene>/ground_truth_depth/dslr_images/<name>.JPG (raw float32, not a JPEG)."""
    p = PurePosixPath(name).parts
    p = p[1:] if p and p[0] == "." else p
    if len(p) == 4 and p[0] == "rgb" and p[2] == "dslr_calibration_jpg" and p[3] in ("cameras.txt", "images.txt"):
        return "calib", p[1], p[3]
    if len(p) == 5 and p[0] == "rgb" and p[2:4] == ("images", "dslr_images"):
        return "rgb", p[1], p[4]
    if len(p) == 6 and p[0] == "depth" and p[3:5] == ("ground_truth_depth", "dslr_images"):
        return "depth", p[2], p[5]
    return None


def write_atomic(path: Path, save) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        save(fh)
    tmp.replace(path)


def handle(kind: str, scene: str, fname: str, data: bytes, out: Path) -> None:
    stem = PurePosixPath(fname).stem
    if kind == "calib":
        write_atomic(out / scene / "calibration" / fname, lambda fh: fh.write(data))
    elif kind == "rgb":
        rgb = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
        write_atomic(out / scene / "rgb" / f"{stem}.png",
                     lambda fh: Image.fromarray(downsample_rgb(rgb)).save(fh, format="PNG"))
    else:
        w, h = Image.open(out / scene / "rgb" / f"{stem}.png").size  # images precede depth in the tar
        h, w = h * DOWNSAMPLE, w * DOWNSAMPLE
        depth = np.frombuffer(data, dtype=np.float32)
        if depth.size != h * w:
            raise ValueError(f"{scene}/{fname}: {depth.size} floats, expected {h} x {w}")
        write_atomic(out / scene / "depth" / f"{stem}.npy",
                     lambda fh: np.save(fh, downsample_depth(depth.reshape(h, w))))


def open_stream(src: str, offset: int):
    if Path(src).exists():
        fh = open(src, "rb")
        fh.seek(offset)
        return fh
    req = urllib.request.Request(src, headers={"Range": f"bytes={offset}-"})
    resp = urllib.request.urlopen(req, timeout=120)
    if offset and resp.status != 206:
        raise RuntimeError(f"{src}: server ignored the Range request (HTTP {resp.status})")
    return resp


def run(src: str, out: Path, retries: int = 20) -> None:
    out.mkdir(parents=True, exist_ok=True)
    state = out / ".progress.json"
    offset = json.loads(state.read_text())["offset"] if state.exists() else 0
    failures = 0
    while True:
        try:
            with open_stream(src, offset) as stream, \
                    tarfile.open(fileobj=stream, mode="r|", bufsize=1 << 20) as tar:
                for m in tar:
                    nxt = offset + m.offset_data + (m.size + 511) // 512 * 512
                    r = route(m.name) if m.isfile() else None
                    if r:
                        handle(*r, tar.extractfile(m).read(), out)
                        print(f"{r[0]:5s} {r[1]}/{r[2]}", flush=True)
                    state.write_text(json.dumps({"offset": nxt}))
                    failures = 0
            return
        except (OSError, http.client.HTTPException, tarfile.TarError, RuntimeError) as e:
            failures += 1
            if failures > retries:
                raise
            offset = json.loads(state.read_text())["offset"] if state.exists() else 0
            print(f"!!! stream error ({e}); resuming at byte {offset} in {min(60, 5 * failures)} s", flush=True)
            time.sleep(min(60, 5 * failures))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tar", default=URL, help="URL or local path of the Marigold ETH3D tar")
    ap.add_argument("--out", type=Path, default=DATA_ROOT / "eth3d")
    args = ap.parse_args()
    done = args.out / ".done"
    count = lambda: {s: len(list((args.out / s / "depth").glob("*.npy"))) for s in INDOOR + OUTDOOR}
    if not done.exists() and sum(count().values()) != N_IMAGES:
        run(args.tar, args.out)
    n = count()
    print("depth maps per scene:", n, "total", sum(n.values()), f"(expect {N_IMAGES})")
    if sum(n.values()) != N_IMAGES:
        raise SystemExit("incomplete: rerun to resume")
    done.touch()


if __name__ == "__main__":
    main()
