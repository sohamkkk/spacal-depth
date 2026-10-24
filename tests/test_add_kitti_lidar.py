import io
import zipfile

import numpy as np
from PIL import Image

from scripts.add_kitti_lidar import add_lidar, index_zip
from src.data.prepared import PixelTable
from src.data.protocols import PROTOCOLS, crop, kitti_benchmark_crop

DRIVE = "2011_09_26_drive_0002_sync"


def _png(depth_m: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.round(depth_m * 256).astype(np.uint16)).save(buf, format="PNG")
    return buf.getvalue()


def test_raw_scan_is_read_at_table_pixels_and_wrong_frames_are_rejected(tmp_path):
    rng = np.random.default_rng(0)
    full_shape = (375, 1242)
    frames = ["0000000010", "0000000011"]
    raws, pix, offsets = {}, {k: [] for k in ("gt", "pred", "row", "col")}, [0]
    for j, f in enumerate(frames):
        gt_full = rng.uniform(2, 60, full_shape).astype(np.float32)
        raw_full = np.where(rng.random(full_shape) < 0.3, gt_full, 0).astype(np.float32)
        if j == 1:
            raw_full *= 2.0                                     # disagrees with the GT: must be rejected
        raws[f] = raw_full
        gt = crop(kitti_benchmark_crop(gt_full), PROTOCOLS["kitti"])
        rows, cols = np.nonzero(np.ones_like(gt[::2, ::2], bool))
        rows, cols = rows * 2, cols * 2
        pix["gt"].append(gt[rows, cols]), pix["pred"].append(gt[rows, cols])
        pix["row"].append(rows.astype(np.uint16)), pix["col"].append(cols.astype(np.uint16))
        offsets.append(offsets[-1] + len(rows))
    t = PixelTable(pix={k: np.concatenate(v) for k, v in pix.items()},
                   img={"sample_id": np.array([f"kitti_val800_{DRIVE}_{f}" for f in frames])},
                   offsets=np.array(offsets), meta={"stride": 2})

    zpath = tmp_path / "data_depth_velodyne.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        for f, raw in raws.items():
            zf.writestr(f"train/{DRIVE}/proj_depth/velodyne_raw/image_02/{f}.png", _png(raw))
    with zipfile.ZipFile(zpath) as zf:
        lidar, stats = add_lidar(t, zf, index_zip(zf))

    sl0 = slice(t.offsets[0], t.offsets[1])
    expect = crop(kitti_benchmark_crop(np.round(raws[frames[0]] * 256) / 256), PROTOCOLS["kitti"])
    expect = expect[t.pix["row"][sl0], t.pix["col"][sl0]]
    assert np.allclose(lidar[sl0], np.where(expect < 80, expect, 0), atol=1e-6)
    assert 0.25 < np.mean(lidar[sl0] > 0) < 0.35
    assert (lidar[t.offsets[1]:] == 0).all()
    assert stats["n_missing"] == 1 and stats["cameras"] == {"image_02": 1}
    assert stats["frac_lattice_returns_without_gt"] < 0.01 and stats["err_median_abs_log"] < 0.01
