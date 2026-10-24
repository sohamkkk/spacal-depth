import io
import json
import tarfile

import numpy as np
from PIL import Image

from scripts.prepare_eth3d import route, run
from src.data.eth3d import DOWNSAMPLE, ETH3D, downsample_depth, read_colmap_focals

H, W = 16, 24  # full resolution of the synthetic images


def _cameras(fx=3000.0):
    return ("# Camera list with one line of data per camera:\n"
            f"0 THIN_PRISM_FISHEYE {W} {H} {fx} {fx} {W / 2} {H / 2} 0.2 0.2 0 0 0 0 0 0\n"
            f"1 THIN_PRISM_FISHEYE {W} {H} {fx + 100} {fx + 100} {W / 2} {H / 2} 0.2 0.2 0 0 0 0 0 0\n")


def _images(names):
    lines = ["# Image list with two lines of data per image:"]
    for i, n in enumerate(names):
        lines += [f"{i} 1 0 0 0 0 0 0 {i % 2} dslr_images/{n}", "1.0 2.0 -1"]
    return "\n".join(lines) + "\n"


def _depth(i):
    gt = np.arange(H * W, dtype=np.float32).reshape(H, W) + i
    gt[0, 0] = np.inf
    return gt


def _mirror_tar(path, scenes):
    """A tiny tar with the member layout of the Marigold ETH3D copy (images first, then depth)."""
    with tarfile.open(path, "w") as tar:
        def add(name, data):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        for scene, names in scenes.items():
            add(f"./rgb/{scene}/dslr_calibration_jpg/cameras.txt", _cameras().encode())
            add(f"./rgb/{scene}/dslr_calibration_jpg/images.txt", _images(names).encode())
            add(f"./rgb/{scene}/dslr_calibration_jpg/points3D.txt", b"ignored")
            for i, n in enumerate(names):
                buf = io.BytesIO()
                Image.fromarray(np.full((H, W, 3), 10 * i, np.uint8)).save(buf, format="JPEG")
                add(f"./rgb/{scene}/images/dslr_images/{n}", buf.getvalue())
        for scene, names in scenes.items():
            for i, n in enumerate(names):
                add(f"./depth/{scene}_dslr_depth/{scene}/ground_truth_depth/dslr_images/{n}", _depth(i).tobytes())


def test_route_maps_mirror_members():
    assert route("./rgb/office/dslr_calibration_jpg/images.txt") == ("calib", "office", "images.txt")
    assert route("./rgb/office/dslr_calibration_jpg/points3D.txt") is None
    assert route("./rgb/office/images/dslr_images/DSC_0001.JPG") == ("rgb", "office", "DSC_0001.JPG")
    assert route("./depth/relief_2_dslr_depth/relief_2/ground_truth_depth/dslr_images/DSC_0458.JPG") == (
        "depth", "relief_2", "DSC_0458.JPG")


def test_colmap_focals_follow_camera_ids(tmp_path):
    (tmp_path / "cameras.txt").write_text(_cameras())
    (tmp_path / "images.txt").write_text(_images(["DSC_0001.JPG", "DSC_0002.JPG"]))
    assert read_colmap_focals(tmp_path) == {"DSC_0001": 3000.0, "DSC_0002": 3100.0}


def test_downsample_depth_takes_a_measured_value_per_block():
    d = np.full((4, 8), np.inf, np.float32)
    d[0, 0], d[1, 1], d[2, 2] = 1.0, 9.0, 2.0  # left block: an edge between 1-2 m and 9 m
    out = downsample_depth(d, 4)
    assert out.shape == (1, 2) and out[0, 0] == 2.0 and np.isinf(out[0, 1])


def test_prepare_then_load(tmp_path):
    _mirror_tar(tmp_path / "eth3d.tar", {"office": ["DSC_0001.JPG", "DSC_0002.JPG"], "facade": ["DSC_0100.JPG"]})
    out = tmp_path / "eth3d"
    run(str(tmp_path / "eth3d.tar"), out)
    ds = ETH3D(root=out, scenes=("office", "facade"))
    assert len(ds) == 3 and ds.all_scene_ids() == ["office", "office", "facade"]
    s = ds[1]
    f = DOWNSAMPLE
    assert s["rgb"].shape == (H // f, W // f, 3) and s["gt"].shape == (H // f, W // f)
    np.testing.assert_array_equal(s["gt"], downsample_depth(_depth(1)))
    assert s["meta"]["focal"] == 3100.0 / f and s["meta"]["domain"] == "indoor"
    assert s["sample_id"] == "eth3d_office_DSC_0002" and ds[2]["meta"]["domain"] == "outdoor"


def test_prepare_resumes_from_saved_offset(tmp_path):
    _mirror_tar(tmp_path / "eth3d.tar", {"office": ["DSC_0001.JPG", "DSC_0002.JPG"]})
    out = tmp_path / "eth3d"
    run(str(tmp_path / "eth3d.tar"), out)
    offsets = {}
    with tarfile.open(tmp_path / "eth3d.tar") as tar:
        for m in tar:
            offsets[m.name] = m.offset
    resume_at = offsets["./depth/office_dslr_depth/office/ground_truth_depth/dslr_images/DSC_0002.JPG"]
    (out / "office" / "depth" / "DSC_0002.npy").unlink()
    (out / ".progress.json").write_text(json.dumps({"offset": resume_at}))
    run(str(tmp_path / "eth3d.tar"), out)
    np.testing.assert_array_equal(np.load(out / "office" / "depth" / "DSC_0002.npy"), downsample_depth(_depth(1)))
