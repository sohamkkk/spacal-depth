"""Decodes a tiny file in the MATLAB v7.3 (HDF5) layout of nyu_depth_v2_labeled.mat."""

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
from scipy.io import savemat  # noqa: E402


def test_nyu_layout_decoding(tmp_path):
    from src.data.nyu import NYUDepthV2

    n = 3
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 255, (n, 480, 640, 3), dtype=np.uint8)
    depth = rng.uniform(0.5, 9.0, (n, 480, 640)).astype(np.float32)
    with h5py.File(tmp_path / "nyu_depth_v2_labeled.mat", "w") as f:
        f["images"] = rgb.transpose(0, 3, 2, 1)       # MATLAB stores (N, 3, W, H)
        f["rawDepths"] = depth.transpose(0, 2, 1)     # (N, W, H)
        f["labels"] = np.zeros((n, 640, 480), np.uint16)
        refs = []
        for i, name in enumerate(["kitchen_0001", "kitchen_0001", "office_0002"]):
            ds = f.create_dataset(f"#refs#/s{i}", data=np.array([ord(c) for c in name], np.uint16)[:, None])
            refs.append(ds.ref)
        f.create_dataset("scenes", data=np.array([refs], dtype=h5py.ref_dtype))
    savemat(tmp_path / "splits.mat", {"testNdxs": np.array([[3]]), "trainNdxs": np.array([[1], [2]])})

    ds = NYUDepthV2(root=tmp_path, expected_test=1)
    assert len(ds) == 3
    assert ds.all_scene_ids() == ["kitchen_0001", "kitchen_0001", "office_0002"]
    s = ds[2]
    assert s["rgb"].shape == (480, 640, 3) and np.array_equal(s["rgb"], rgb[2])
    assert np.allclose(s["gt"], depth[2])
    assert s["meta"]["official_test"] and not ds[0]["meta"]["official_test"]
