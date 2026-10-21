import numpy as np
import pytest

from src.data.prepared import PixelTable, build_table
from src.data.synthetic import write_synthetic_cache


@pytest.fixture(scope="module")
def table(tmp_path_factory):
    root = tmp_path_factory.mktemp("cache")
    write_synthetic_cache(root, "nyu", n_scenes=8, imgs_per_scene=3, h=40, w=48)
    t = build_table("nyu", "dav2-indoor-large", ("dav2-indoor-base", "dav2-indoor-small"),
                    stride=2, min_valid=50, cache_root=root)
    return t, root


def test_table_layout(table):
    t, _ = table
    assert t.n_images == 24
    assert t.offsets[0] == 0 and t.offsets[-1] == t.n_pixels
    assert np.all(t.counts == t.img["n_valid"])
    for f in ("pred", "gt", "row", "col", "dflip", "conf", "grad_unc", "ens_std", "edge", "grad_logpred"):
        assert len(t.pix[f]) == t.n_pixels, f
    assert np.all(t.pix["row"] % 2 == 0) and np.all(t.pix["col"] % 2 == 0)  # lattice
    assert np.all(t.pix["gt"] > 0)
    assert len(np.unique(t.img["scene_id"])) == 8


def test_subset_is_contiguous_and_consistent(table):
    t, _ = table
    s = t.subset(np.array([5, 1, 7]))
    assert s.img["sample_id"].tolist() == [t.img["sample_id"][i] for i in (1, 5, 7)]
    i = 1  # second image of the subset = original image 5
    assert np.array_equal(s.pix["gt"][s.offsets[i] : s.offsets[i + 1]], t.pix["gt"][t.offsets[5] : t.offsets[6]])
    assert np.array_equal(np.bincount(s.image_of_pixel()), s.counts)


def test_save_load_roundtrip(table, tmp_path):
    t, _ = table
    p = tmp_path / "t.npz"
    t.save(p)
    u = PixelTable.load(p)
    assert u.meta == t.meta
    assert np.array_equal(u.offsets, t.offsets)
    for k in t.pix:
        assert np.array_equal(u.pix[k], t.pix[k])
    for k in t.img:
        assert np.array_equal(u.img[k], t.img[k])


def test_field_missing_in_some_images_is_dropped(tmp_path):
    write_synthetic_cache(tmp_path, "nyu", n_scenes=4, imgs_per_scene=2, h=40, w=48)
    f = next((tmp_path / "nyu" / "dav2-indoor-large").glob("*.npz"))
    with np.load(f) as d:
        payload = {k: d[k] for k in d.files if k != "grad_unc"}  # e.g. gradient pass OOM on this image
    np.savez_compressed(f, **payload)
    t = build_table("nyu", "dav2-indoor-large", stride=2, min_valid=50, cache_root=tmp_path)
    assert "grad_unc" not in t.pix and "med_grad_unc" not in t.img
    assert t.meta["dropped_fields"] == ["grad_unc"]
    assert all(len(v) == t.n_pixels for v in t.pix.values())
    assert all(len(v) == t.n_images for v in t.img.values())


def test_features_are_label_free(table):
    """x-only features must not change when GT changes."""
    t, root = table
    gt_file = next((root / "nyu" / "gt").glob("*.npz"))
    with np.load(gt_file) as d:
        payload = {k: d[k] for k in d.files}
    payload["gt"] = payload["gt"] * 1.7
    np.savez_compressed(gt_file, **payload)
    t2 = build_table("nyu", "dav2-indoor-large", ("dav2-indoor-base", "dav2-indoor-small"),
                     stride=2, min_valid=50, cache_root=root)
    for k in ("pred", "dflip", "grad_logpred", "local_std", "edge", "ens_std"):
        assert np.array_equal(t.pix[k], t2.pix[k]), k
    for k in t.img:
        if k.startswith(("med_", "p90_", "std_")):
            assert np.allclose(t.img[k], t2.img[k]), k
