import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.splits import scene_level_split, assign_split


def test_split_is_scene_level_not_frame_level():
    # 3 scenes, uneven frame counts per scene -- frames must never cross pools
    scene_ids = ["a"] * 10 + ["b"] * 3 + ["c"] * 7
    split = scene_level_split(scene_ids, frac_head_train=0.0, frac_calibration=0.34, seed=1)
    all_scenes = split["calibration"] | split["test"]
    assert all_scenes == {"a", "b", "c"}
    assert split["calibration"].isdisjoint(split["test"])
    # every frame from one scene must land in exactly one pool
    for s in ("a", "b", "c"):
        assert assign_split(s, split) in ("calibration", "test")


def test_three_way_split_partitions_fully():
    scene_ids = [f"scene_{i}" for i in range(20)]
    split = scene_level_split(scene_ids, frac_head_train=0.2, frac_calibration=0.4, seed=0)
    union = split["head_train"] | split["calibration"] | split["test"]
    assert union == set(scene_ids)
    assert split["head_train"].isdisjoint(split["calibration"])
    assert split["calibration"].isdisjoint(split["test"])
    assert split["head_train"].isdisjoint(split["test"])


def test_unknown_scene_raises():
    split = scene_level_split(["a", "b"], frac_calibration=0.5, seed=0)
    try:
        assign_split("not_a_scene", split)
        assert False, "expected KeyError"
    except KeyError:
        pass


def _img_table(dataset, scene_ids, sample_ids):
    import numpy as np
    from src.data.prepared import PixelTable

    n = len(scene_ids)
    return PixelTable(
        pix={"pred": np.ones(n, np.float32), "gt": np.ones(n, np.float32)},
        img={"scene_id": np.array(scene_ids), "sample_id": np.array(sample_ids)},
        offsets=np.arange(n + 1),
        meta={"dataset": dataset},
    )


def test_exchange_units_scene_by_default_scan_for_diode():
    from experiments.common import exchange_units

    nyu = _img_table("nyu", ["kitchen_0001", "kitchen_0001"], ["nyu_00000", "nyu_00001"])
    assert list(exchange_units(nyu)) == ["kitchen_0001", "kitchen_0001"]
    diode = _img_table(
        "diode_indoor",
        ["scene_00019", "scene_00019", "scene_00020"],
        ["diode_indoor_00019_00183_indoors_000_010", "diode_indoor_00019_00184_indoors_090_000",
         "diode_indoor_00020_00190_indoors_000_020"],
    )
    assert list(exchange_units(diode)) == ["scene_00019/scan_00183", "scene_00019/scan_00184",
                                           "scene_00020/scan_00190"]


if __name__ == "__main__":
    test_split_is_scene_level_not_frame_level()
    test_three_way_split_partitions_fully()
    test_unknown_scene_raises()
    print("splits: all checks passed")
