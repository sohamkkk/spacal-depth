"""Scene-level split (section 8.1): adjacent frames are near-duplicates,
so splitting by frame index silently breaks exchangeability between
calibration and test. Always split by unique scene id instead.
"""

import re

import numpy as np

_DIODE_SCAN = re.compile(r"(\d{5})_(\d{5})_(?:indoors|outdoor)_")


def exchange_units(table) -> np.ndarray:
    """Per-image exchangeable unit of a PixelTable: the scene, except for
    DIODE val, which has only ~3 scenes per domain; there the scan
    (scene_XXXXX/scan_XXXXX, encoded in the file stem) is used, a weaker
    exchangeability assumption."""
    scenes = np.asarray(table.img["scene_id"]).astype(str)
    if not str(table.meta.get("dataset", "")).startswith("diode"):
        return scenes
    sample_ids = np.asarray(table.img["sample_id"]).astype(str)
    matches = [_DIODE_SCAN.search(s) for s in sample_ids]
    if not any(matches):  # not the official file naming (e.g. synthetic caches)
        return scenes
    scans = []
    for sid, scene, m in zip(sample_ids, scenes, matches):
        if m is None or not scene.endswith(m.group(1)):
            raise ValueError(f"cannot parse DIODE scan from sample_id {sid!r} (scene {scene!r})")
        scans.append(f"{scene}/scan_{m.group(2)}")
    return np.asarray(scans)


def scene_level_split(
    scene_ids: list[str],
    frac_head_train: float = 0.0,
    frac_calibration: float = 0.5,
    seed: int = 0,
) -> dict[str, set[str]]:
    """Partitions UNIQUE scenes (not frames) into head-train/calibration/test
    pools. frac_head_train=0 skips that pool (no S5 head to train)."""
    unique_scenes = sorted(set(scene_ids))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_scenes)

    n = len(unique_scenes)
    n_head = int(n * frac_head_train)
    n_cal = int(n * frac_calibration)

    return {
        "head_train": set(unique_scenes[:n_head]),
        "calibration": set(unique_scenes[n_head : n_head + n_cal]),
        "test": set(unique_scenes[n_head + n_cal :]),
    }


def assign_split(scene_id: str, split_map: dict[str, set[str]]) -> str:
    for pool, scenes in split_map.items():
        if scene_id in scenes:
            return pool
    raise KeyError(f"scene_id {scene_id!r} not found in any split pool")
