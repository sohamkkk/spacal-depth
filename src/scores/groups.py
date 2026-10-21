"""Per-pixel group labels.

CALIBRATION groups (allowed for Mondrian K3) must be functions of x only:
predicted-depth bins and predicted-depth edges.
ANALYSIS groups may use y or labels — they are only for reporting
conditional coverage, never for choosing lambda.
"""

import numpy as np

from src.data.prepared import PixelTable
from src.data.protocols import DOMAIN
from src.data.protocols import NYU_GLASS_IDS as GLASS_IDS
from src.data.protocols import NYU_MIRROR_IDS as MIRROR_IDS
from src.data.protocols import NYU_WINDOW_IDS as WINDOW_IDS

DEPTH_EDGES = {
    "indoor": np.array([1.5, 3.0, 5.0]),   # bins: <1.5, 1.5-3, 3-5, >=5 m
    "outdoor": np.array([10.0, 20.0, 40.0]),  # bins: <10, 10-20, 20-40, >=40 m
}


def depth_bins(depth: np.ndarray, dataset: str) -> tuple[np.ndarray, list[str]]:
    edges = DEPTH_EDGES[DOMAIN[dataset]]
    names = [f"<{edges[0]:g}"] + [f"{a:g}-{b:g}" for a, b in zip(edges[:-1], edges[1:])] + [f">={edges[-1]:g}"]
    return np.digitize(depth, edges).astype(np.int64), names


def calibration_groups(table: PixelTable, kind: str) -> tuple[np.ndarray, list[str]]:
    """x-only groups usable by K3."""
    if kind == "pred_depth":
        return depth_bins(table.pix["pred"], table.meta["dataset"])
    if kind == "edge":
        return table.pix["edge"].astype(np.int64), ["smooth", "edge"]
    raise ValueError(f"unknown calibration group kind {kind!r}")


def analysis_groups(table: PixelTable) -> dict[str, tuple[np.ndarray, list[str]]]:
    """name -> (labels in [0, n) or -1 for 'not in any group', names)."""
    out = {
        "pred_depth": calibration_groups(table, "pred_depth"),
        "gt_depth": depth_bins(table.pix["gt"], table.meta["dataset"]),
        "edge": calibration_groups(table, "edge"),
    }
    if table.meta["dataset"] == "nyu":
        lab = table.pix["label"]
        sem = np.full(len(lab), -1, dtype=np.int64)
        sem[np.isin(lab, MIRROR_IDS)] = 0
        sem[np.isin(lab, WINDOW_IDS)] = 1
        sem[np.isin(lab, GLASS_IDS)] = 2
        out["reflective"] = (sem, ["mirror", "window", "glass"])
    return out
