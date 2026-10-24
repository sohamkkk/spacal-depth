"""Reads the caches written by scripts/cache_predictions.py. Pure numpy —
runs on any CPU (laptop or the remote box)."""

import json
from pathlib import Path

import numpy as np

from src.config import CACHE_ROOT

PRUNED_MARKER = "PRUNED"  # written by scripts/prune_cache.py: only RGB samples remain


def is_pruned(dataset: str, cache_root: Path | None = None) -> bool:
    return (_root(dataset, cache_root) / PRUNED_MARKER).exists()


def _root(dataset: str, cache_root: Path | None) -> Path:
    return (cache_root or CACHE_ROOT) / dataset


def list_samples(dataset: str, model_tag: str, cache_root: Path | None = None) -> list[str]:
    """Sample ids that have both GT and a prediction from `model_tag`."""
    root = _root(dataset, cache_root)
    gt_ids = {p.stem for p in (root / "gt").glob("*.npz")}
    pred_ids = {p.stem for p in (root / model_tag).glob("*.npz")}
    ids = sorted(gt_ids & pred_ids)
    if not ids:
        raise FileNotFoundError(
            f"no cached samples for {dataset}/{model_tag} under {root} — run scripts/cache_predictions.py first"
        )
    return ids


def load_gt(dataset: str, sample_id: str, cache_root: Path | None = None) -> dict:
    with np.load(_root(dataset, cache_root) / "gt" / f"{sample_id}.npz") as d:
        return {k: d[k] for k in d.files}


def load_pred(dataset: str, model_tag: str, sample_id: str, cache_root: Path | None = None) -> dict:
    with np.load(_root(dataset, cache_root) / model_tag / f"{sample_id}.npz") as d:
        return {k: d[k] for k in d.files}


def load_manifest(dataset: str, model_tag: str, cache_root: Path | None = None) -> dict:
    return json.loads((_root(dataset, cache_root) / model_tag / "manifest.json").read_text())
