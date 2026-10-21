"""Shared plumbing for experiment scripts: table I/O, scene-level splits,
result directories and run specs."""

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PREPARED_ROOT, REPO_ROOT, RESULTS_ROOT
from src.data.prepared import PixelTable
from src.data.splits import exchange_units, scene_level_split  # noqa: F401  (exchange_units re-exported)

DEFAULT_STRIDE = {"nyu": 4, "kitti": 2, "diode_indoor": 2, "diode_outdoor": 2}


def table_path(dataset: str, model: str) -> Path:
    return PREPARED_ROOT / f"{dataset}__{model}.npz"


def load_table(dataset: str, model: str) -> PixelTable:
    path = table_path(dataset, model)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run `python -m experiments.prepare_tables` first")
    return PixelTable.load(path)


def split_tables(table: PixelTable, seed: int, frac_head: float = 0.3, frac_cal: float = 0.35):
    """Unit-level head-train / calibration / test pools (no scene, or DIODE scan, in two pools)."""
    scenes = exchange_units(table)
    pools = scene_level_split(list(scenes), frac_head, frac_cal, seed)
    out = []
    for name in ("head_train", "calibration", "test"):
        idx = np.flatnonzero(np.isin(scenes, list(pools[name])))
        if len(idx) == 0:
            raise ValueError(f"empty {name} pool (seed {seed}); too few scenes for these fractions")
        out.append(table.subset(idx))
    return tuple(out)


def subsample_scenes(table: PixelTable, n_images: int, rng: np.random.Generator) -> PixelTable:
    """Whole scenes in random order until >= n_images images (keeps scene exchangeability)."""
    units = exchange_units(table)
    scenes = np.unique(units)
    rng.shuffle(scenes)
    chosen, total = [], 0
    counts = pd.Series(units).value_counts()
    for s in scenes:
        chosen.append(s)
        total += counts[s]
        if total >= n_images:
            break
    return table.subset(np.flatnonzero(np.isin(units, chosen)))


def results_dir(exp_id: str, *parts: str) -> Path:
    d = RESULTS_ROOT.joinpath(exp_id, *parts)
    d.mkdir(parents=True, exist_ok=True)
    return d


def git_rev() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=5
        ).stdout.strip() or "uncommitted"
    except Exception:
        return "unknown"


def code_hash() -> str:
    """sha256 over the code that produces results (the cluster copy is rsynced without .git)."""
    h = hashlib.sha256()
    for path in sorted(p for d in ("src", "experiments", "scripts") for p in (REPO_ROOT / d).rglob("*")
                       if p.suffix in (".py", ".sh") and "__pycache__" not in p.parts):
        h.update(str(path.relative_to(REPO_ROOT)).encode())
        h.update(path.read_bytes())
    return h.hexdigest()[:16]


def write_spec(out_dir: Path, spec: dict, name: str = "spec.json") -> None:
    spec = {
        **spec,
        "written": datetime.now(timezone.utc).isoformat(),
        "git": git_rev(),
        "code_sha256": code_hash(),
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    (out_dir / name).write_text(json.dumps(spec, indent=2, default=str))


def to_markdown(df: pd.DataFrame, digits: int = 4) -> str:
    """Pipe table without the optional `tabulate` dependency."""
    def fmt(v):
        if isinstance(v, float):
            return "inf" if np.isinf(v) else ("" if np.isnan(v) else f"{v:.{digits}g}")
        return str(v)

    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def tag_family(model: str) -> str:
    return "dav2" if model.startswith("dav2") else model.split("-")[0]
