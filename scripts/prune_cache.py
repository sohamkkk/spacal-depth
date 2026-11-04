"""Free disk after a dataset is fully processed (E0 done, tables prepared):
delete cached arrays except the samples that carry an RGB crop (every
20th sample), which experiments/qualitative.py needs for figures.

    python -m scripts.prune_cache --dataset nyu            # dry run: shows what would go
    python -m scripts.prune_cache --dataset nyu --yes
"""

import argparse

import numpy as np

from experiments.common import table_path
from src.config import CACHE_ROOT, RESULTS_ROOT
from src.data.cache import PRUNED_MARKER
from src.data.registry import DATASETS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=DATASETS, required=True)
    ap.add_argument("--yes", action="store_true", help="actually delete")
    args = ap.parse_args()

    root = CACHE_ROOT / args.dataset
    tags = sorted(p.name for p in root.iterdir() if p.is_dir() and p.name != "gt" and not p.name.startswith("."))
    missing = [t for t in tags if not table_path(args.dataset, t).exists()]
    e0 = RESULTS_ROOT / "e0" / "per_image.csv"
    if missing:
        raise SystemExit(f"refusing: no prepared table yet for {missing}; run experiments.prepare_tables first")
    if not e0.exists() or args.dataset not in open(e0).read():
        raise SystemExit(f"refusing: E0 has no rows for {args.dataset}; run experiments.e0_base_metrics first")

    keep, drop = [], []
    for gt_file in sorted((root / "gt").glob("*.npz")):
        with np.load(gt_file) as d:
            (keep if "rgb" in d.files else drop).append(gt_file.stem)
    files = [root / "gt" / f"{s}.npz" for s in drop] + [root / t / f"{s}.npz" for t in tags for s in drop]
    files = [f for f in files if f.exists()]
    size = sum(f.stat().st_size for f in files) / 1e9
    print(f"{args.dataset}: keep {len(keep)} samples with RGB, delete {len(drop)} samples ({len(files)} files, {size:.2f} GB)")
    if not args.yes:
        print("dry run — add --yes to delete")
        return
    for f in files:
        f.unlink()
    (root / PRUNED_MARKER).write_text(f"pruned; kept {len(keep)} RGB samples for figures\n")
    print(f"deleted; wrote {root / PRUNED_MARKER} (E0 / prepare_tables now skip this dataset)")


if __name__ == "__main__":
    main()
