"""Build PixelTables (prepared/<dataset>__<model>.npz) from caches. CPU only.

    python -m experiments.prepare_tables --all
    python -m experiments.prepare_tables --dataset nyu --model dav2-indoor-large --ensemble dav2-indoor-base dav2-indoor-small

With --all, every (dataset, model) pair found in the cache is prepared;
DA-V2 Large gets the S/B/L size ensemble automatically when Base and
Small caches exist.
"""

import argparse
import time

from experiments.common import DEFAULT_STRIDE, table_path
from src.config import CACHE_ROOT
from src.data.cache import is_pruned
from src.data.prepared import build_table
from src.data.registry import DATASETS


def discover(datasets=DATASETS) -> list[tuple[str, str, tuple[str, ...]]]:
    jobs = []
    for ds in datasets:
        root = CACHE_ROOT / ds
        if not root.exists() or is_pruned(ds):
            continue
        tags = sorted(p.name for p in root.iterdir() if p.is_dir() and p.name != "gt")
        for tag in tags:
            ens: tuple[str, ...] = ()
            if tag.startswith("dav2-") and tag.endswith("-large"):
                stem = tag[: -len("-large")]
                ens = tuple(t for t in (f"{stem}-base", f"{stem}-small") if t in tags)
            jobs.append((ds, tag, ens))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="every (dataset, model) pair found in the cache")
    ap.add_argument("--datasets", nargs="*", choices=DATASETS, default=list(DATASETS),
                    help="with --all: restrict discovery to these datasets (avoid half-cached ones)")
    ap.add_argument("--dataset", choices=DATASETS)
    ap.add_argument("--model")
    ap.add_argument("--ensemble", nargs="*", default=[])
    ap.add_argument("--stride", type=int, default=0, help="lattice stride (default per dataset)")
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.all:
        jobs = discover(args.datasets)
    elif args.dataset and args.model:
        jobs = [(args.dataset, args.model, tuple(args.ensemble))]
    else:
        ap.error("give --all or --dataset and --model")
    if not jobs:
        print(f"nothing cached under {CACHE_ROOT}; using whatever tables already exist in prepared/")
        return

    for ds, tag, ens in jobs:
        out = table_path(ds, tag)
        if out.exists() and not args.force:
            print(f"skip {out.name} (exists; --force to rebuild)")
            continue
        t0 = time.perf_counter()
        table = build_table(ds, tag, ens, stride=args.stride or DEFAULT_STRIDE[ds], max_images=args.max_images)
        table.save(out)
        print(
            f"{out.name}: {table.n_images} images ({table.meta['dropped_images']} dropped), "
            f"{table.n_pixels:,} pixels, fields={sorted(table.pix)}  [{time.perf_counter() - t0:.0f}s]"
        )


if __name__ == "__main__":
    main()
