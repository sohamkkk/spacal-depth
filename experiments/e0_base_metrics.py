"""E0: standard depth metrics of every cached model, straight from the caches
(full evaluation crop, all valid pixels — not the PixelTable lattice).
Reported on the official test subset (comparability with papers) and on
all images.

    python -m experiments.e0_base_metrics
    python -m experiments.e0_base_metrics --datasets kitti   # add/refresh one dataset, keep the others

Rows for datasets not processed in this run are kept from the existing
per_image.csv, so datasets can be evaluated one at a time (Colab flow:
cache -> E0 -> prepare -> prune, per dataset).
"""

import argparse

import pandas as pd

from experiments.common import results_dir, to_markdown, write_spec
from src.config import CACHE_ROOT
from src.data.cache import is_pruned, list_samples, load_gt, load_pred
from src.data.protocols import PROTOCOLS, clamp_pred
from src.data.registry import DATASETS
from src.metrics.depth_metrics import all_metrics

METRIC_COLS = ["AbsRel", "RMSE", "delta1", "delta2", "delta3", "SILog"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS), choices=DATASETS)
    args = ap.parse_args()

    rows, done = [], set()
    for ds in args.datasets:
        root = CACHE_ROOT / ds
        if not root.exists():
            continue
        if is_pruned(ds):
            print(f"{ds}: cache pruned, keeping its existing E0 rows")
            continue
        for tag in sorted(p.name for p in root.iterdir() if p.is_dir() and p.name != "gt"):
            for sid in list_samples(ds, tag):
                g, p = load_gt(ds, sid), load_pred(ds, tag, sid)
                if not g["mask"].any():
                    continue
                pred = clamp_pred(p["pred"].astype("float32"), PROTOCOLS[ds])
                rows.append({"dataset": ds, "model": tag, "sample_id": sid,
                             "official_test": bool(g["official_test"]),
                             **all_metrics(pred, g["gt"], g["mask"].astype(bool)),
                             **{k: float(p[k]) for k in ("t_pred", "t_flip", "t_grad") if k in p}})
            print(f"{ds}/{tag}: done")
            done.add(ds)
    if not rows:
        print(f"no caches under {CACHE_ROOT} for {args.datasets}; E0 skipped (existing results/e0 left untouched)")
        return

    out = results_dir("e0")
    per_image = pd.DataFrame(rows)
    old_path = out / "per_image.csv"
    if old_path.exists():
        old = pd.read_csv(old_path)
        per_image = pd.concat([old[~old.dataset.isin(done)], per_image], ignore_index=True)
    per_image.to_csv(old_path, index=False)

    tables = []
    for subset, frame in (("official_test", per_image[per_image.official_test.astype(bool)]), ("all", per_image)):
        agg = frame.groupby(["dataset", "model"])[METRIC_COLS].mean()
        agg["n_images"] = frame.groupby(["dataset", "model"]).size()
        tables.append(agg.assign(subset=subset).reset_index())
    summary = pd.concat(tables)
    summary.to_csv(out / "summary.csv", index=False)
    (out / "summary.md").write_text(to_markdown(summary))
    write_spec(out, {"experiment": "E0 base metrics", "cache_root": str(CACHE_ROOT), "datasets_refreshed": sorted(done)})
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
