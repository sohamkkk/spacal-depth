"""Assemble scripts/omnidc_forward.py chunks into one prediction array per variant, in table order. CPU only.

    python -m scripts.omnidc_merge --anchors .scratch/omni_anchors/arkitscenes \
        --pred-dir <forward --out> --out results/omnidc/arkitscenes

Writes <out>/<variant>.npz with pred (float32 per table pixel; NaN where the variant had no anchor),
n_sparse (per image), sample_id, offsets. Fails if any table image is missing or duplicated.
"""

import argparse
import re
from pathlib import Path

import numpy as np

CHUNK = re.compile(r"^s\d+of\d+_c\d+\.npz$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchors", type=Path, required=True)
    ap.add_argument("--pred-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    with np.load(args.anchors / "table_values.npz") as v:
        sids, offsets = v["sample_id"], v["offsets"]
    pos = {s: i for i, s in enumerate(sids)}
    files = sorted(p for p in args.pred_dir.iterdir() if CHUNK.match(p.name))
    if not files:
        raise SystemExit(f"no chunks in {args.pred_dir}")

    names, pred, n_sparse, seen = None, {}, {}, np.zeros(len(sids), np.int32)
    for f in files:
        with np.load(f) as z:
            these = sorted(k[len("pred__"):] for k in z.files if k.startswith("pred__"))
            if names is None:
                names = these
                pred = {n: np.full(int(offsets[-1]), np.nan, np.float32) for n in names}
                n_sparse = {n: np.zeros(len(sids), np.int32) for n in names}
            elif these != names:
                raise ValueError(f"{f.name}: variants {these} differ from {names}")
            idx = np.array([pos[s] for s in z["sample_id"]])
            if not np.array_equal(z["counts"], np.diff(offsets)[idx]):
                raise ValueError(f"{f.name}: pixel counts differ from the table")
            seen[idx] += 1
            starts = np.concatenate([[0], np.cumsum(z["counts"])])
            for n in names:
                p = z[f"pred__{n}"]
                for j, i in enumerate(idx):
                    pred[n][offsets[i] : offsets[i + 1]] = p[starts[j] : starts[j + 1]]
                n_sparse[n][idx] = z[f"n_sparse__{n}"]
    if (seen != 1).any():
        raise SystemExit(f"{int((seen == 0).sum())} images missing, {int((seen > 1).sum())} duplicated")

    args.out.mkdir(parents=True, exist_ok=True)
    for n in names:
        np.savez(args.out / f"{n}.npz", pred=pred[n], n_sparse=n_sparse[n], sample_id=sids, offsets=offsets)
        no_anchor = int((n_sparse[n] == 0).sum())
        print(f"{n}: {np.isfinite(pred[n]).mean():.4%} pixels predicted, {no_anchor} images without anchors, "
              f"median anchors {int(np.median(n_sparse[n]))}")


if __name__ == "__main__":
    main()
