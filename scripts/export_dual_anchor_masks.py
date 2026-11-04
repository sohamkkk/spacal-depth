"""Export budget-split structured-anchor pixel indices for OMNI-DC (probes excluded).

Writes one npz per (pattern, seed, draw) under <out>/<dataset>/<pattern>/:
  sample_id, row, col, structured (bool per table pixel), probe (bool), k, k_random, seed, draw
and <out>/<dataset>/table_values.npz (sample_id, offsets, row, col, gt, lidar) for the anchor depths.

GPU code should set sparse depth only where structured is True; leave probe pixels unknown.

    SPACAL_PREPARED_ROOT=prepared_new python scripts/export_dual_anchor_masks.py \
        --dataset arkitscenes --model unidepthv2-large --patterns uniform cluster \
        --ks 200 --n-seeds 1 --draws 1 --out .scratch/omni_anchors
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from experiments.common import load_table, split_tables
from experiments.e11_anchors import log_err
from src.calibrators.anchors import PATTERNS, anchor_keys, draw_split_anchors
from src.calibrators.dual import split_roles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", required=True, help="any table for this dataset (positions are model-agnostic)")
    ap.add_argument("--patterns", nargs="+", default=["uniform", "cluster"], choices=PATTERNS)
    ap.add_argument("--ks", type=int, nargs="+", default=[200])
    ap.add_argument("--dist-k-random", type=int, default=20)
    ap.add_argument("--n-seeds", type=int, default=1)
    ap.add_argument("--draws", type=int, default=1)
    ap.add_argument("--out", type=Path, default=Path(".scratch/omni_anchors"))
    args = ap.parse_args()

    table = load_table(args.dataset, args.model)
    table.pix.pop("label", None)
    root = args.out / args.dataset
    root.mkdir(parents=True, exist_ok=True)
    values = {"gt": table.pix["gt"].astype(np.float32)}
    if table.has("lidar"):
        values["lidar"] = table.pix["lidar"].astype(np.float32)
    np.savez_compressed(root / "table_values.npz", sample_id=table.img["sample_id"], offsets=table.offsets,
                        row=table.pix["row"], col=table.pix["col"], **values)
    print(f"wrote {root / 'table_values.npz'}  fields={sorted(values)}", flush=True)

    for pattern in args.patterns:
        for seed in range(args.n_seeds):
            head, cal, test = split_tables(table, seed)
            # masks needed for every image the completion model will see; export the full table order
            d = log_err(table)
            for k in args.ks:
                for draw_i in range(args.draws):
                    rng = np.random.default_rng([seed, int(k) + 1, draw_i, 99])
                    key = None if pattern == "uniform" else anchor_keys(pattern, table.pix, table.offsets, rng)
                    draw = draw_split_anchors(d, table.offsets, k, args.dist_k_random, rng, key)
                    pred_draw, probes = split_roles(draw, d, table.offsets)
                    path = root / pattern / f"k{k}_seed{seed}_draw{draw_i}.npz"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(
                        path,
                        sample_id=table.img["sample_id"],
                        offsets=table.offsets,
                        row=table.pix["row"],
                        col=table.pix["col"],
                        structured=pred_draw.anchor,
                        probe=probes,
                        k=np.int32(k),
                        k_random=np.int32(args.dist_k_random),
                        seed=np.int32(seed),
                        draw=np.int32(draw_i),
                        pattern=np.array(pattern),
                    )
                    n_s = int(pred_draw.anchor.sum())
                    n_p = int(probes.sum())
                    print(f"wrote {path}  structured={n_s:,} probes={n_p:,}", flush=True)


if __name__ == "__main__":
    main()
