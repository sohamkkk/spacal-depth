"""Sanity report on the prepared tables (run first in run_all_cpu.sh):
split units per dataset (DIODE scans), and the scale of UniDepthV2's native
confidence against the actual |log error|, pooled and within-image.

    python -m experiments.diag_tables
"""

import numpy as np
from scipy.stats import spearmanr

from experiments.common import load_table, table_path
from src.data.splits import exchange_units

PAIRS = [("nyu", "dav2-indoor-large"), ("kitti", "dav2-outdoor-large"),
         ("diode_indoor", "dav2-indoor-large"), ("diode_outdoor", "dav2-outdoor-large"),
         ("nyu", "unidepthv2-large"), ("kitti", "unidepthv2-large"),
         ("diode_indoor", "unidepthv2-large"), ("diode_outdoor", "unidepthv2-large")]
Q = [1, 10, 50, 90, 99]


def main():
    rng = np.random.default_rng(0)
    for ds, model in PAIRS:
        if not table_path(ds, model).exists():
            print(f"{ds}/{model}: no table")
            continue
        t = load_table(ds, model)
        units = exchange_units(t)
        print(f"\n{ds}/{model}: {t.n_images} images, {len(set(t.img['scene_id']))} scenes, "
              f"{len(set(units))} split units, {t.n_pixels:,} px")
        if not t.has("conf"):
            continue
        c = t.pix["conf"].astype(np.float64)
        e = np.abs(np.log(t.pix["gt"].astype(np.float64) / t.pix["pred"].astype(np.float64)))
        i = rng.choice(t.n_pixels, min(200_000, t.n_pixels), replace=False)
        print(f"  conf       percentiles {Q}: {np.round(np.percentile(c, Q), 4)}")
        print(f"  |log err|  percentiles {Q}: {np.round(np.percentile(e, Q), 4)}")
        print(f"  spearman(conf, |log err|) = {spearmanr(c[i], e[i]).statistic:.3f}; "
              f"median(|log err|)/median(conf) = {np.median(e) / np.median(c):.4g}")
        imgs = rng.choice(t.n_images, min(200, t.n_images), replace=False)
        within = [spearmanr(c[t.offsets[j]:t.offsets[j + 1]], e[t.offsets[j]:t.offsets[j + 1]]).statistic
                  for j in imgs]
        print(f"  within-image spearman over {len(imgs)} images: median {np.nanmedian(within):.3f}, "
              f"IQR [{np.nanpercentile(within, 25):.3f}, {np.nanpercentile(within, 75):.3f}]")


if __name__ == "__main__":
    main()
