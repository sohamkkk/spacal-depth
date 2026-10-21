"""Write a synthetic cache for a laptop dry run of the whole CPU pipeline.

    SPACAL_CACHE_ROOT=/tmp/spacal/cache python -m experiments.make_synthetic_cache
"""

import argparse

from src.config import CACHE_ROOT
from src.data.synthetic import write_synthetic_cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-scenes", type=int, default=60)
    ap.add_argument("--imgs-per-scene", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if "/tmp" not in str(CACHE_ROOT) and "synthetic" not in str(CACHE_ROOT):
        raise SystemExit(f"refusing to write synthetic data into {CACHE_ROOT}; set SPACAL_CACHE_ROOT to a scratch dir")
    write_synthetic_cache(CACHE_ROOT, "nyu", n_scenes=args.n_scenes, imgs_per_scene=args.imgs_per_scene, seed=args.seed)
    write_synthetic_cache(
        CACHE_ROOT, "diode_indoor", n_scenes=args.n_scenes // 2, imgs_per_scene=args.imgs_per_scene,
        sigma_g=0.2, seed=args.seed + 1,
    )
    print(f"synthetic caches written under {CACHE_ROOT}")


if __name__ == "__main__":
    main()
