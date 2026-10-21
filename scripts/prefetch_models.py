"""Download every model checkpoint into $HF_HOME on the LOGIN node (which has
internet), so GPU jobs on compute nodes can run with HF_HUB_OFFLINE=1.
Instantiates each model once on CPU, which also triggers any secondary
downloads and proves the imports work.

    python -m scripts.prefetch_models            # all models
    python -m scripts.prefetch_models --skip-unidepth
    python -m scripts.prefetch_models --only-depthpro
"""

import argparse
import os

from huggingface_hub import snapshot_download

from src.models.depth_anything import CHECKPOINTS as DAV2
from src.models.unidepth import CHECKPOINTS as UNIDEPTH

DEPTHPRO = "apple/DepthPro-hf"  # duplicated from src/models/depth_pro.py, which imports torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-unidepth", action="store_true")
    ap.add_argument("--skip-depthpro", action="store_true")
    ap.add_argument("--only-depthpro", action="store_true", help="the other checkpoints are already cached")
    ap.add_argument("--no-instantiate", action="store_true", help="download files only (low-RAM login nodes)")
    args = ap.parse_args()
    print("HF_HOME =", os.environ.get("HF_HOME", "(default ~/.cache/huggingface)"))

    if not args.skip_depthpro:
        print("downloading", DEPTHPRO, "->", snapshot_download(DEPTHPRO))
        if not args.no_instantiate:
            from src.models.depth_pro import DepthProModel

            DepthProModel(device="cpu")
            print("depthpro: loads OK")
    if args.only_depthpro:
        return
    for repo in DAV2.values():
        print("downloading", repo, "->", snapshot_download(repo))
    if not args.skip_unidepth:
        print("downloading", UNIDEPTH["large"], "->", snapshot_download(UNIDEPTH["large"]))

    if args.no_instantiate:
        return
    from src.models.depth_anything import DepthAnythingV2Metric

    for domain in ("indoor", "outdoor"):
        DepthAnythingV2Metric(domain, "small", device="cpu")
        print(f"dav2-{domain}: loads OK")
    if not args.skip_unidepth:
        from src.models.unidepth import UniDepthV2Model

        UniDepthV2Model("large", device="cpu")
        print("unidepthv2-large: loads OK")
    print("all checkpoints cached; GPU jobs can use HF_HUB_OFFLINE=1")


if __name__ == "__main__":
    main()
