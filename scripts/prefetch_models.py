"""Download every model checkpoint into $HF_HOME (and Metric3D's torch.hub repo and
weights into $TORCH_HOME) on the LOGIN node (which has internet), so GPU jobs on
compute nodes can run with HF_HUB_OFFLINE=1. Instantiates each model once on CPU,
which also triggers any secondary downloads and proves the imports work.
Use the same HF_HOME / TORCH_HOME as slurm/env.sh:

    HF_HOME=$PWD/.hf_cache TORCH_HOME=$PWD/.torch_cache python -m scripts.prefetch_models
    python -m scripts.prefetch_models --skip-unidepth
    python -m scripts.prefetch_models --only-depthpro
    python -m scripts.prefetch_models --only-metric3d    # needs mmengine (or mmcv) installed
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
    ap.add_argument("--skip-metric3d", action="store_true")
    ap.add_argument("--only-metric3d", action="store_true", help="the other checkpoints are already cached")
    ap.add_argument("--no-instantiate", action="store_true", help="download files only (low-RAM login nodes)")
    args = ap.parse_args()
    print("HF_HOME =", os.environ.get("HF_HOME", "(default ~/.cache/huggingface)"))
    print("TORCH_HOME =", os.environ.get("TORCH_HOME", "(default ~/.cache/torch)"))

    if args.only_metric3d or not args.skip_metric3d:
        import torch

        from src.models.metric3d import HUB_REPO, Metric3DModel, local_repo

        print("Metric3D entries:", torch.hub.list(HUB_REPO, trust_repo=True), "->", local_repo())
        Metric3DModel("large", device="cpu")  # the weights download on first instantiation
        print("metric3d-large: loads OK")
    if args.only_metric3d:
        return

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
