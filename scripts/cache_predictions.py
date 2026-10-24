"""Run on the REMOTE GPU box. One pass of a frozen model over a dataset; writes
everything later CPU experiments need so the GPU is never revisited:

  cache/<dataset>/gt/<sample_id>.npz        gt (float32, eval crop), mask, labels*, rgb*, ids
  cache/<dataset>/<model_tag>/<sample_id>.npz
      pred (float32), pred_flip (float16)*, conf (float16)*, grad_unc (float16)*,
      t_pred / t_flip / t_grad (seconds, for the E10 cost table)
  cache/<dataset>/<model_tag>/manifest.json  model id, versions, args, protocol

Arrays are cropped to the evaluation region of the dataset protocol
(src/data/protocols.py). Re-running skips samples that already exist.

Usage (remote):
    SPACAL_DEVICE=cuda python -m scripts.cache_predictions --dataset nyu --model dav2-indoor-large --flip --grad
"""

import argparse
import json
import platform
import time
from datetime import datetime, timezone

import numpy as np

from src.config import CACHE_ROOT, DEVICE
from src.data.protocols import PROTOCOLS, crop, valid_mask
from src.data.registry import DATASETS, build_dataset
from src.models.base import cuda_timer
from src.models.registry import build_model


def write_gt(sample: dict, protocol, out_path, save_rgb: bool) -> None:
    gt = crop(sample["gt"], protocol).astype(np.float32)
    extra = crop(sample["extra_mask"], protocol) if "extra_mask" in sample else None
    payload = {
        "gt": gt,
        "mask": valid_mask(gt, protocol, extra),
        "scene_id": sample["scene_id"],
        "sample_id": sample["sample_id"],
        "official_test": bool(sample["meta"].get("official_test", True)),
        "subset": str(sample["meta"].get("subset", "")),
    }
    if "labels" in sample:
        payload["labels"] = crop(sample["labels"], protocol).astype(np.uint16)
    if save_rgb:
        payload["rgb"] = crop(sample["rgb"], protocol).astype(np.uint8)
    np.savez_compressed(out_path, **payload)


def library_versions() -> dict:
    import torch

    versions = {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda}
    try:
        import transformers

        versions["transformers"] = transformers.__version__
    except ImportError:
        pass
    if torch.cuda.is_available():
        versions["gpu"] = torch.cuda.get_device_name(0)
    return versions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=DATASETS, required=True)
    ap.add_argument("--model", required=True, help="e.g. dav2-indoor-large, unidepthv2-large")
    ap.add_argument("--flip", action="store_true", help="also cache the horizontally flipped prediction")
    ap.add_argument("--grad", action="store_true", help="also cache gradient-based uncertainty (DA-V2 only)")
    ap.add_argument("--rgb-every", type=int, default=20, help="store the RGB crop for every k-th sample (figures)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N samples (smoke test)")
    ap.add_argument("--amp", action="store_true",
                    help="fp16 autocast for forward passes (~3x faster on T4); gradient uncertainty stays fp32")
    args = ap.parse_args()
    import torch

    def amp():
        return torch.autocast("cuda", dtype=torch.float16, enabled=args.amp and DEVICE.startswith("cuda"))

    protocol = PROTOCOLS[args.dataset]
    dataset = build_dataset(args.dataset)
    model = build_model(args.model, device=DEVICE)
    if args.grad and not model.has_grad_uncertainty:
        raise ValueError(f"{args.model} has no gradient-uncertainty implementation")

    gt_dir = CACHE_ROOT / args.dataset / "gt"
    out_dir = CACHE_ROOT / args.dataset / model.tag
    gt_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        **model.info(),
        "dataset": args.dataset,
        "protocol": {"name": protocol.name, "max_depth": protocol.max_depth, "stride": protocol.stride},
        "args": vars(args),
        "versions": library_versions(),
        "started": datetime.now(timezone.utc).isoformat(),
        "n_samples": len(dataset),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    n = min(len(dataset), args.limit) if args.limit else len(dataset)
    t_start, done, grad_oom = time.perf_counter(), 0, 0
    for idx in range(n):
        sample = dataset[idx]
        sid = sample["sample_id"]
        gt_path, pred_path = gt_dir / f"{sid}.npz", out_dir / f"{sid}.npz"
        if not gt_path.exists():
            write_gt(sample, protocol, gt_path, save_rgb=(idx % args.rgb_every == 0))
        if pred_path.exists():
            continue

        timings: dict[str, float] = {}
        kw = {"focal": sample["meta"]["focal"]} if model.needs_focal else {}
        with cuda_timer(DEVICE, timings, "t_pred"), amp():
            out = model.predict(sample["rgb"], **kw)
        if out["depth"].shape != sample["gt"].shape:
            raise RuntimeError(f"{sid}: pred {out['depth'].shape} vs gt {sample['gt'].shape}")
        payload = {"pred": crop(out["depth"], protocol).astype(np.float32)}
        if "conf" in out:
            payload["conf"] = crop(out["conf"], protocol).astype(np.float16)
        if args.flip:
            with cuda_timer(DEVICE, timings, "t_flip"), amp():
                out_f = model.predict_flip(sample["rgb"], **kw)
            payload["pred_flip"] = crop(out_f["depth"], protocol).astype(np.float16)
            if "conf" in out_f:
                payload["conf_flip"] = crop(out_f["conf"], protocol).astype(np.float16)
        if args.grad:
            try:
                with cuda_timer(DEVICE, timings, "t_grad"):
                    g = model.gradient_uncertainty(sample["rgb"])
                payload["grad_unc"] = crop(g, protocol).astype(np.float16)
            except torch.cuda.OutOfMemoryError:
                # small-memory GPUs (T4) on large inputs: keep the sample, drop this one map;
                # build_table then omits grad_unc for the whole dataset
                torch.cuda.empty_cache()
                timings.pop("t_grad", None)
                grad_oom += 1
                print(f"[{sid}] gradient uncertainty OOM; cached without grad_unc", flush=True)
        np.savez_compressed(pred_path, **payload, **{k: np.float32(v) for k, v in timings.items()})
        done += 1
        if done % 50 == 0:
            rate = done / (time.perf_counter() - t_start)
            print(f"[{args.dataset}/{model.tag}] {idx + 1}/{n}  {rate:.2f} img/s", flush=True)

    manifest["finished"] = datetime.now(timezone.utc).isoformat()
    manifest["grad_oom_this_run"] = grad_oom
    manifest["n_cached"] = len(list(out_dir.glob("*.npz")))
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"done: {manifest['n_cached']} cached predictions in {out_dir}")


if __name__ == "__main__":
    main()
