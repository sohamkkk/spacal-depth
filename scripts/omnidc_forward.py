"""OMNI-DC (Zuo et al., ICCV 2025; v1.1, Depth-Anything-V2 prior) on ARKitScenes, fed sparse depth only at the
structured anchors of scripts/export_dual_anchor_masks.py; probe pixels stay unknown (0) to the network.
GPU. Needs a clone of https://github.com/princeton-vl/OMNI-DC (no build step, no apex/mmcv) and the zips.

One variant = (mask file, source). source clean: the anchor depth is the table's laser GT; lidar: the table's
iPad depth (anchors with no iPad value are left unknown). Every variant of an image runs in one batch, so the
RGB is decoded once and the frozen Depth-Anything-V2 prior is evaluated once per image.

    SPACAL_DATA_ROOT=/content/data python -m scripts.omnidc_forward --omnidc /content/OMNI-DC \
        --anchors /content/omni_anchors/arkitscenes --masks uniform/k200_seed0_draw0 cluster/k200_seed0_draw0 \
        --sources clean lidar --out /content/drive/MyDrive/spacal/omnidc/arkitscenes --shard 0 --n-shards 1

Output: <out>/s{shard}of{n}_c{chunk}.npz with sample_id, counts (table pixels per image) and, per variant v,
pred__v (OMNI-DC depth at the table pixels, CSR order) and n_sparse__v (anchors given per image).
Chunks already on disk are skipped, so a rerun resumes. scripts/omnidc_merge.py assembles them (CPU).
"""

import argparse
import logging
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.data.arkitscenes import ARKitScenes

HF_REPO = "zuoym15/OMNI-DC"
# src/testing_scripts/demo.sh of the OMNI-DC repo (v1.1 checkpoint); the defaults differ and do not load
OMNI_FLAGS = [
    "--max_depth", "300.0", "--data_normalize_median", "1", "--num_resolution", "3",
    "--multi_resolution_learnable_gradients_weights", "uniform", "--load_dav2", "1", "--gpus", "0",
    "--GRU_iters", "1", "--optim_layer_input_clamp", "1.0", "--depth_activation_format", "exp",
    "--whiten_sparse_depths", "1", "--gru_internal_whiten_method", "median", "--backbone_mode", "rgbd",
    "--pred_confidence_input", "1",
]
PAD_TO = 4 * 2 ** (3 - 1)  # 4 * 2^(num_resolution - 1)
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class SharedRGB(nn.Module):
    """Runs the wrapped RGB-only module on the first batch element and broadcasts: valid because every batch
    holds variants of one image."""

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, x):
        out = self.module(x[:1])
        return out.expand(x.shape[0], *out.shape[1:])


def build_model(omnidc_root: Path, device: str) -> nn.Module:
    src = omnidc_root / "src"
    if not (src / "model" / "ognidc.py").exists():
        raise FileNotFoundError(f"no OMNI-DC checkout at {omnidc_root}")
    sys.path.insert(0, str(src))
    # pvt.py imports mmcv/mmseg only to load an ImageNet init when given a str path; it gets a Path, so never
    for name in ("mmseg", "mmseg.utils", "mmcv", "mmcv.runner"):
        sys.modules[name] = types.ModuleType(name)
    sys.modules["mmseg.utils"].get_root_logger = logging.getLogger
    sys.modules["mmcv.runner"].load_checkpoint = lambda *a, **k: None
    argv, sys.argv = sys.argv, ["omnidc", *OMNI_FLAGS]
    try:
        import config as omni_config
    finally:
        sys.argv = argv
    from model import ognidc
    import pvt
    import torchvision

    # backbone and DA-V2 inits read local files; the HF checkpoint below overwrites every weight anyway
    pvt.get_resnet34 = lambda pretrained=True: torchvision.models.resnet34()

    class _DA(ognidc.DepthAnythingV2):
        def load_state_dict(self, state_dict, *a, **k):
            return None if state_dict is None else super().load_state_dict(state_dict, *a, **k)

    ognidc.DepthAnythingV2 = _DA
    real_load = torch.load
    torch.load = lambda f, *a, **k: None if "depth_anything_v2_" in str(f) and not Path(f).exists() else real_load(f, *a, **k)
    try:
        net = ognidc.OGNIDC(omni_config.args)
    finally:
        torch.load = real_load

    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    missing, unexpected = net.load_state_dict(load_file(hf_hub_download(HF_REPO, "model.safetensors")), strict=False)
    if missing or unexpected:
        raise RuntimeError(f"OMNI-DC checkpoint mismatch: missing={missing[:5]}... unexpected={unexpected[:5]}...")
    net.depth_module = SharedRGB(net.depth_module)
    return net.eval().to(device)


@torch.no_grad()
def complete(net, rgb: np.ndarray, sparse: np.ndarray, device: str, amp: bool) -> np.ndarray:
    """rgb H x W x 3 uint8, sparse V x H x W metres (0 = unknown) -> V x H x W dense depth."""
    v, h, w = sparse.shape
    x = torch.from_numpy(rgb).to(device).permute(2, 0, 1)[None].float() / 255.0
    x = ((x - MEAN.to(device)) / STD.to(device)).expand(v, -1, -1, -1)
    dep = torch.from_numpy(sparse).to(device)[:, None]
    ph, pw = (-h) % PAD_TO, (-w) % PAD_TO
    if ph or pw:
        x = nn.functional.pad(x, (0, pw, 0, ph))
        dep = nn.functional.pad(dep, (0, pw, 0, ph))
    sample = {"rgb": x.contiguous(), "dep": dep, "K": torch.eye(3, device=device).expand(v, 3, 3).clone(), "pattern": 0}
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        pred = net(sample)["pred"]
    return pred[:, 0, :h, :w].float().cpu().numpy()


class Frames(torch.utils.data.Dataset):
    def __init__(self, ds: ARKitScenes, idx: list[int]):
        self.ds, self.idx = ds, idx

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, j):
        it = self.ds[self.idx[j]]
        return j, it["rgb"], it["gt"], it["sample_id"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--omnidc", type=Path, required=True, help="OMNI-DC repo checkout")
    ap.add_argument("--anchors", type=Path, required=True, help="<export --out>/<dataset> (has table_values.npz)")
    ap.add_argument("--masks", nargs="+", required=True, help="mask files relative to --anchors, without .npz")
    ap.add_argument("--sources", nargs="+", default=["clean", "lidar"], choices=("clean", "lidar"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=50, help="images per output file")
    ap.add_argument("--batch", type=int, default=8, help="variants per forward pass")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="images of this shard (smoke test)")
    ap.add_argument("--amp", action="store_true", help="bf16 autocast (default fp32, as published)")
    args = ap.parse_args()
    device = "cuda"
    if not torch.cuda.is_available():
        raise SystemExit("needs a CUDA GPU")

    vals = np.load(args.anchors / "table_values.npz")
    sids, offsets, rows, cols = vals["sample_id"], vals["offsets"], vals["row"], vals["col"]
    depth = {"clean": vals["gt"], "lidar": vals["lidar"] if "lidar" in vals.files else None}
    if "lidar" in args.sources and depth["lidar"] is None:
        raise SystemExit("table_values.npz has no lidar field")
    variants, structured = [], {}
    for m in args.masks:
        with np.load(args.anchors / f"{m}.npz") as z:
            if not (np.array_equal(z["sample_id"], sids) and np.array_equal(z["offsets"], offsets)
                    and np.array_equal(z["row"], rows) and np.array_equal(z["col"], cols)):
                raise ValueError(f"{m}: positions differ from table_values.npz")
            if (z["structured"] & z["probe"]).any():
                raise ValueError(f"{m}: a pixel is both structured anchor and probe")
            structured[m] = z["structured"]
        variants += [(m, s, f"{m.replace('/', '__')}__{s}") for s in args.sources]
    print(f"{len(variants)} variants: {[v[2] for v in variants]}", flush=True)

    ds = ARKitScenes()
    where = {ds.sample_id(i): i for i in range(len(ds))}
    missing = [s for s in sids if s not in where]
    if missing:
        raise ValueError(f"{len(missing)} table images not in the ARKitScenes loader, e.g. {missing[:3]}")
    mine = np.arange(len(sids))[args.shard :: args.n_shards]
    if args.limit:
        mine = mine[: args.limit]
    chunks = [mine[i : i + args.chunk] for i in range(0, len(mine), args.chunk)]
    args.out.mkdir(parents=True, exist_ok=True)
    todo = [(c, imgs) for c, imgs in enumerate(chunks)
            if not (args.out / f"s{args.shard}of{args.n_shards}_c{c:04d}.npz").exists()]
    print(f"shard {args.shard}/{args.n_shards}: {len(mine)} images, {len(chunks)} chunks, {len(todo)} to do", flush=True)
    if not todo:
        return

    net = build_model(args.omnidc, device)
    order = [int(i) for _, imgs in todo for i in imgs]
    loader = torch.utils.data.DataLoader(Frames(ds, [where[sids[i]] for i in order]), batch_size=None,
                                         num_workers=args.workers, prefetch_factor=4 if args.workers else None)
    chunk_of = {int(i): c for c, imgs in todo for i in imgs}
    buf: dict[int, dict] = {}
    t0, done = time.perf_counter(), 0
    for j, rgb, gt, sid in loader:
        i = order[int(j)]
        rgb, gt = np.asarray(rgb), np.asarray(gt)
        if sid != sids[i]:
            raise ValueError(f"loader gave {sid}, expected {sids[i]}")
        sl = slice(offsets[i], offsets[i + 1])
        r, c = rows[sl].astype(np.int64), cols[sl].astype(np.int64)
        g = gt[r, c]
        bad = ~np.isclose(g, depth["clean"][sl], rtol=1e-2, atol=1e-3)
        if bad.mean() > 0.01:
            raise ValueError(f"{sid}: loader GT disagrees with the table at {bad.mean():.1%} of pixels")
        sparse = np.zeros((len(variants), *gt.shape), np.float32)
        for vi, (m, s, _) in enumerate(variants):
            on = structured[m][sl]
            val = depth[s][sl]
            on = on & (val > 0)
            sparse[vi, r[on], c[on]] = val[on]
        n_sparse = (sparse > 0).reshape(len(variants), -1).sum(1)
        pred = np.full((len(variants), len(r)), np.nan, np.float32)
        ok = np.flatnonzero(n_sparse > 0)
        for b in range(0, len(ok), args.batch):
            sel = ok[b : b + args.batch]
            pred[sel] = complete(net, rgb, sparse[sel], device, args.amp)[:, r, c]
        buf[i] = {"pred": pred, "n_sparse": n_sparse, "count": len(r)}
        done += 1
        if done % 25 == 0:
            dt = time.perf_counter() - t0
            print(f"{done}/{len(order)} images  {done / dt:.2f} img/s  eta {(len(order) - done) / done * dt / 60:.0f} min", flush=True)
        cid = chunk_of[i]
        imgs = chunks[cid]
        if all(int(k) in buf for k in imgs):
            out = {"sample_id": sids[imgs], "counts": np.array([buf[int(k)]["count"] for k in imgs], np.int64)}
            for vi, (_, _, name) in enumerate(variants):
                out[f"pred__{name}"] = np.concatenate([buf[int(k)]["pred"][vi] for k in imgs])
                out[f"n_sparse__{name}"] = np.array([buf[int(k)]["n_sparse"][vi] for k in imgs], np.int32)
            path = args.out / f"s{args.shard}of{args.n_shards}_c{cid:04d}.npz"
            tmp = path.with_name(path.stem + ".tmp.npz")
            np.savez(tmp, **out)
            tmp.rename(path)
            for k in imgs:
                del buf[int(k)]
    print(f"shard {args.shard}/{args.n_shards} finished: {done} images in {(time.perf_counter() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
