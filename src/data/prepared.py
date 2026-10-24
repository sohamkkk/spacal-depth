"""PixelTable: the compact, CPU-friendly representation every experiment uses.

Per-pixel arrays are stored image-contiguously (CSR layout): image i owns
rows offsets[i]:offsets[i+1]. Only valid-GT pixels on a fixed sub-lattice
(stride s, a deterministic function of position) are kept, so per-image
coverage is estimated on that lattice. All x-only features (gradients,
local variation, edge flags, image-level summaries) are computed on the
FULL evaluation crop before subsampling, so they never depend on where GT
happens to exist.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.ndimage import sobel, uniform_filter

from src.data.cache import list_samples, load_gt, load_pred
from src.data.protocols import PROTOCOLS, clamp_pred

EDGE_QUANTILE = 0.9  # top 10 % of predicted log-depth gradient magnitude = "edge"


@dataclass
class PixelTable:
    pix: dict[str, np.ndarray]
    img: dict[str, np.ndarray]
    offsets: np.ndarray
    meta: dict = field(default_factory=dict)

    @property
    def n_images(self) -> int:
        return len(self.offsets) - 1

    @property
    def n_pixels(self) -> int:
        return int(self.offsets[-1])

    @property
    def counts(self) -> np.ndarray:
        return np.diff(self.offsets)

    def image_of_pixel(self) -> np.ndarray:
        return np.repeat(np.arange(self.n_images), self.counts)

    def has(self, name: str) -> bool:
        return name in self.pix

    def subset(self, image_idx: np.ndarray) -> "PixelTable":
        """Images in ascending index order (keeps the CSR layout contiguous)."""
        image_idx = np.unique(np.asarray(image_idx, dtype=np.int64))
        counts = self.counts[image_idx]
        keep_img = np.zeros(self.n_images, dtype=bool)
        keep_img[image_idx] = True
        sel = np.repeat(keep_img, self.counts)
        return PixelTable(
            pix={k: v[sel] for k, v in self.pix.items()},
            img={k: v[image_idx] for k, v in self.img.items()},
            offsets=np.concatenate([[0], np.cumsum(counts)]).astype(np.int64),
            meta=dict(self.meta),
        )

    def select_pixels(self, mask: np.ndarray, min_count: int = 1) -> tuple["PixelTable", np.ndarray]:
        """Pixels where mask is True, dropping images left with fewer than min_count.
        Returns (table, keep_img) with keep_img over this table's images."""
        counts = np.bincount(self.image_of_pixel()[mask], minlength=self.n_images)
        keep_img = counts >= max(min_count, 1)
        sel = mask & np.repeat(keep_img, self.counts)
        return PixelTable(
            pix={k: v[sel] for k, v in self.pix.items()},
            img={k: v[keep_img] for k, v in self.img.items()},
            offsets=np.concatenate([[0], np.cumsum(counts[keep_img])]).astype(np.int64),
            meta=dict(self.meta),
        ), keep_img

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {f"pix__{k}": v for k, v in self.pix.items()}
        arrays.update({f"img__{k}": v for k, v in self.img.items()})
        np.savez(path, offsets=self.offsets, meta=np.array(json.dumps(self.meta)), **arrays)

    @staticmethod
    def load(path: Path) -> "PixelTable":
        with np.load(path) as d:
            pix = {k[5:]: d[k] for k in d.files if k.startswith("pix__")}
            img = {k[5:]: d[k] for k in d.files if k.startswith("img__")}
            return PixelTable(pix=pix, img=img, offsets=d["offsets"], meta=json.loads(str(d["meta"])))


def _safe_log(x: np.ndarray, protocol) -> np.ndarray:
    return np.log(clamp_pred(x.astype(np.float32), protocol))


def _image_features(logp: np.ndarray, maps: dict[str, np.ndarray]) -> dict[str, float]:
    feats = {"med_logpred": float(np.median(logp)), "std_logpred": float(np.std(logp))}
    for name, m in maps.items():
        feats[f"med_{name}"] = float(np.median(m))
        feats[f"p90_{name}"] = float(np.quantile(m, 0.9))
    return feats


def build_table(
    dataset: str,
    model_tag: str,
    ensemble_tags: tuple[str, ...] = (),
    stride: int = 4,
    min_valid: int = 100,
    max_images: int = 0,
    cache_root: Path | None = None,
) -> PixelTable:
    protocol = PROTOCOLS[dataset]
    sample_ids = list_samples(dataset, model_tag, cache_root)
    for tag in ensemble_tags:
        sample_ids = sorted(set(sample_ids) & set(list_samples(dataset, tag, cache_root)))
    if max_images:
        sample_ids = sample_ids[:max_images]

    pix_lists: dict[str, list] = {}
    img_lists: dict[str, list] = {}
    dropped = 0
    for sid in sample_ids:
        g = load_gt(dataset, sid, cache_root)
        p = load_pred(dataset, model_tag, sid, cache_root)
        gt, mask = g["gt"].astype(np.float32), g["mask"].astype(bool)
        pred = clamp_pred(p["pred"].astype(np.float32), protocol)
        if pred.shape != gt.shape:
            raise ValueError(f"{sid}: pred {pred.shape} vs gt {gt.shape}")
        h, w = gt.shape
        logp = np.log(pred)

        maps: dict[str, np.ndarray] = {
            "grad_logpred": np.hypot(sobel(logp, axis=0), sobel(logp, axis=1)) / 8.0,
        }
        mean = uniform_filter(logp, 7)
        maps["local_std"] = np.sqrt(np.maximum(uniform_filter(logp**2, 7) - mean**2, 0.0))
        if "pred_flip" in p:
            maps["dflip"] = np.abs(logp - _safe_log(p["pred_flip"], protocol))
        if "conf" in p:
            maps["conf"] = np.maximum(p["conf"].astype(np.float32), 0.0)
        if "grad_unc" in p:
            maps["grad_unc"] = np.maximum(p["grad_unc"].astype(np.float32), 0.0)
        if ensemble_tags:
            members = [logp] + [_safe_log(load_pred(dataset, t, sid, cache_root)["pred"], protocol) for t in ensemble_tags]
            maps["ens_std"] = np.std(np.stack(members), axis=0)
        edge = maps["grad_logpred"] >= np.quantile(maps["grad_logpred"], EDGE_QUANTILE)

        lattice = np.zeros_like(mask)
        lattice[::stride, ::stride] = True
        keep = mask & lattice
        n_valid = int(keep.sum())
        if n_valid < min_valid:
            dropped += 1
            continue
        rows, cols = np.nonzero(keep)
        per_pixel = {
            "pred": pred[keep],
            "gt": gt[keep],
            "row": rows.astype(np.uint16),
            "col": cols.astype(np.uint16),
            "edge": edge[keep],
            "label": g["labels"][keep].astype(np.uint16) if "labels" in g else np.zeros(n_valid, np.uint16),
        }
        per_pixel.update({k: v[keep].astype(np.float32) for k, v in maps.items()})
        for k, v in per_pixel.items():
            pix_lists.setdefault(k, []).append(v)

        per_image = {
            "scene_id": str(g["scene_id"]),
            "sample_id": sid,
            "official_test": bool(g["official_test"]),
            "subset": str(g.get("subset", "")),
            "height": h,
            "width": w,
            "n_valid": n_valid,
        }
        per_image.update(_image_features(logp, {k: v for k, v in maps.items() if k != "grad_logpred"}))
        for k, v in per_image.items():
            img_lists.setdefault(k, []).append(v)

    if not img_lists:
        raise ValueError(f"no images with >= {min_valid} valid lattice pixels in {dataset}/{model_tag}")
    counts = np.array(img_lists["n_valid"])
    n_kept = len(counts)
    # an auxiliary map missing for some images (e.g. gradient OOM) is dropped for the whole dataset
    partial = sorted(k for k, v in pix_lists.items() if len(v) != n_kept)
    for k in partial:
        print(f"warning: field {k!r} present in {len(pix_lists[k])}/{n_kept} images of {dataset}/{model_tag}; dropped")
        del pix_lists[k]
        for stat in [s for s in img_lists if s.endswith(f"_{k}")]:
            del img_lists[stat]
    img_lists = {k: v for k, v in img_lists.items() if len(v) == n_kept}
    return PixelTable(
        pix={k: np.concatenate(v) for k, v in pix_lists.items()},
        img={k: np.array(v) for k, v in img_lists.items()},
        offsets=np.concatenate([[0], np.cumsum(counts)]).astype(np.int64),
        meta={
            "dataset": dataset,
            "model": model_tag,
            "ensemble": list(ensemble_tags),
            "stride": stride,
            "min_valid": min_valid,
            "dropped_images": dropped,
            "max_depth": PROTOCOLS[dataset].max_depth,
            "dropped_fields": partial,
        },
    )
