"""Per-dataset evaluation protocols (crop, depth range, storage stride).

Every loader returns full-resolution arrays; `apply_protocol` turns them into
the evaluation region that all metrics, scores and calibration use. Keeping
this in one place guarantees calibration and test pixels follow identical rules.
"""

from dataclasses import dataclass

import numpy as np

MIN_DEPTH = 1e-3


@dataclass(frozen=True)
class Protocol:
    name: str
    max_depth: float
    min_depth: float = MIN_DEPTH
    stride: int = 1  # spatial subsampling applied when caching (DIODE is 768x1024)

    def crop_box(self, h: int, w: int) -> tuple[int, int, int, int]:
        """(top, bottom, left, right) of the evaluation crop for an h x w map."""
        return 0, h, 0, w


@dataclass(frozen=True)
class NYUProtocol(Protocol):
    def crop_box(self, h, w):
        if (h, w) != (480, 640):
            raise ValueError(f"NYU Eigen crop is defined on 480x640, got {h}x{w}")
        return 45, 471, 41, 601


@dataclass(frozen=True)
class KITTIProtocol(Protocol):
    def crop_box(self, h, w):
        # Garg crop, applied after the KITTI benchmark crop (352 x 1216)
        return int(0.40810811 * h), int(0.99189189 * h), int(0.03594771 * w), int(0.96405229 * w)


PROTOCOLS: dict[str, Protocol] = {
    "nyu": NYUProtocol("nyu", max_depth=10.0),
    "kitti": KITTIProtocol("kitti", max_depth=80.0),
    "diode_indoor": Protocol("diode_indoor", max_depth=10.0, stride=2),
    "diode_outdoor": Protocol("diode_outdoor", max_depth=80.0, stride=2),
    "eth3d": Protocol("eth3d", max_depth=80.0, stride=2),  # after the loader's 4x reduction: 756 x 504
    "arkitscenes": Protocol("arkitscenes", max_depth=10.0),  # after the loader's 3x reduction: 480 x 640
}

# which Depth-Anything-V2 metric checkpoint family matches each dataset's domain (ETH3D mixes both; the
# outdoor depth range is used for its depth groups, and DA-V2 is not run on it)
DOMAIN = {"nyu": "indoor", "kitti": "outdoor", "diode_indoor": "indoor", "diode_outdoor": "outdoor",
          "eth3d": "outdoor", "arkitscenes": "indoor"}

# NYU raw 894-class ids (1-based) of hard-case classes, from the labelled .mat `names`
NYU_MIRROR_IDS = (122,)
NYU_WINDOW_IDS = (59,)
NYU_GLASS_IDS = (216, 293, 412, 612, 642, 861, 889)


def kitti_benchmark_crop(arr: np.ndarray) -> np.ndarray:
    """Standard KB crop to 352 x 1216 (bottom-centre), used before inference."""
    h, w = arr.shape[:2]
    top, left = h - 352, (w - 1216) // 2
    return arr[top : top + 352, left : left + 1216]


def crop(arr: np.ndarray, protocol: Protocol) -> np.ndarray:
    t, b, l, r = protocol.crop_box(*arr.shape[:2])
    out = arr[t:b, l:r]
    return out[:: protocol.stride, :: protocol.stride]


def valid_mask(gt: np.ndarray, protocol: Protocol, extra: np.ndarray | None = None) -> np.ndarray:
    m = np.isfinite(gt) & (gt > protocol.min_depth) & (gt < protocol.max_depth)
    if extra is not None:
        m &= extra.astype(bool)
    return m


def clamp_pred(pred: np.ndarray, protocol: Protocol) -> np.ndarray:
    """Standard eval clipping; also guards logs against bilinear undershoot."""
    pred = np.nan_to_num(pred, nan=protocol.min_depth, posinf=protocol.max_depth)
    return np.clip(pred, protocol.min_depth, protocol.max_depth)
