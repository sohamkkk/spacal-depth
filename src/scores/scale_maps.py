"""Non-conformity scores r = |y - yhat|_space / s(x) and their intervals.

A score is a (space, scale model) pair:
  space "lin": residual |y - yhat|,          interval [max(yhat - q, 0), yhat + q]
  space "log": residual |log y - log yhat|,  interval [yhat e^-q, yhat e^q]
with q = lambda * s(x). Every scale model is fit on the HEAD-TRAIN pool only
(never calibration or test) and then evaluated on any table.

Scores in the paper:
  S0  lin, s = 1            S1  lin, s = yhat         S2  log, s = 1
  S3  log, s = |dlog flip| + c   S3e log, s = ensemble log-std + c
  S4  log, s = gradient uncertainty / median + c       S4c log, s = UniDepth conf + c
  S5  log, learned quantile head (src/scores/learned.py)
  S6  log, factorized a(x) * b(x) (src/scores/learned.py)
"""

from dataclasses import dataclass

import numpy as np

from src.data.prepared import PixelTable

SPACES = ("lin", "log")


def residual(table: PixelTable, space: str) -> np.ndarray:
    pred, gt = table.pix["pred"].astype(np.float64), table.pix["gt"].astype(np.float64)
    if space == "lin":
        return np.abs(pred - gt)
    if space == "log":
        return np.abs(np.log(pred) - np.log(gt))
    raise ValueError(f"unknown space {space!r}")


def interval(pred: np.ndarray, q: np.ndarray, space: str) -> tuple[np.ndarray, np.ndarray]:
    """q = lambda * s(x) per pixel; q = inf gives the trivial interval [0, inf)."""
    pred = pred.astype(np.float64)
    if space == "lin":
        return np.maximum(pred - q, 0.0), pred + q
    if space == "log":
        with np.errstate(over="ignore"):
            return pred * np.exp(-q), pred * np.exp(q)
    raise ValueError(f"unknown space {space!r}")


class ScaleModel:
    name = "base"

    def fit(self, head: PixelTable) -> "ScaleModel":
        return self

    def predict(self, table: PixelTable) -> np.ndarray:
        raise NotImplementedError


class ConstantScale(ScaleModel):
    name = "const"

    def predict(self, table):
        return np.ones(table.n_pixels)


class RelativeScale(ScaleModel):
    name = "relative"

    def predict(self, table):
        return table.pix["pred"].astype(np.float64)


class HeuristicScale(ScaleModel):
    """s = T(min(field / m, cap)) + c.
    m   head-pool median (puts unitless fields on a common footing),
    cap head-pool 99th percentile of field / m,
    T   identity / sqrt / log1p: compresses heavy right tails (gradient
        magnitudes span orders of magnitude, so even the capped ratio can
        make log intervals e^{lambda s} explode),
    c   floor so near-zero heuristic values cannot give zero-width
        intervals.
    (T, c) are chosen jointly by interval score on a held-out half of the
    head pool (head data only, so validity is untouched).
    """

    FLOOR_GRID = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0)
    TRANSFORMS = {"id": lambda x: x, "sqrt": np.sqrt, "log1p": np.log1p}

    def __init__(self, field: str, floor_frac: float | None = None, space: str = "log", seed: int = 0,
                 transform: str | None = None):
        if transform is not None and transform not in self.TRANSFORMS:
            raise ValueError(f"transform must be one of {list(self.TRANSFORMS)}")
        self.field, self.floor_frac, self.space, self.seed = field, floor_frac, space, seed
        self.transform = transform if transform is not None or floor_frac is None else "id"
        self.name = f"heur-{field}"

    def _capped(self, table):
        return np.minimum(table.pix[self.field].astype(np.float64) / self.m, self.cap)

    def predict_field(self, values: np.ndarray) -> np.ndarray:
        """s for raw field values of any shape (e.g. a full-resolution map rebuilt from the cache)."""
        x = np.minimum(np.asarray(values, dtype=np.float64) / self.m, self.cap)
        return self.TRANSFORMS[self.transform](x) + self.floor_frac

    def fit(self, head):
        if not head.has(self.field):
            raise KeyError(f"table has no field {self.field!r} (was it cached?)")
        f = head.pix[self.field].astype(np.float64)
        med = float(np.median(f))
        self.m = med if med > 0 else float(np.mean(f)) + 1e-12
        self.cap = float(np.quantile(f / self.m, 0.99))
        if self.floor_frac is None or self.transform is None:
            self.transform, self.floor_frac = self._select(head)
        return self

    def _select(self, head, alpha: float = 0.1) -> tuple[str, float]:
        from src.calibrators.crc import InfeasibleRiskError, crc_lambda_exact
        from src.metrics.interval_metrics import interval_score

        transforms = [self.transform] if self.transform is not None else list(self.TRANSFORMS)
        floors = [self.floor_frac] if self.floor_frac is not None else list(self.FLOOR_GRID)
        if head.n_images < 10:
            return transforms[-1], (floors[0] if len(floors) == 1 else 0.25)
        perm = np.random.default_rng(self.seed + 7).permutation(head.n_images)
        a, b = head.subset(perm[: head.n_images // 2]), head.subset(perm[head.n_images // 2 :])
        ra = residual(a, self.space)
        xa, xb = self._capped(a), self._capped(b)
        gt_b = b.pix["gt"].astype(np.float64)
        best, choice = np.inf, (transforms[-1], floors[0])
        for t in transforms:
            fa, fb = self.TRANSFORMS[t](xa), self.TRANSFORMS[t](xb)
            for c in floors:
                try:
                    lam = crc_lambda_exact(ra / (fa + c), a.offsets, alpha)
                except InfeasibleRiskError:
                    continue
                lo, hi = interval(b.pix["pred"], lam * (fb + c), self.space)
                with np.errstate(over="ignore", invalid="ignore"):
                    score = np.mean(interval_score(lo, hi, gt_b, alpha) / gt_b)
                if np.isfinite(score) and score < best:
                    best, choice = score, (t, c)
        return choice

    def predict(self, table):
        return self.predict_field(table.pix[self.field])


@dataclass(frozen=True)
class ScoreSpec:
    name: str
    space: str
    scale: str          # key into build_scale_model
    needs: tuple = ()   # table fields required


SCORES = {
    "S0-const": ScoreSpec("S0-const", "lin", "const"),
    "S1-rel": ScoreSpec("S1-rel", "lin", "relative"),
    "S2-log": ScoreSpec("S2-log", "log", "const"),
    "S3-flip": ScoreSpec("S3-flip", "log", "heur:dflip", ("dflip",)),
    "S3e-ens": ScoreSpec("S3e-ens", "log", "heur:ens_std", ("ens_std",)),
    "S4-grad": ScoreSpec("S4-grad", "log", "heur:grad_unc", ("grad_unc",)),
    "S4c-conf": ScoreSpec("S4c-conf", "log", "heur:conf", ("conf",)),
    "S5-learned": ScoreSpec("S5-learned", "log", "learned"),
    "S6-fact": ScoreSpec("S6-fact", "log", "factorized:learned"),
    "S6-fact-conf": ScoreSpec("S6-fact-conf", "log", "factorized:heur:conf", ("conf",)),
    "S6-fact-flip": ScoreSpec("S6-fact-flip", "log", "factorized:heur:dflip", ("dflip",)),
}


def available_scores(table: PixelTable, names=None) -> list[str]:
    names = names or list(SCORES)
    return [n for n in names if all(table.has(f) for f in SCORES[n].needs)]


def build_scale_model(key: str, beta: float = 0.1, seed: int = 0) -> ScaleModel:
    if key == "const":
        return ConstantScale()
    if key == "relative":
        return RelativeScale()
    if key.startswith("heur:"):
        return HeuristicScale(key.split(":", 1)[1], seed=seed)
    from src.scores.learned import FactorizedScale, LearnedQuantileScale  # sklearn only when needed

    if key == "learned":
        return LearnedQuantileScale(seed=seed)
    if key.startswith("factorized:"):
        return FactorizedScale(build_scale_model(key.split(":", 1)[1], beta, seed), beta=beta, seed=seed)
    raise ValueError(f"unknown scale model {key!r}")


# --- backward-compatible helpers used by older tests / notebooks ---
def s0_constant(pred):
    return np.ones_like(pred)


def s1_relative(pred):
    return pred


def log_residual(pred, gt):
    return np.abs(np.log(pred) - np.log(gt))


def normalized_score(pred, gt, scale):
    return np.abs(pred - gt) / scale
