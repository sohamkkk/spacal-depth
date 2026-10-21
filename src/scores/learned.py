"""Learned scale models, fit on the HEAD-TRAIN pool only.

S5  LearnedQuantileScale: gradient-boosted (1 - beta_h) quantile of the
    absolute log residual from x-only pixel + image features (a cheap,
    post-hoc stand-in for a learned uncertainty head on frozen outputs).
S6  FactorizedScale: s(x)_j = a(x) * b(x)_j.
      b = any within-image shape model (S5, flip, UniDepth conf, ...)
      a = image-level difficulty from label-free image features, fit by
          ridge regression to log a*_i, a*_i = Q_{1-beta}(r_ij / b_ij) on
          held-out head images (target-only: no calibration data used).
    Motivation: metric monocular depth error is dominated by a per-image
    global scale error, so the "how wrong is this whole image" factor
    should be separated from the "where in the image" factor.
    Bounded on purpose: log a is clipped to the central 98 % of its head
    targets and s to the head's 99.9 % absolute log residual. Unbounded, the
    product a * b reached values no head pixel had, and log-space intervals
    pred * exp(+-lambda s) turned that into widths of 1e3-1e29.
LearnedVarianceScale: E[(log residual)^2] head, used only by the
    uncalibrated Gaussian baseline (raw_baselines.py).
"""

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data.prepared import PixelTable
from src.scores.scale_maps import ScaleModel, residual

PIX_FEATURES = ("grad_logpred", "local_std", "edge", "dflip", "conf", "grad_unc", "ens_std")
CHUNK = 2_000_000


def image_feature_names(table: PixelTable) -> list[str]:
    return sorted(k for k in table.img if k.startswith(("med_", "p90_")) or k == "std_logpred")


def image_features(table: PixelTable, names: list[str]) -> np.ndarray:
    return np.stack([table.img[k].astype(np.float64) for k in names], axis=1)


def pixel_features(table: PixelTable, img_names: list[str], sel: np.ndarray | None = None) -> np.ndarray:
    """[log pred, normalised row/col, pixel fields..., broadcast image features]."""
    iop = table.image_of_pixel()
    if sel is not None:
        iop = iop[sel]
    take = (lambda a: a[sel]) if sel is not None else (lambda a: a)
    cols = [
        np.log(take(table.pix["pred"]).astype(np.float32)),
        take(table.pix["row"]).astype(np.float32) / table.img["height"][iop],
        take(table.pix["col"]).astype(np.float32) / table.img["width"][iop],
    ]
    cols += [take(table.pix[f]).astype(np.float32) for f in PIX_FEATURES if f in table.pix]
    cols += [table.img[k].astype(np.float32)[iop] for k in img_names]
    return np.stack(cols, axis=1)


def _subsample(table: PixelTable, per_image: int, rng: np.random.Generator) -> np.ndarray:
    idx = []
    for i in range(table.n_images):
        lo, hi = table.offsets[i], table.offsets[i + 1]
        n = hi - lo
        idx.append(lo + (rng.choice(n, per_image, replace=False) if n > per_image else np.arange(n)))
    return np.concatenate(idx)


class _GBMScale(ScaleModel):
    def __init__(self, per_image: int = 2000, max_iter: int = 200, seed: int = 0):
        self.per_image, self.max_iter, self.seed = per_image, max_iter, seed

    def _gbm(self, **kw):
        return HistGradientBoostingRegressor(
            max_iter=self.max_iter, learning_rate=0.1, max_leaf_nodes=63, min_samples_leaf=200,
            l2_regularization=1.0, random_state=self.seed, **kw,
        )

    def _train_xy(self, head: PixelTable):
        self.img_names = image_feature_names(head)
        sel = _subsample(head, self.per_image, np.random.default_rng(self.seed))
        return pixel_features(head, self.img_names, sel), residual(head, "log")[sel]

    def _raw_predict(self, table: PixelTable) -> np.ndarray:
        out = np.empty(table.n_pixels)
        for s in range(0, table.n_pixels, CHUNK):
            sel = np.arange(s, min(s + CHUNK, table.n_pixels))
            out[sel] = self.model.predict(pixel_features(table, self.img_names, sel))
        return out


class LearnedQuantileScale(_GBMScale):
    name = "learned-q"

    def __init__(self, quantile: float = 0.9, **kw):
        super().__init__(**kw)
        self.quantile = quantile

    def fit(self, head):
        X, y = self._train_xy(head)
        self.model = self._gbm(loss="quantile", quantile=self.quantile).fit(X, y)
        self.floor = max(0.05 * float(np.median(self.model.predict(X))), 1e-4)
        return self

    def predict(self, table):
        return np.maximum(self._raw_predict(table), self.floor)


class LearnedVarianceScale(_GBMScale):
    """sigma(x) = sqrt(E[(log y - log yhat)^2 | x])."""
    name = "learned-var"

    def fit(self, head):
        X, y = self._train_xy(head)
        self.model = self._gbm(loss="squared_error").fit(X, y**2)
        return self

    def predict(self, table):
        return np.sqrt(np.maximum(self._raw_predict(table), 1e-8))


class FactorizedScale(ScaleModel):
    def __init__(self, base: ScaleModel, beta: float = 0.1, seed: int = 0):
        self.base, self.beta, self.seed = base, beta, seed
        self.name = f"fact[{base.name}]"

    def fit(self, head):
        n = head.n_images
        perm = np.random.default_rng(self.seed + 1).permutation(n)
        learned_base = isinstance(self.base, _GBMScale)
        idx_b, idx_a = (perm[: n // 2], perm[n // 2 :]) if learned_base else (perm, perm)
        self.base.fit(head.subset(idx_b))
        held = head.subset(idx_a)

        res_held = residual(held, "log")
        ratio = res_held / self.base.predict(held)
        target = np.empty(held.n_images)
        for i in range(held.n_images):
            target[i] = np.quantile(ratio[held.offsets[i] : held.offsets[i + 1]], 1 - self.beta)
        log_target = np.log(np.maximum(target, 1e-6))
        self.clip = tuple(np.quantile(log_target, [0.01, 0.99]))
        log_target = np.clip(log_target, *self.clip)

        self.img_names = image_feature_names(held)
        X = image_features(held, self.img_names)
        self.reg = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 12))).fit(X, log_target)
        self.s_max = max(float(np.quantile(res_held, 0.999)), 1e-4)
        self.r2_head = float(self.reg.score(X, log_target))
        return self

    def image_factor(self, table: PixelTable) -> np.ndarray:
        return np.exp(np.clip(self.reg.predict(image_features(table, self.img_names)), *self.clip))

    def predict(self, table):
        a = self.image_factor(table)
        return np.minimum(a[table.image_of_pixel()] * self.base.predict(table), self.s_max)
