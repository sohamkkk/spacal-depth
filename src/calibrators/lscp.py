"""LSCP (Jiang & Xie, arXiv 2412.01098) inside one image: the baseline closest to A-dist.

The image is LSCP's spatial field and its anchors are the observed locations. An anchor's feature vector is
the log residuals of its n_nb nearest other anchors (nearest first) and its target is its own residual; a
quantile random forest (Meinshausen 2006) fitted on these pairs predicts, for every other pixel, quantiles of
its residual from the residuals of its n_nb nearest anchors. The interval is [Q(beta), Q(1 - alpha + beta)]
with beta in [0, alpha] chosen per pixel to minimise the width (the paper's beta*).

Its coverage guarantee is asymptotic (stationarity and spatial mixing of the field), with no finite-sample
statement, and it uses no calibration images. Differences from the paper: n_nb is fixed instead of chosen
by 5-fold cross-validation (with 20 anchors per image a cross-validation fold holds 4 points), and beta is
searched on a grid.
"""

import numpy as np
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestRegressor

from src.calibrators.anchors import AnchorDraw
from src.data.prepared import PixelTable

N_NB = 8
N_TREES = 50
MIN_LEAF = 5          # quantregForest's default node size
MAX_FEATURES = 1 / 3  # quantregForest's default mtry for regression
N_BETA = 5


def qrf_weights(forest: RandomForestRegressor, x_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    """Meinshausen weights w[j, i]: mean over trees of 1{test j and train i share a leaf} / leaf size."""
    leaf_tr, leaf_te = forest.apply(x_train), forest.apply(x_test)
    n = len(x_train)
    w = np.zeros((len(x_test), n))
    for t in range(leaf_tr.shape[1]):
        ids, inv, cnt = np.unique(leaf_tr[:, t], return_inverse=True, return_counts=True)
        per_leaf = np.zeros((len(ids), n))
        per_leaf[inv, np.arange(n)] = 1.0 / cnt[inv]
        w += per_leaf[np.searchsorted(ids, leaf_te[:, t])]
    return w / leaf_tr.shape[1]


def weighted_quantiles(y: np.ndarray, w: np.ndarray, taus: np.ndarray) -> np.ndarray:
    """(len(taus), len(w)): per row of w, the smallest y whose cumulative weight reaches tau
    (tau = 0: the smallest y with positive weight)."""
    order = np.argsort(y)
    ys, cw = y[order], np.cumsum(w[:, order], axis=1)
    out = np.empty((len(taus), len(w)))
    for j, tau in enumerate(taus):
        idx = (cw < max(tau - 1e-12, 1e-12)).sum(axis=1)
        out[j] = ys[np.minimum(idx, len(ys) - 1)]
    return out


def lscp_image(d_anchor, xy_anchor, xy_eval, alphas, seed, n_nb=N_NB) -> dict:
    """alpha -> (lower, upper) log-residual offsets for the evaluation pixels of one image;
    (-inf, inf) when the image has fewer than n_nb + 2 anchors."""
    m = len(xy_eval)
    if len(d_anchor) < n_nb + 2:
        return {a: (np.full(m, -np.inf), np.full(m, np.inf)) for a in alphas}
    tree = cKDTree(xy_anchor)
    nb_anchor = tree.query(xy_anchor, k=n_nb + 1)[1][:, 1:]     # itself first, at distance 0
    nb_eval = tree.query(xy_eval, k=n_nb)[1].reshape(m, n_nb)
    forest = RandomForestRegressor(N_TREES, min_samples_leaf=MIN_LEAF, max_features=MAX_FEATURES,
                                   random_state=seed).fit(d_anchor[nb_anchor], d_anchor)
    w = qrf_weights(forest, d_anchor[nb_anchor], d_anchor[nb_eval])
    out, cols = {}, np.arange(m)
    for a in alphas:
        betas = np.linspace(0, a, N_BETA)
        q = weighted_quantiles(d_anchor, w, np.concatenate([betas, 1 - a + betas]))
        lo, hi = q[:N_BETA], q[N_BETA:]
        best = np.argmin(hi - lo, axis=0)
        out[a] = (lo[best, cols], hi[best, cols])
    return out


def lscp_offsets(t: PixelTable, d: np.ndarray, draw: AnchorDraw, alphas, seed: int = 0) -> dict:
    """alpha -> (lower, upper) log offsets on t's evaluation pixels (t.pix order restricted to draw.eval)."""
    xy = np.column_stack([t.pix["row"], t.pix["col"]]).astype(np.float64)
    parts = {a: ([], []) for a in alphas}
    for i in range(t.n_images):
        sl = slice(t.offsets[i], t.offsets[i + 1])
        a_mask = draw.anchor[sl]
        if (~a_mask).sum() == 0:
            continue
        res = lscp_image(d[sl][a_mask], xy[sl][a_mask], xy[sl][~a_mask], alphas, seed + i)
        for a, (lo, hi) in res.items():
            parts[a][0].append(lo)
            parts[a][1].append(hi)
    return {a: (np.concatenate(lo), np.concatenate(hi)) for a, (lo, hi) in parts.items()}
