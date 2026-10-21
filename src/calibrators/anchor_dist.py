"""A-dist: anchor-geometry-aware cross-image calibration (docs/ANCHOR_METHOD_PROPOSAL.md).

Per pixel x of an image with anchors A (log residuals d_a observed at test time):
  m(x)      local shift from the anchors (kind: median = the image's anchor median, as A-hybrid;
            knn = median of the K nearest anchors; kernel = Gaussian weights on the K nearest;
            bilateral = kernel x similarity of log predicted depth)
  sigma(x)  anchor spread (global: the image's LOO spread, as A-hybrid; local: median |d_a - m(x)|
            over the K nearest anchors)
  delta(x)  distance to the nearest anchor / image diagonal
  s(x)      = b(x) * (sigma(x) + tau) * g(delta(x)), g non-decreasing, fitted on head-train
Score |d - m| / s; one CRC lambda per distance bin (Mondrian), calibrated across calibration images
(over scenes by default). Bandwidths, tau, g and bin merges come from head-train only, so for every
anchor rule applied identically to all images the per-bin guarantee E[bin miscoverage] <= alpha holds
for a new image passing that bin's min-pixel filter (Proposition 1 of the proposal).
"""

from dataclasses import dataclass, replace

import numpy as np
from scipy.spatial import cKDTree
from sklearn.isotonic import IsotonicRegression

from src.calibrators.anchors import AnchorDraw
from src.calibrators.crc import InfeasibleRiskError, crc_lambda_exact, crc_lambda_exact_units
from src.calibrators.mondrian import group_offsets
from src.data.prepared import PixelTable

KINDS = ("median", "knn", "kernel", "bilateral")
DIST_EDGES = (0.02, 0.05, 0.1, 0.2)
TAU_GRID = (0.25, 0.5, 1.0, 2.0, 4.0)
H_GRID = (0.05, 0.1, 0.2)
HD_GRID = (0.1, 0.3)


@dataclass(frozen=True)
class FieldConfig:
    kind: str = "kernel"
    K: int = 16
    h: float = 0.1            # spatial bandwidth, fraction of the image diagonal
    h_d: float = 0.2          # bilateral bandwidth in log predicted depth
    spread: str = "local"     # local | global


@dataclass
class Fields:
    shift: np.ndarray
    sigma: np.ndarray         # NaN where undefined
    delta: np.ndarray         # inf for images without anchors


@dataclass
class DistModel:
    cfg: FieldConfig
    tau: float
    sigma_fill: float
    edges: tuple[float, ...] = DIST_EDGES
    g_knots: np.ndarray | None = None
    g_vals: np.ndarray | None = None
    s_cap: float = np.inf
    head_score: float = np.nan   # log interval score on head half B (lower is better)

    def bins(self, f: Fields) -> np.ndarray:
        return np.digitize(f.delta, self.edges)

    def scale(self, f: Fields, b: np.ndarray) -> np.ndarray:
        s = b * (np.nan_to_num(f.sigma, nan=self.sigma_fill) + self.tau)
        if self.g_knots is not None:
            s = s * np.interp(np.minimum(f.delta, 1e9), self.g_knots, self.g_vals)
        # widths are exp(lambda * s): an uncapped product of three factors puts e^10-wide intervals on a few pixels
        return np.minimum(s, self.s_cap)


def anchor_fields(t: PixelTable, d: np.ndarray, draw: AnchorDraw, cfg: FieldConfig) -> Fields:
    """m, sigma and delta for every pixel of t (anchor values only; evaluation pixels' d never used)."""
    if cfg.kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {cfg.kind!r}")
    n = t.n_pixels
    shift, sigma, delta = np.zeros(n), np.full(n, np.nan), np.full(n, np.inf)
    xy = np.column_stack([t.pix["row"], t.pix["col"]]).astype(np.float64)
    logp = np.log(t.pix["pred"].astype(np.float64))
    diag = np.hypot(t.img["height"], t.img["width"]).astype(np.float64)
    img_shift = np.nan_to_num(draw.shift, nan=0.0)
    for i in range(t.n_images):
        lo, hi = t.offsets[i], t.offsets[i + 1]
        a = np.flatnonzero(draw.anchor[lo:hi]) + lo
        shift[lo:hi] = img_shift[i]
        if len(a) == 0:
            continue
        kk = min(cfg.K, len(a))
        dist, nn = cKDTree(xy[a]).query(xy[lo:hi], k=kk)
        dist, nn = dist.reshape(hi - lo, kk) / diag[i], nn.reshape(hi - lo, kk)
        da = d[a][nn]
        delta[lo:hi] = dist[:, 0]
        if cfg.kind == "knn":
            shift[lo:hi] = np.median(da, axis=1)
        elif cfg.kind in ("kernel", "bilateral"):
            w = np.exp(-0.5 * (dist / cfg.h) ** 2)
            if cfg.kind == "bilateral":
                w *= np.exp(-0.5 * ((logp[lo:hi, None] - logp[a][nn]) / cfg.h_d) ** 2)
            ws = w.sum(axis=1)
            ok = ws > 1e-8                              # far from every anchor: fall back to the image median
            shift[lo:hi][ok] = (w[ok] * da[ok]).sum(axis=1) / ws[ok]
        if cfg.spread == "local":
            # around the neighbours' own median, not m(x): a bilateral m away from its neighbours
            # would otherwise read as a large spread
            med = np.median(da, axis=1, keepdims=True)
            sigma[lo:hi] = np.median(np.abs(da - med), axis=1)
        else:
            sigma[lo:hi] = draw.sigma[i]
    return Fields(shift, sigma, delta)


def _crc(r, offsets, units, alpha) -> float:
    try:
        lam = crc_lambda_exact(r, offsets, alpha) if units is None else crc_lambda_exact_units(r, offsets, units, alpha)
    except InfeasibleRiskError:
        return float("inf")
    return max(float(lam), 0.0)


def bin_lambdas(r, image_of_pixel, bins, n_bins, n_images, alpha, units=None, min_pixels=10) -> np.ndarray:
    """One CRC lambda per bin over the images with >= min_pixels pixels in it (inf if too few).
    Arrays must be in CSR order; units (per image) switches to CRC over scenes."""
    lams = np.full(n_bins, np.inf)
    for b in range(n_bins):
        sel, offsets, keep = group_offsets(image_of_pixel, bins == b, n_images, min_pixels)
        if len(offsets) >= 2:
            lams[b] = _crc(r[sel], offsets, None if units is None else units[keep], alpha)
    return lams


def merge_edges(delta, image_of_pixel, n_images, edges=DIST_EDGES, min_pixels=10, min_frac=0.05) -> tuple:
    """Drop edges so that every bin has >= min_frac of the images with >= min_pixels pixels in it
    (sparse bins merge into their nearer neighbour). Decided on head-train only."""
    edges = list(edges)
    while edges:
        b = np.digitize(delta, edges)
        cnt = np.bincount(image_of_pixel * (len(edges) + 1) + b, minlength=n_images * (len(edges) + 1))
        n_ok = (cnt.reshape(n_images, -1) >= min_pixels).sum(axis=0)
        sparse = np.flatnonzero(n_ok < min_frac * n_images)
        if len(sparse) == 0:
            break
        j = sparse[-1]
        edges.pop(j - 1 if j > 0 else 0)
    return tuple(edges)


def _fit_g(r0, delta, n_q=20):
    ok = np.isfinite(delta)
    r0, delta = r0[ok], delta[ok]
    qs = np.unique(np.quantile(delta, np.linspace(0, 1, n_q + 1)))
    idx = np.clip(np.digitize(delta, qs[1:-1]), 0, len(qs) - 2)
    cnt = np.bincount(idx, minlength=len(qs) - 1)
    keep = cnt > 0
    centers = np.array([np.median(delta[idx == j]) for j in np.flatnonzero(keep)])
    med = np.array([np.median(r0[idx == j]) for j in np.flatnonzero(keep)])
    vals = IsotonicRegression(increasing=True).fit(centers, med, sample_weight=cnt[keep]).predict(centers)
    vals = np.maximum(vals, 1e-6)
    return centers, vals / vals[0]


def fit_dist_model(head, d_h, b_h, draw_h, cfg, alpha, half_a, use_g=True, single_bin=False,
                   min_pixels=10, fields=None, cap_q=0.999) -> tuple[DistModel, float]:
    """tau (by log interval score: lambda on half A of head-train, score on half B), then g, bin merges and
    the scale cap (cap_q quantile of the head scale; None = no cap).
    Returns the model and its head score (lower is better; used to choose bandwidths)."""
    f = fields if fields is not None else anchor_fields(head, d_h, draw_h, cfg)
    img = head.image_of_pixel()
    ev = draw_h.eval
    fill = float(np.nanmedian(f.sigma[ev]))
    base = max(fill if np.isfinite(fill) else 1.0, 1e-6)
    # tau grid on the image-level spread (as pick_tau): with clustered anchors the local spread is
    # near 0 almost everywhere, and a floor relative to it lets a noisy sigma set the width
    ref = float(np.nanmedian(draw_h.sigma))
    ref = max(ref if np.isfinite(ref) else base, 1e-6)
    res = np.abs(d_h - f.shift)
    sig = np.nan_to_num(f.sigma, nan=base)
    sa, sb = ev & half_a[img], ev & ~half_a[img]
    cnt_a = np.bincount(img[sa], minlength=head.n_images)
    off_a = np.concatenate([[0], np.cumsum(cnt_a[cnt_a > 0])])
    best, tau = np.inf, TAU_GRID[0] * ref
    for c in TAU_GRID:
        den = b_h * (sig + c * ref)
        lam = _crc((res / den)[sa], off_a, None, alpha)
        if not np.isfinite(lam):
            continue
        q = lam * den[sb]
        score = float((2 * q + (2 / alpha) * np.maximum(res[sb] - q, 0)).mean())
        if score < best:
            best, tau = score, c * ref
    model = DistModel(cfg, tau, base, edges=() if single_bin else DIST_EDGES, head_score=best)
    if use_g:
        model.g_knots, model.g_vals = _fit_g((res / (b_h * (sig + tau)))[ev], f.delta[ev])
    if not single_bin:
        model.edges = merge_edges(f.delta[ev], img[ev], head.n_images, min_pixels=min_pixels)
    if cap_q is not None:
        model.s_cap = float(np.quantile(model.scale(f, b_h)[ev], cap_q))
    return model, best


def select_model(head, d_h, b_h, draw_h, kind, alpha, half_a, **kw) -> DistModel:
    """Bandwidths for kernel / bilateral chosen on head-train by the same score;
    kind="auto" also chooses the kind."""
    grid = []
    for k in KINDS if kind == "auto" else (kind,):
        cfg = FieldConfig(kind=k, spread="global" if k == "median" else "local")
        if k == "kernel":
            grid += [replace(cfg, h=h) for h in H_GRID]
        elif k == "bilateral":
            grid += [replace(cfg, h=h, h_d=hd) for h in H_GRID for hd in HD_GRID]
        else:
            grid.append(cfg)
    fits = [fit_dist_model(head, d_h, b_h, draw_h, c, alpha, half_a, **kw) for c in grid]
    return min(fits, key=lambda ms: ms[1])[0]


def min_units_for(alpha) -> int:
    """Fewest calibration units for a CRC risk budget (n + 1) alpha - 1 of at least one unit's loss;
    below it lambda is the bin's largest residual, or inf below 1 / alpha - 1 units."""
    return int(np.ceil(2 / alpha - 1e-9)) - 1


def merge_sparse_bins(bins, image_of_pixel, units, n_bins, min_pixels, min_units) -> np.ndarray:
    """Group id per distance bin such that every group has >= min_units calibration units with an image of
    >= min_pixels pixels in it; a sparse bin joins its nearer neighbour. Depends on where the calibration
    pixels lie, never on their residuals."""
    group = np.arange(n_bins)
    u = np.unique(np.asarray(units), return_inverse=True)[1]
    n_img = len(u)
    while len(ids := np.unique(group)) > 1:
        cnt = np.bincount(image_of_pixel * n_bins + group[bins], minlength=n_img * n_bins).reshape(n_img, n_bins)
        n_units = np.array([len(np.unique(u[cnt[:, j] >= min_pixels])) for j in ids])
        sparse = ids[n_units < min_units]
        if len(sparse) == 0:
            break
        pos = np.searchsorted(ids, sparse[-1])
        group[group == ids[pos]] = ids[pos - 1] if pos > 0 else ids[1]
    return group


def calibrate(model, t, d, b, draw, alpha, units=None, min_pixels=10, fields=None, min_units=None) -> np.ndarray:
    """Per-bin lambdas on calibration table t (evaluation pixels only); bins with fewer than min_units
    calibration units (default min_units_for(alpha)) share the lambda of their nearer neighbour."""
    f = fields if fields is not None else anchor_fields(t, d, draw, model.cfg)
    ev = draw.eval
    r = (np.abs(d - f.shift) / model.scale(f, b))[ev]
    n_bins = len(model.edges) + 1
    img, bins = t.image_of_pixel()[ev], model.bins(f)[ev]
    group = merge_sparse_bins(bins, img, units if units is not None else np.arange(t.n_images), n_bins,
                              min_pixels, min_units_for(alpha) if min_units is None else min_units)
    return bin_lambdas(r, img, group[bins], n_bins, t.n_images, alpha, units, min_pixels)[group]


def interval(model, t, b, lams, fields) -> tuple[np.ndarray, np.ndarray]:
    """[pred exp(m - lambda_bin s), pred exp(m + lambda_bin s)] on every pixel of t."""
    q = lams[model.bins(fields)] * model.scale(fields, b)
    pred = t.pix["pred"].astype(np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        lo, hi = pred * np.exp(fields.shift - q), pred * np.exp(fields.shift + q)
    inf = ~np.isfinite(q)
    lo[inf], hi[inf] = 0.0, np.inf
    return lo, hi
