"""Sparse-anchor conformal calibration (E11).

At test time an image comes with k metric anchors: a uniformly random subset
of its valid GT pixels (on KITTI these are LiDAR returns). Calibration images
get anchors drawn the same way, and every method is evaluated on the
non-anchor pixels only. Everything is in log depth, d = log y - log yhat.

  A-rescale  shift by the anchors' median d, then cross-image CRC on
             |d - m| / s(x). Marginal guarantee (anchors are test-time features).
  A-hybrid   additionally divides by the image's anchor residual scale
             sigma = median_j |d_j - median_{-j} d| (leave-one-out), floored by
             tau. Same marginal guarantee; aims at equalising per-image coverage.
  A-local    within-image split conformal: the first half of the anchors
             fixes the shift, the second half sets the width. Because anchor
             and evaluation pixels are a uniform random partition of the image's
             GT pixels, the guarantee holds conditionally on the image
             (Mao, Martin & Reich, arXiv 2006.15640, Lemma 1) for a random GT
             pixel; it needs ceil((k_c + 1)(1 - alpha)) <= k_c, else width = inf.
             E11 runs it with s = 1 only (see experiments/e11_anchors.py).
  A-both     hull of A-hybrid (any s) and A-local: carries both guarantees.

s(x) must not depend on anchor values (it is computed from the raw
prediction), otherwise the within-image exchangeability of A-local breaks.

Structured anchors (E13, `anchor_keys`): real anchors are not a uniform sample
of the GT pixels. Each pattern gives every pixel a random key; the k smallest
keys of an image are its anchors, and the A-local halves are then split at
random among them. The rule is applied independently and identically to every
image, so images stay exchangeable and A-rescale / A-hybrid keep their
cross-image guarantee; A-local's within-image guarantee does not survive,
because anchor and evaluation pixels are no longer a uniform random partition.
  rows     anchors on N_ROWS random lattice rows (a few-beam scan line)
  cluster  the k pixels nearest one random centre pixel
  edges    uniform among predicted-depth edge pixels (feature-point-like)
  near     uniform among the 30 % nearest predicted pixels (range-limited sensor)
"""

from dataclasses import dataclass

import numpy as np


def within_image_rank(offsets: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Position of every pixel in a uniformly random order of its own image."""
    counts = np.diff(offsets)
    img = np.repeat(np.arange(len(counts)), counts)
    order = np.lexsort((rng.random(len(img)), img))
    rank = np.empty(len(img), dtype=np.int64)
    rank[order] = np.arange(len(img)) - offsets[img[order]]
    return rank


def _sorted_groups(values: np.ndarray, groups: np.ndarray, n_groups: int):
    order = np.lexsort((values, groups))
    cnt = np.bincount(groups, minlength=n_groups)
    start = np.concatenate([[0], np.cumsum(cnt)[:-1]])
    return values[order], cnt, start


def group_median(values: np.ndarray, groups: np.ndarray, n_groups: int) -> np.ndarray:
    """Median of `values` per group id (NaN for empty groups)."""
    v, cnt, start = _sorted_groups(values, groups, n_groups)
    out = np.full(n_groups, np.nan)
    ok = cnt > 0
    out[ok] = 0.5 * (v[(start + (cnt - 1) // 2)[ok]] + v[(start + cnt // 2)[ok]])
    return out


def group_loo_spread(values: np.ndarray, groups: np.ndarray, n_groups: int) -> np.ndarray:
    """Per group: median_j |x_j - median(x without x_j)| (NaN if fewer than 3 values)."""
    v, cnt, start = _sorted_groups(values, groups, n_groups)
    g = np.repeat(np.arange(n_groups), cnt)
    p = np.arange(len(v)) - start[g]           # position of x_j inside its sorted group
    L = cnt[g] - 1                             # size after leaving x_j out

    def without_j(q):                          # q-th order statistic of the group minus x_j
        return v[np.clip(start[g] + q + (q >= p), 0, len(v) - 1)]  # clip: groups < 3 are NaN-ed below

    loo_med = 0.5 * (without_j((L - 1) // 2) + without_j(L // 2))
    out = group_median(np.abs(v - loo_med), g, n_groups)
    out[cnt < 3] = np.nan
    return out


def group_conformal_quantile(scores: np.ndarray, groups: np.ndarray, n_groups: int, alpha: float) -> np.ndarray:
    """Per group split-conformal quantile: ceil((n+1)(1-alpha))-th smallest, inf if > n."""
    v, cnt, start = _sorted_groups(scores, groups, n_groups)
    rank = np.ceil((cnt + 1) * (1 - alpha)).astype(np.int64)
    out = np.full(n_groups, np.inf)
    ok = (rank <= cnt) & (cnt > 0)
    out[ok] = v[(start + rank - 1)[ok]]
    return out


@dataclass
class AnchorDraw:
    """One anchor draw on one table. Masks are over the table's pixels."""
    k: int
    anchor: np.ndarray        # all anchors
    scale_half: np.ndarray    # A-local: anchors that fix the shift
    cal_half: np.ndarray      # A-local: anchors that set the width
    shift: np.ndarray         # per image: median d over all anchors (0 if k == 0)
    shift_half: np.ndarray    # per image: median d over scale_half
    sigma: np.ndarray         # per image: LOO anchor spread (NaN if < 3 anchors)
    k_img: np.ndarray         # anchors actually drawn per image

    @property
    def eval(self) -> np.ndarray:
        return ~self.anchor


PATTERNS = ("uniform", "rows", "cluster", "edges", "near")
N_ROWS = 4
NEAR_FRAC = 0.3


def rank_by_key(offsets: np.ndarray, key: np.ndarray) -> np.ndarray:
    """Position of every pixel when its image is sorted by `key` (ascending)."""
    img = np.repeat(np.arange(len(offsets) - 1), np.diff(offsets))
    order = np.lexsort((key, img))
    rank = np.empty(len(img), dtype=np.int64)
    rank[order] = np.arange(len(img)) - offsets[img[order]]
    return rank


def anchor_keys(pattern: str, pix: dict[str, np.ndarray], offsets: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Per-pixel keys whose k smallest per image are that image's anchors.
    Uses x-only fields (row, col, edge, pred) so anchor positions never depend on GT values."""
    if pattern not in PATTERNS[1:]:
        raise ValueError(f"pattern must be one of {PATTERNS[1:]}, got {pattern!r}")
    n_img = len(offsets) - 1
    counts = np.diff(offsets)
    img = np.repeat(np.arange(n_img), counts)
    u = rng.random(len(img))
    if pattern == "cluster":
        centre = offsets[:-1] + (rng.random(n_img) * counts).astype(np.int64)
        r, c = pix["row"].astype(np.float64), pix["col"].astype(np.float64)
        return np.hypot(r - r[centre][img], c - c[centre][img]) + 1e-3 * u
    if pattern == "rows":
        rows = pix["row"].astype(np.int64)
        span = int(rows.max()) + 1
        groups, row_id = np.unique(img * span + rows, return_inverse=True)
        g_img = groups // span
        # random priority per (image, row); the N_ROWS lowest of an image are its scan lines
        order = np.lexsort((rng.random(len(groups)), g_img))
        g_rank = np.empty(len(groups), dtype=np.int64)
        g_rank[order] = np.arange(len(groups)) - np.searchsorted(g_img[order], g_img[order])
        return u + (g_rank[row_id] >= N_ROWS)
    if pattern == "edges":
        return u + ~pix["edge"].astype(bool)
    pred = pix["pred"].astype(np.float64)
    near = rank_by_key(offsets, pred) < np.maximum(1, (NEAR_FRAC * counts).astype(np.int64))[img]
    return u + ~near


def _anchor_stats(d, img, n_img, anchor, scale_half):
    """shift, shift_half (0 where empty) and sigma from the anchor values d."""
    shift = group_median(d[anchor], img[anchor], n_img)
    shift_half = np.nan_to_num(group_median(d[scale_half], img[scale_half], n_img), nan=0.0)
    return shift, shift_half, group_loo_spread(d[anchor], img[anchor], n_img)


def draw_anchors(d: np.ndarray, offsets: np.ndarray, k: int, rng: np.random.Generator,
                 key: np.ndarray | None = None, avail: np.ndarray | None = None) -> AnchorDraw:
    """k anchors per image (at most half its pixels, so evaluation pixels remain).
    key = None draws them uniformly; otherwise the k smallest keys (see anchor_keys).
    avail restricts anchors to those pixels (e.g. where a LiDAR return exists), at most all of them."""
    n_img = len(offsets) - 1
    counts = np.diff(offsets)
    img = np.repeat(np.arange(n_img), counts)
    k_img = np.minimum(k, counts // 2)
    if avail is not None:
        k_img = np.minimum(k_img, np.bincount(img[avail], minlength=n_img))
        key = (rng.random(len(img)) if key is None else key) + 1e9 * ~avail
    if k == 0:
        none = np.zeros(len(d), dtype=bool)
        zeros = np.zeros(n_img)
        return AnchorDraw(0, none, none, none, zeros, zeros, np.full(n_img, np.nan), k_img)
    if key is None:
        rank = within_image_rank(offsets, rng)
        anchor = rank < k_img[img]
        scale_half = rank < (k_img // 2)[img]
    else:
        anchor = rank_by_key(offsets, key) < k_img[img]
        # A-local halves split uniformly among the chosen anchors
        half_rank = rank_by_key(offsets, np.where(anchor, rng.random(len(img)), np.inf))
        scale_half = anchor & (half_rank < (k_img // 2)[img])
    cal_half = anchor & ~scale_half
    return AnchorDraw(k, anchor, scale_half, cal_half, *_anchor_stats(d, img, n_img, anchor, scale_half), k_img)


def with_values(draw: AnchorDraw, d_obs: np.ndarray, offsets: np.ndarray) -> AnchorDraw:
    """The same anchor positions with shift and spread from other anchor values (e.g. a noisy sensor's)."""
    n_img = len(offsets) - 1
    img = np.repeat(np.arange(n_img), np.diff(offsets))
    return AnchorDraw(draw.k, draw.anchor, draw.scale_half, draw.cal_half,
                      *_anchor_stats(d_obs, img, n_img, draw.anchor, draw.scale_half), draw.k_img)


def draw_split_anchors(d: np.ndarray, offsets: np.ndarray, k: int, k_random: int, rng: np.random.Generator,
                       key: np.ndarray | None = None) -> AnchorDraw:
    """Budget split (proposal §5): k - k_random anchors by the rule (k smallest keys; uniform if key is None),
    then k_random uniform among the remaining pixels. The A-local halves are split among the random anchors
    only, so A-local on this draw keeps its within-image guarantee whatever the rule does."""
    n_img = len(offsets) - 1
    counts = np.diff(offsets)
    img = np.repeat(np.arange(n_img), counts)
    k_img = np.minimum(k, counts // 2)
    k_s = np.maximum(k_img - k_random, 0)
    k_r = k_img - k_s
    rank = rank_by_key(offsets, key) if key is not None else within_image_rank(offsets, rng)
    structured = rank < k_s[img]
    rand = rank_by_key(offsets, np.where(structured, np.inf, rng.random(len(img)))) < k_r[img]
    rand &= ~structured
    anchor = structured | rand
    half_rank = rank_by_key(offsets, np.where(rand, rng.random(len(img)), np.inf))
    scale_half = rand & (half_rank < (k_r // 2)[img])
    cal_half = rand & ~scale_half
    return AnchorDraw(k, anchor, scale_half, cal_half, *_anchor_stats(d, img, n_img, anchor, scale_half), k_img)


def pick(arrays: list[np.ndarray], which: np.ndarray) -> np.ndarray:
    """Element j of arrays[which[j]] (arrays of equal shape)."""
    out = arrays[0].copy()
    for i in range(1, len(arrays)):
        m = which == i
        out[m] = arrays[i][m]
    return out


def mix_draws(draws: list[AnchorDraw], choice: np.ndarray, offsets: np.ndarray) -> AnchorDraw:
    """Image i takes its anchors from draws[choice[i]] (draws of the same table, e.g. one per layout).
    With an independent uniform choice per image, this is a draw under a uniform mixture of the rules."""
    px = np.repeat(choice, np.diff(offsets))
    per_px = {f: pick([getattr(dr, f) for dr in draws], px) for f in ("anchor", "scale_half", "cal_half")}
    per_img = {f: pick([getattr(dr, f) for dr in draws], choice) for f in ("shift", "shift_half", "sigma", "k_img")}
    return AnchorDraw(draws[0].k, **per_px, **per_img)


def nearest_anchor_distance(row: np.ndarray, col: np.ndarray, offsets: np.ndarray, anchor: np.ndarray,
                            diag: np.ndarray) -> np.ndarray:
    """Per pixel: distance to the nearest anchor of its image / that image's diagonal (inf: no anchors)."""
    from scipy.spatial import cKDTree

    out = np.full(len(row), np.inf)
    xy = np.column_stack([row, col]).astype(np.float64)
    for i in range(len(offsets) - 1):
        sl = slice(offsets[i], offsets[i + 1])
        a = anchor[sl]
        if a.any():
            out[sl] = cKDTree(xy[sl][a]).query(xy[sl])[0] / diag[i]
    return out


def local_halfwidth(d: np.ndarray, s: np.ndarray, offsets: np.ndarray, draw: AnchorDraw, alpha: float) -> np.ndarray:
    """A-local per-image quantile of |d - shift_half| / s over the cal-half anchors."""
    n_img = len(offsets) - 1
    img = np.repeat(np.arange(n_img), np.diff(offsets))
    m = draw.cal_half
    r = np.abs(d[m] - draw.shift_half[img[m]]) / s[m]
    return group_conformal_quantile(r, img[m], n_img, alpha)
