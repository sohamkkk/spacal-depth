"""Per-image coverage histograms + qualitative interval maps (one seed).

Recomputes one split so it can show what the run CSVs cannot:
(a) the distribution of per-image coverage for K0 / K1 / V4 (S2-log and
    the factorized score), and
(b) full-resolution maps for test images whose RGB was cached, picked by
    their K1 coverage (worst, 25th percentile, median):
      RGB (+ anchors) | prediction | GT | |log error|
      K1 width | K1 missed pixels | A-both width | A-both missed pixels
    K1 uses a heuristic score with a full-resolution map (S4c-conf if the
    model has confidence, else S3-flip, else S2-log); A-both is the E11
    method with k anchors (A-hybrid on S2-log, hull A-local).
Scales are fit on head-train only, as in the study.

    python -m experiments.qualitative --dataset nyu --model dav2-indoor-large
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from scipy.ndimage import distance_transform_edt  # noqa: E402

from experiments.common import load_table, results_dir, split_tables  # noqa: E402
from experiments.e11_anchors import crc_units, eval_view, log_err, pick_tau  # noqa: E402
from src.calibrators.anchors import draw_anchors, group_conformal_quantile, group_loo_spread  # noqa: E402
from src.calibrators.registry import calibrate  # noqa: E402
from src.config import CACHE_ROOT  # noqa: E402
from src.data.cache import load_gt, load_pred  # noqa: E402
from src.data.protocols import PROTOCOLS, clamp_pred  # noqa: E402
from src.data.splits import exchange_units  # noqa: E402
from src.scores.scale_maps import SCORES, available_scores, build_scale_model, interval, residual  # noqa: E402

MAP_SCORES = ("S4c-conf", "S3-flip", "S2-log")
PICKS = {"worst": 0.0, "p25": 0.25, "median": 0.5}
ERR_MAX = 0.5  # |log error| colour range: e^0.5 = 65 % off


def per_image_cov(test, lo, hi):
    c = (test.pix["gt"] >= lo) & (test.pix["gt"] <= hi)
    return np.add.reduceat(c.astype(float), test.offsets[:-1]) / test.counts


def coverage_histograms(args, head, cal, test, table, out):
    fact = "S6-fact-conf" if "S6-fact-conf" in available_scores(table) else "S6-fact"
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    for ax, name in zip(axes, ("S2-log", fact)):
        spec = SCORES[name]
        model = build_scale_model(spec.scale, beta=args.beta, seed=args.seed).fit(head)
        s_cal, s_test = model.predict(cal), model.predict(test)
        r_cal = residual(cal, spec.space) / s_cal
        for method in ("K0", "K1", "V4"):
            res = calibrate(method, cal, r_cal, test, None, args.alpha, args.beta)
            cov = per_image_cov(test, *interval(test.pix["pred"], res.lam_pix * s_test, spec.space))
            ax.hist(cov, bins=np.linspace(0, 1, 51), histtype="step", lw=1.5,
                    label=f"{method}: P(cov≥{1 - args.beta:.2f})={np.mean(cov >= 1 - args.beta):.2f}")
        ax.axvline(1 - args.alpha, color="k", ls=":", lw=1)
        ax.set(title=f"{args.dataset} — {name}", xlabel="per-image coverage", xlim=(0, 1))
        ax.legend(fontsize=7, loc="upper left")
    axes[0].set_ylabel("# test images")
    for ext in ("png", "pdf"):
        fig.savefig(out / f"per_image_coverage__{args.dataset}__{args.model}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def full_res_field(field, p, pred, proto):
    """Same definitions as src/data/prepared.build_table, on the whole crop."""
    if field == "dflip":
        return np.abs(np.log(pred) - np.log(clamp_pred(p["pred_flip"].astype(np.float32), proto)))
    return np.maximum(p[field].astype(np.float32), 0.0)


def anchor_calibration(head, cal, k, alpha, seed):
    """A-hybrid (S2-log) lambda and tau exactly as in E11, for one anchor draw."""
    rng = np.random.default_rng([seed, k, 99])
    d_h, d_c = log_err(head), log_err(cal)
    tau = pick_tau(head, d_h, np.ones(head.n_pixels), draw_anchors(d_h, head.offsets, k, rng), alpha, rng)
    dc = draw_anchors(d_c, cal.offsets, k, rng)
    vc, sc = eval_view(cal, dc.eval)
    img = cal.image_of_pixel()
    r = np.abs(d_c - dc.shift[img]) / (dc.sigma[img] + tau)
    return crc_units(r[sc], vc.offsets, exchange_units(vc), alpha), tau


def anchor_interval(pred, gt, mask, k, lam_h, tau, alpha, rng):
    """A-both on one full-resolution image: k random valid GT pixels as anchors."""
    idx = np.flatnonzero(mask.ravel())
    k = min(k, len(idx) // 2)
    pick = rng.permutation(idx)[:k]
    d = np.log(gt.ravel()[pick].astype(np.float64)) - np.log(pred.ravel()[pick].astype(np.float64))
    shift, sigma = np.median(d), group_loo_spread(d, np.zeros(k, int), 1)[0]
    half = k // 2  # first half fixes A-local's shift, second half its width (as in draw_anchors)
    shift_half = np.median(d[:half])
    q_loc = group_conformal_quantile(np.abs(d[half:] - shift_half), np.zeros(k - half, int), 1, alpha)[0]
    q_h = lam_h * (sigma + tau)
    with np.errstate(over="ignore"):
        lo = np.minimum(pred * np.exp(shift - q_h), pred * np.exp(shift_half - q_loc))
        hi = np.maximum(pred * np.exp(shift + q_h), pred * np.exp(shift_half + q_loc))
    anchors = np.zeros(mask.shape, bool)
    anchors.ravel()[pick] = True
    return lo, hi, anchors


def densify(values, mask, radius=2.0):
    """Sparse GT (KITTI LiDAR): spread each valid pixel over a small disc so it is visible."""
    dist, (ii, jj) = distance_transform_edt(~mask, return_indices=True)
    return values[ii, jj], dist <= radius


def missed_rgb(gt_ok, covered):
    img = np.full(gt_ok.shape + (3,), 0.85)
    img[gt_ok & covered] = (0.55, 0.75, 0.95)
    img[gt_ok & ~covered] = (0.85, 0.1, 0.1)
    return img


def plot_example(path, title, rgb, pred, gt, mask, panels, anchors, sparse):
    """panels: [(name, lo, hi, eval_mask), ...] — two of them.
    Widths are shown in metres on one shared scale: the relative width of an
    s = 1 method (A-both) is constant over the image, so a relative-width map
    would be a flat colour; metres show where the interval is wide."""
    h, w = pred.shape
    show = (lambda v, m: densify(v, m)) if sparse else (lambda v, m: (v, m))
    gt_v, gt_m = show(gt, mask)
    err = np.abs(np.log(pred) - np.log(np.maximum(gt, 1e-3)))
    err_v, err_m = show(err, mask)
    vmax_d = float(np.quantile(gt[mask], 0.99))
    widths = [np.where(np.isfinite(hi), hi - lo, np.nan) for _, lo, hi, _ in panels]
    finite = np.concatenate([x[np.isfinite(x)] for x in widths])
    vmax_w = float(np.quantile(finite, 0.98)) if finite.size else 1.0

    if w / h > 2:  # KITTI-like strips: 4 rows x 2 so each panel stays readable
        fig, axs = plt.subplots(4, 2, figsize=(14, 4 * 6.2 * h / w + 1.6))
        top = [axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]]
        meth = [(axs[2, 0], axs[2, 1]), (axs[3, 0], axs[3, 1])]
    else:
        fig, axs = plt.subplots(2, 4, figsize=(17, 2 * 3.3 * h / w + 1.4))
        top = list(axs[0])
        meth = [(axs[1, 0], axs[1, 1]), (axs[1, 2], axs[1, 3])]
    cm_bad = lambda name: plt.get_cmap(name).with_extremes(bad=(0.85, 0.85, 0.85))  # noqa: E731

    top[0].imshow(rgb)
    ar, ac = np.nonzero(anchors)
    top[0].scatter(ac, ar, s=14, c="cyan", edgecolors="k", linewidths=0.5)
    top[0].set_title(f"RGB + {len(ar)} anchors (cyan)", fontsize=9)
    for ax, v, m, cmap, vmax, label in (
        (top[1], pred, np.ones_like(mask), "magma", vmax_d, "prediction (m)"),
        (top[2], gt_v, gt_m, "magma", vmax_d, "ground truth (m)"),
        (top[3], err_v, err_m, "inferno", ERR_MAX, "|log error|"),
    ):
        im = ax.imshow(np.ma.masked_where(~m, v), cmap=cm_bad(cmap), vmin=0, vmax=vmax)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        ax.set_title(label, fontsize=9)

    for j, ((name, lo, hi, ev), wmap, (ax_w, ax_m)) in enumerate(zip(panels, widths, meth)):
        im = ax_w.imshow(np.ma.masked_invalid(wmap), cmap=cm_bad("viridis"), vmin=0, vmax=vmax_w)
        fig.colorbar(im, ax=ax_w, fraction=0.046, pad=0.02, label="m")
        with np.errstate(invalid="ignore", divide="ignore"):
            rel = float(np.nanmedian(wmap / pred))
        ax_w.set_title(f"{name}: width in m (median {rel:.2f} × depth)", fontsize=9)
        covered = (gt >= lo) & (gt <= hi)
        cov = covered[ev].mean()
        cv, cm = show(covered.astype(float), ev)
        ax_m.imshow(missed_rgb(cm, cv > 0.5))
        if j == 1:
            ax_m.scatter(ac, ar, s=14, c="k", marker="x", linewidths=1.0)
        ax_m.set_title(f"{name}: missed pixels (coverage {cov:.3f})", fontsize=9)
    meth[1][1].legend(handles=[Patch(color=(0.55, 0.75, 0.95), label="covered"), Patch(color=(0.85, 0.1, 0.1), label="missed"),
                              Patch(color=(0.85, 0.85, 0.85), label="no GT"),
                              plt.Line2D([], [], color="k", marker="x", ls="", label="anchor (not scored)")],
                     fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=4, frameon=False)
    for ax in axs.ravel():
        ax.set_xlim(-0.5, w - 0.5)
        ax.set_ylim(h - 0.5, -0.5)
        ax.axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--k", type=int, default=20, help="anchors per image for the A-both panels")
    ap.add_argument("--picks", nargs="+", default=list(PICKS), choices=list(PICKS))
    args = ap.parse_args()

    table = load_table(args.dataset, args.model)
    table.pix.pop("label", None)
    head, cal, test = split_tables(table, args.seed)
    out = results_dir("figures")
    coverage_histograms(args, head, cal, test, table, out)

    map_score = next(n for n in MAP_SCORES if n in available_scores(table))
    spec = SCORES[map_score]
    model = build_scale_model(spec.scale, beta=args.beta, seed=args.seed).fit(head)
    lam = calibrate("K1", cal, residual(cal, spec.space) / model.predict(cal), test, None, args.alpha).lam_summary
    cov_test = per_image_cov(test, *interval(test.pix["pred"], lam * model.predict(test), spec.space))
    lam_h, tau = anchor_calibration(head, cal, args.k, args.alpha, args.seed)

    gt_dir = CACHE_ROOT / args.dataset / "gt"
    has_rgb = []
    for i, sid in enumerate(test.img["sample_id"]):
        f = gt_dir / f"{sid}.npz"
        if f.exists():
            with np.load(f) as z:
                if "rgb" in z.files:
                    has_rgb.append(i)
    if not has_rgb:
        raise SystemExit(f"no test image with a cached RGB under {gt_dir}: set SPACAL_CACHE_ROOT")
    order = sorted(has_rgb, key=lambda i: cov_test[i])
    chosen = {}
    for p in args.picks:
        i = order[int(round(PICKS[p] * (len(order) - 1)))]
        if i not in chosen.values():
            chosen[p] = i
    print(f"{len(has_rgb)} test images with RGB; K1 lambda {lam:.3f} ({map_score}); "
          f"A-hybrid lambda {lam_h:.3f}, tau {tau:.3f} (k={args.k})")

    proto = PROTOCOLS[args.dataset]
    rng = np.random.default_rng(args.seed)
    for pick, i in chosen.items():
        sid = str(test.img["sample_id"][i])
        g, p = load_gt(args.dataset, sid), load_pred(args.dataset, args.model, sid)
        pred = clamp_pred(p["pred"].astype(np.float32), proto).astype(np.float64)
        gt, mask = g["gt"].astype(np.float64), g["mask"].astype(bool)
        s = model.predict_field(full_res_field(model.field, p, pred, proto)) if hasattr(model, "field") else np.ones_like(pred)
        lo_k1, hi_k1 = interval(pred, lam * s, spec.space)
        lo_a, hi_a, anchors = anchor_interval(pred, gt, mask, args.k, lam_h, tau, args.alpha, rng)
        panels = [(f"K1 · {map_score}", lo_k1, hi_k1, mask), (f"A-both · k={args.k}", lo_a, hi_a, mask & ~anchors)]
        title = (f"{args.dataset} / {args.model} — {sid} — {pick} K1 image "
                 f"(lattice coverage {cov_test[i]:.3f}, target {1 - args.alpha:.2f})")
        plot_example(out / f"qual__{args.dataset}__{args.model}__{pick}__{sid}", title, g["rgb"], pred, gt, mask,
                     panels, anchors, sparse=mask.mean() < 0.3)
        print(f"  {pick:>6} {sid}: K1 {((gt >= lo_k1) & (gt <= hi_k1))[mask].mean():.3f}  "
              f"A-both {((gt >= lo_a) & (gt <= hi_a))[mask & ~anchors].mean():.3f}")
    print(f"wrote histogram and {len(chosen)} maps to {out}")


if __name__ == "__main__":
    main()
