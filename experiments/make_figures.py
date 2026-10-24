"""Paper figures from result CSVs (anything missing is skipped).

    python -m experiments.make_figures
Writes results/figures/*.png and *.pdf.
"""

import glob

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.common import results_dir  # noqa: E402
from experiments.summarize import validity_metric  # noqa: E402
from src.config import RESULTS_ROOT  # noqa: E402

OUT = None


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}")


def fig_e1():
    p = RESULTS_ROOT / "e1" / "summary.csv"
    if not p.exists():
        return
    s = pd.read_csv(p)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for cal, mk in (("K0", "o"), ("K1", "s"), ("V4", "^")):
        t = s[(s.calibrator == cal) & (s.ell == s.ell.max())]
        axes[0].plot(t.sigma_g, t.cov_img_sd, mk + "-", label=cal)
        axes[1].plot(t.sigma_g, t.v4_rate, mk + "-", label=cal)
    axes[0].set(xlabel="global scale error SD $\\sigma_g$", ylabel="per-image coverage SD", title="Spread across images")
    axes[1].axhline(0.9, color="k", ls=":", lw=1)
    axes[1].set(xlabel="global scale error SD $\\sigma_g$", ylabel="P(image coverage $\\geq 1-\\beta$)", title="Image event (V4)")
    axes[0].legend()
    save(fig, "e1_synthetic")

    fig, ax = plt.subplots(figsize=(5, 3.6))
    t = s[s.calibrator == "K1"]
    for ell, tt in t.groupby("ell"):
        ax.plot(tt.sigma_g, tt.cov_run_sd, "o-", label=f"$\\ell$={ell:g}")
    ax.axhline(t.beta_sd_n_images.iloc[0], color="k", ls="--", lw=1, label="Beta, n = #images")
    ax.axhline(t.beta_sd_n_pixels.iloc[0], color="gray", ls=":", lw=1, label="Beta, n = #pixels")
    ax.set(xlabel="$\\sigma_g$", ylabel="run-to-run SD of coverage", yscale="log", title="K1: spread follows images")
    ax.legend(fontsize=7)
    save(fig, "e1_run_spread")


def fig_main(exp_id="main", alpha=0.1):
    files = glob.glob(str(RESULTS_ROOT / exp_id / "*" / "runs_*.csv"))
    if not files:
        return
    runs = pd.concat(map(pd.read_csv, files))
    runs = runs[np.isclose(runs.alpha, alpha)]
    runs["validity"] = [r[validity_metric(r.calibrator)] for _, r in runs.iterrows()]
    for (ds, model), t in runs.groupby(["dataset", "model"]):
        agg = t.groupby(["score", "calibrator"]).agg(w=("median_rel_width", "mean"), v=("validity", "mean"),
                                                     pix=("cov_pixel", "mean")).reset_index()
        agg = agg[np.isfinite(agg.w) & (agg.calibrator != "raw")]
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        cmap = plt.get_cmap("tab20")
        for i, (cal, tt) in enumerate(agg.groupby("calibrator")):
            ax.scatter(tt.w, tt.v, label=cal, s=28, color=cmap(i), marker="osD^v<>P*Xh"[i % 11])
            for _, r in tt.iterrows():
                ax.annotate(r.score.split("-")[0], (r.w, r.v), fontsize=5, alpha=0.6,
                            xytext=(2, 2), textcoords="offset points")
        ax.axhline(1 - alpha, color="k", ls=":", lw=1)
        lo = min(0.8, agg.v.min() - 0.01) if len(agg) else 0.8
        ax.set(xscale="log", ylim=(lo, 1.005), xlabel="median relative width (width / prediction), log scale",
               ylabel="coverage of the calibrator's target notion", title=f"{ds} / {model}, $\\alpha$={alpha}")
        for axis in (ax.xaxis.set_major_formatter, ax.xaxis.set_minor_formatter):
            axis(matplotlib.ticker.FormatStrFormatter("%.2g"))
        ax.tick_params(axis="x", which="minor", labelsize=6)
        ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
        save(fig, f"main_width_vs_validity__{ds}__{model}")

        k = t[(t.calibrator.isin(["K0", "K1", "V4"])) & (t.score == "S2-log")]
        if not k.empty:
            fig, ax = plt.subplots(figsize=(5, 3.6))
            m = k.groupby("calibrator")[["cov_pixel", "cov_image_mean", "frac_img_cov_below_1ma_m10", "v4_rate"]].mean()
            m.plot.bar(ax=ax)
            ax.axhline(1 - alpha, color="k", ls=":", lw=1)
            ax.set(title=f"Pooled vs per-image ({ds}, S2-log)", ylim=(0, 1))
            ax.legend(fontsize=6)
            save(fig, f"pooled_vs_image__{ds}__{model}")

    gfiles = glob.glob(str(RESULTS_ROOT / exp_id / "*" / "groups_*.csv"))
    if gfiles:
        g = pd.concat(map(pd.read_csv, gfiles))
        g = g[np.isclose(g.alpha, alpha) & (g.group_family == "pred_depth") & g.calibrator.isin(["K1", "K3-depth"])]
        for (ds, model), t in g.groupby(["dataset", "model"]):
            for score in ("S2-log", "S5-learned"):
                tt = t[t.score == score]
                if tt.empty:
                    continue
                piv = tt.groupby(["group", "calibrator"]).cov_image_mean.mean().unstack()
                fig, ax = plt.subplots(figsize=(5, 3.4))
                piv.plot.bar(ax=ax)
                ax.axhline(1 - alpha, color="k", ls=":", lw=1)
                ax.set(ylabel="image-avg coverage in bin", xlabel="predicted depth bin (m)",
                       title=f"Mondrian vs marginal ({ds}, {score})", ylim=(0.5, 1))
                save(fig, f"groups_depth__{ds}__{model}__{score}")


def fig_e8():
    for p in glob.glob(str(RESULTS_ROOT / "e8" / "*" / "summary.csv")):
        s = pd.read_csv(p)
        name = p.split("/")[-2]
        fig, ax = plt.subplots(figsize=(5, 3.6))
        for (score, cal), t in s.groupby(["score", "calibrator"]):
            ax.plot(t.n_cal_images, t.cov_run_sd, "o-", label=f"{score}/{cal}")
        ax.plot(s.n_cal_images, s.beta_sd_n_images, "k--", lw=1, label="Beta, n=#images")
        ax.plot(s.n_cal_images, s.beta_sd_n_pixels, ":", color="gray", lw=1, label="Beta, n=#pixels")
        ax.set(xscale="log", yscale="log", xlabel="# calibration images", ylabel="run-to-run SD of coverage", title=name)
        ax.legend(fontsize=6)
        save(fig, f"e8_calsize__{name}")


def fig_e5():
    for p in glob.glob(str(RESULTS_ROOT / "e5" / "*" / "runs.csv")):
        r = pd.read_csv(p)
        name = p.split("/")[-2]
        m = r.groupby(["eval", "calibrator"]).cov_image_mean.mean().unstack()
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        m.plot.bar(ax=ax)
        ax.axhline(1 - r.alpha.iloc[0], color="k", ls=":", lw=1)
        ax.set(ylabel="image-avg coverage", title=name, ylim=(0.4, 1))
        save(fig, f"e5_shift__{name}")


E11_ORDER = ("nyu__unidepthv2-large", "nyu__dav2-indoor-large", "nyu__depthpro", "nyu__metric3d-large",
             "kitti__unidepthv2-large", "kitti__dav2-outdoor-large", "kitti__depthpro", "kitti__metric3d-large",
             "diode_indoor__dav2-indoor-large",
             "diode_outdoor__dav2-outdoor-large")
E11_METHODS = (("A-rescale", "tab:blue", "o"), ("A-hybrid", "tab:orange", "s"),
               ("A-local", "tab:green", "^"), ("A-both", "tab:red", "D"))
E11_ROWS = (("cov_image_mean", "image-avg coverage", False), ("cov_image_sd", "per-image coverage SD", False),
            ("cond_short_frac", "share of short images", False), ("rel_width", "mean relative width", True))


def fig_e11(base="S2-log", alpha=0.1):
    """Anchor budget: coverage, per-image spread, short-image share and width against k.

    k is placed on evenly spaced ticks; infinite widths (k below a method's minimum) are left out.
    """
    found = {p.split("/")[-2]: pd.read_csv(p) for p in glob.glob(str(RESULTS_ROOT / "e11" / "*" / "summary.csv"))}
    found = {n: s for n, s in found.items()
             if (np.isfinite(s.rel_width) & s.base.isin([base, "S5-learned" if base == "S5-shape" else base])).any()}
    names = [n for n in E11_ORDER if n in found] + sorted(set(found) - set(E11_ORDER))
    if not names:
        return
    fig, axes = plt.subplots(len(E11_ROWS), len(names), figsize=(2.6 * len(names), 2.1 * len(E11_ROWS)),
                             sharex=True, squeeze=False)
    for j, name in enumerate(names):
        s = found[name].copy()
        s["k"] = s.k.astype(str)
        if base == "S5-shape":  # its k = 0 reference is stored under the learned score it is built from
            s.loc[(s.base == "S5-learned") & (s.k == "0"), "base"] = base
        s = s[np.isclose(s.alpha, alpha) & (s.base == base)]
        ks = [k for k in s.k.unique() if k != "oracle"]
        ks = sorted(ks, key=int)
        pos = {k: i for i, k in enumerate(ks)}
        ref = s[(s.method == "A-rescale") & (s.k == "0")]
        orc = s[(s.method == "A-rescale") & (s.k == "oracle")]
        for i, (col, label, logy) in enumerate(E11_ROWS):
            ax = axes[i, j]
            for method, color, mk in E11_METHODS:
                t = s[(s.method == method) & s.k.isin(ks)]
                t = t[np.isfinite(t.rel_width)]
                if t.empty:
                    continue
                ax.plot([pos[k] for k in t.k], t[col], mk + "-", color=color, ms=3.5, lw=1.2, label=method)
            if not ref.empty and np.isfinite(ref.rel_width.iloc[0]):
                ax.axhline(ref[col].iloc[0], color="gray", ls=":", lw=1, label="no anchors")
            if not orc.empty and np.isfinite(orc.rel_width.iloc[0]):
                ax.axhline(orc[col].iloc[0], color="k", ls="--", lw=0.8, label="oracle scale")
            if col == "cov_image_mean":
                ax.axhline(1 - alpha, color="k", lw=0.6)
            if logy:
                ax.set_yscale("log")
                lo, hi = ax.get_ylim()
                ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2g"))
                ax.yaxis.set_minor_formatter(matplotlib.ticker.FormatStrFormatter("%.2g") if hi / lo < 10
                                             else matplotlib.ticker.NullFormatter())
            ax.grid(alpha=0.25, lw=0.5)
            if j == 0:
                ax.set_ylabel(label, fontsize=8)
            ax.tick_params(labelsize=7)
        axes[0, j].set_title(name.replace("__", "\n"), fontsize=8)
        axes[-1, j].set_xticks(range(len(ks)), ks)
        axes[-1, j].set_xlabel("anchors per image k", fontsize=8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    for ax in axes.flat:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            if lab not in labels:
                handles.append(h)
                labels.append(lab)
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, f"e11_anchor_budget__{base}")


E14_LAYOUTS = ("uniform", "rows", "cluster", "edges", "near")
E14_BINS = ("<0.02", "0.02-0.05", "0.05-0.1", "0.1-0.2", ">=0.2")
E14_METHODS = (("A-hybrid", "tab:orange", "s"), ("A-local", "tab:green", "^"), ("A-dist-auto", "tab:purple", "o"))


def fig_e14(base="S2-log", alpha=0.1, k_bins=50):
    """A-dist: coverage by distance to the nearest anchor (k = k_bins) and width relative to A-hybrid against k,
    per anchor layout; lines are the mean (top) or median (bottom) over dataset/model pairs, bands their range."""
    dist, summ = [], []
    for p in glob.glob(str(RESULTS_ROOT / "e14" / "*" / "dist.csv")):
        parts = p.split("/")[-2].split("__")
        layout = parts[2] if len(parts) > 2 else "uniform"
        dist.append(pd.read_csv(p).assign(pair="__".join(parts[:2]), layout=layout))
        summ.append(pd.read_csv(p.replace("dist.csv", "summary.csv")).assign(pair="__".join(parts[:2]), layout=layout))
    if not dist:
        return
    dist, summ = pd.concat(dist, ignore_index=True), pd.concat(summ, ignore_index=True)
    dist = dist[np.isclose(dist.alpha, alpha) & (dist.base == base) & (dist.k.astype(str) == str(k_bins))]
    summ = summ[np.isclose(summ.alpha, alpha) & (summ.base == base)]
    cov = (dist.groupby(["pair", "layout", "method", "dist_bin"])
           .agg(coverage=("cov_image_mean", "mean"), n_img=("n_img", "mean")).reset_index())
    cov = cov[cov.n_img >= 20]
    h = summ[summ.method == "A-hybrid"].set_index(["pair", "layout", "k"]).median_rel_width
    a = summ[summ.method == "A-dist-auto"].set_index(["pair", "layout", "k"]).median_rel_width
    ratio = (a / h).dropna().rename("ratio").reset_index()

    layouts = [lay for lay in E14_LAYOUTS if lay in set(cov.layout)]
    fig, axes = plt.subplots(2, len(layouts), figsize=(2.7 * len(layouts), 5.0), squeeze=False)
    for j, lay in enumerate(layouts):
        ax = axes[0, j]
        for method, color, mk in E14_METHODS:
            t = cov[(cov.layout == lay) & (cov.method == method)]
            g = t.groupby("dist_bin").coverage.agg(["mean", "min", "max"]).reindex(E14_BINS)
            if g["mean"].notna().sum() == 0:
                continue
            x = np.arange(len(E14_BINS))
            ax.plot(x, g["mean"], mk + "-", color=color, ms=3.5, lw=1.2, label=method)
            ax.fill_between(x, g["min"], g["max"], color=color, alpha=0.15, lw=0)
        ax.axhline(1 - alpha, color="k", lw=0.6)
        ax.set_xticks(range(len(E14_BINS)), E14_BINS, rotation=35, fontsize=6)
        ax.set_xlabel("distance to nearest anchor / diagonal", fontsize=7)
        ax.set_title(lay, fontsize=9)
        ax.grid(alpha=0.25, lw=0.5)
        ax.tick_params(labelsize=7)

        ax = axes[1, j]
        t = ratio[ratio.layout == lay]
        g = t.groupby("k").ratio.agg(["median", "min", "max"]).sort_index()
        x = np.arange(len(g))
        ax.plot(x, g["median"], "o-", color="tab:purple", ms=3.5, lw=1.2)
        ax.fill_between(x, g["min"], g["max"], color="tab:purple", alpha=0.15, lw=0)
        ax.axhline(1, color="k", lw=0.6)
        ax.set_xticks(x, [str(k) for k in g.index], fontsize=7)
        ax.set_xlabel("anchors per image k", fontsize=8)
        ax.grid(alpha=0.25, lw=0.5)
        ax.tick_params(labelsize=7)
    axes[0, 0].set_ylabel(f"image-avg coverage in bin\n(k = {k_bins})", fontsize=8)
    axes[1, 0].set_ylabel("A-dist-auto / A-hybrid\nmedian width", fontsize=8)
    for i in range(2):
        lo = min(ax.get_ylim()[0] for ax in axes[i])
        hi = max(ax.get_ylim()[1] for ax in axes[i])
        for ax in axes[i]:
            ax.set_ylim(lo, hi)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(h_pad=2.2)
    save(fig, f"e14_distance__{base}")


E15_METHODS = ("A-rescale", "A-hybrid", "A-dist")
E15_CAL = (*E14_LAYOUTS, "mix", "max")


def fig_e15(alpha=0.1, k=50):
    """Calibration vs test anchor layout: coverage of each method calibrated under each layout (rows) and tested
    under each layout (columns), mean over dataset/model pairs; last panel the width cost of @max over matched."""
    summ = []
    for p in glob.glob(str(RESULTS_ROOT / "e15" / "*" / "summary.csv")):
        ds, model, test = p.split("/")[-2].split("__")
        summ.append(pd.read_csv(p).assign(pair=f"{ds}__{model}", test=test))
    if not summ:
        return
    s = pd.concat(summ, ignore_index=True)
    s = s[np.isclose(s.alpha, alpha) & (s.k.astype(str) == str(k))]
    s["fam"], s["cal"] = s.method.str.split("@").str[0], s.method.str.split("@").str[1]
    tests = [t for t in E14_LAYOUTS if t in set(s.test)]
    fig, axes = plt.subplots(1, 6, figsize=(14, 3.4),
                             gridspec_kw={"width_ratios": [1, 1, 1, 0.05, 0.12, 1.3], "wspace": 0.3})
    axes[4].axis("off")
    for ax, fam in zip(axes, E15_METHODS):
        m = (s[s.fam == fam].pivot_table(index="cal", columns="test", values="cov_image_mean", aggfunc="mean")
             .reindex(index=list(E15_CAL), columns=tests))
        im = ax.imshow(m.values, cmap="RdBu", vmin=1 - alpha - 0.12, vmax=1 - alpha + 0.12, aspect="auto")
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                v = m.values[i, j]
                if np.isfinite(v):
                    bold = "bold" if E15_CAL[i] == tests[j] else "normal"
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5, weight=bold)
        ax.axhline(len(E14_LAYOUTS) - 0.5, color="k", lw=0.8)
        ax.set_xticks(range(len(tests)), tests, rotation=35, fontsize=7)
        ax.set_yticks(range(len(E15_CAL)), E15_CAL if ax is axes[0] else [], fontsize=7)
        ax.set_xlabel("test layout", fontsize=8)
        ax.set_title(fam, fontsize=9)
    axes[0].set_ylabel("calibration layout", fontsize=8)
    cb = fig.colorbar(im, cax=axes[3])
    cb.set_label(f"image-avg coverage (target {1 - alpha:.2f})", fontsize=7)
    cb.ax.tick_params(labelsize=6)

    ax = axes[5]
    key = ["pair", "test"]
    width = 0.8 / len(E15_METHODS)
    for i, (fam, color) in enumerate(zip(E15_METHODS, ("tab:gray", "tab:blue", "tab:purple"))):
        matched = s[(s.fam == fam) & (s.cal == s.test)].set_index(key).median_rel_width
        mx = s[(s.fam == fam) & (s.cal == "max")].set_index(key).median_rel_width
        r = (mx / matched).groupby("test").agg(["median", "min", "max"]).reindex(tests)
        x = np.arange(len(tests)) + (i - 1) * width
        ax.bar(x, r["median"], width, color=color, label=fam)
        ax.errorbar(x, r["median"], yerr=[r["median"] - r["min"], r["max"] - r["median"]], fmt="none",
                    ecolor="k", lw=0.6, capsize=1.5)
    ax.axhline(1, color="k", lw=0.6)
    ax.set_xticks(range(len(tests)), tests, rotation=35, fontsize=7)
    ax.set_xlabel("test layout", fontsize=8)
    ax.set_ylabel("@max / matched median width", fontsize=8)
    ax.set_title("cost of covering every layout", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    ax.grid(alpha=0.25, lw=0.5, axis="y")
    ax.tick_params(labelsize=7)
    save(fig, f"e15_layouts__k{k}")


def main():
    global OUT
    OUT = results_dir("figures")
    fig_e1()
    fig_main()
    fig_e8()
    fig_e5()
    fig_e11()
    fig_e11(base="S5-shape")
    fig_e14()
    fig_e14(base="S5-shape")
    fig_e15()
    print(f"figures in {OUT}")


if __name__ == "__main__":
    main()
