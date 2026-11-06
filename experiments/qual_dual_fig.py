"""Qualitative A-local vs A-dual figures from experiments.qual_dual: render stage.

    python -m experiments.qual_dual_fig sheet                    # contact sheet of all candidates (choosing frames)
    python -m experiments.qual_dual_fig teaser --id <sample_id>  # paper teaser strip -> paper/figures/teaser_qual.pdf
    python -m experiments.qual_dual_fig sensor --ids <id> ...    # Supp. sensor figure -> paper/figures/sensor_qual.pdf
    python -m experiments.qual_dual_fig layouts --id <id>        # one frame under all five layouts (report)
    python -m experiments.qual_dual_fig models --ids <id> ...    # RGB, laser, iPad, two models, error (report)

Backgrounds are the RGB frames of scripts/export_arkit_frames.py (<frames>/<sample_id>_rgb.png) when present,
else the laser depth on the stride-8 grid. Miss maps colour the evaluation pixels (neither structured anchor
nor probe) whose laser truth lies outside the interval.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402
from scipy.stats import beta as beta_dist  # noqa: E402

from experiments.common import results_dir  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
STRIDE = 8
FAR, NEAR = np.array([0.835, 0.369, 0.0]), np.array([0.0, 0.447, 0.698])  # Okabe-Ito vermillion / blue
LOCAL_C, DUAL_C = "tab:green", "tab:blue"
STYLE = {"font.size": 6, "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
         "xtick.major.size": 2, "ytick.major.size": 2, "pdf.fonttype": 42}


class Frames:
    def __init__(self, run_dir: Path, frames_dir: Path):
        self.maps = np.load(run_dir / "maps.npz")
        self.per_img = pd.read_csv(run_dir / "per_image.csv")
        self.cand = pd.read_csv(run_dir / "candidates.csv")
        self.frames_dir = frames_dir

    def get(self, sid: str, key: str) -> np.ndarray:
        return self.maps[f"{sid}/{key}"]

    def shape(self, sid: str) -> tuple[int, int]:
        r = self.per_img[self.per_img.sample_id == sid].iloc[0]
        return int(r.height), int(r.width)

    def grid(self, sid: str, values: np.ndarray, fill=np.nan) -> np.ndarray:
        h, w = self.shape(sid)
        out = np.full(((h + STRIDE - 1) // STRIDE, (w + STRIDE - 1) // STRIDE) + values.shape[1:], fill,
                      dtype=np.result_type(values, np.float32))
        out[self.get(sid, "row") // STRIDE, self.get(sid, "col") // STRIDE] = values
        return out

    def background(self, sid: str, force_depth: bool = False) -> tuple[np.ndarray, bool]:
        p = self.frames_dir / f"{sid}_rgb.png"
        if p.exists() and not force_depth:
            return np.asarray(Image.open(p).convert("RGB")) / 255.0, True
        d = self.grid(sid, self.get(sid, "gt"))
        lo, hi = np.nanpercentile(d, [2, 98])
        rgb = plt.get_cmap("Greys_r")(0.15 + 0.7 * np.clip((d - lo) / (hi - lo), 0, 1))[..., :3]
        rgb[np.isnan(d)] = 0.0
        return rgb, False

    def miss(self, sid: str, cell: str, meth: str) -> dict:
        gt, ev = self.get(sid, "gt"), self.get(sid, "eval")
        lo, hi = self.get(sid, f"{cell}/{meth}/lo"), self.get(sid, f"{cell}/{meth}/hi")
        far, near = ev & (gt > hi), ev & (gt < lo)
        return {"far": far, "near": near, "cov": 1 - (far | near)[ev].mean(),
                "width": float(np.median(((hi - lo) / gt)[ev]))}


def show_background(ax, fr: Frames, sid: str, fade: float = 0.0, force_depth: bool = False):
    h, w = fr.shape(sid)
    img, _ = fr.background(sid, force_depth)
    if fade:
        grey = img.mean(axis=-1, keepdims=True)
        img = (1 - fade) * (0.5 * img + 0.5 * grey) + fade
    ax.imshow(img, extent=(0, w, h, 0), interpolation="nearest" if img.shape[0] < h else "antialiased")
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.4)


def overlay_miss(ax, fr: Frames, sid: str, m: dict, alpha: float = 0.85):
    h, w = fr.shape(sid)
    rgba = np.zeros((len(m["far"]), 4))
    rgba[m["far"], :3], rgba[m["near"], :3] = FAR, NEAR
    rgba[m["far"] | m["near"], 3] = alpha
    ax.imshow(fr.grid(sid, rgba, fill=0.0), extent=(0, w, h, 0), interpolation="nearest")


def draw_anchors(ax, fr: Frames, sid: str, probes: bool = True, scale: float = 1.0):
    r, c = fr.get(sid, "row") + STRIDE / 2, fr.get(sid, "col") + STRIDE / 2
    s = fr.get(sid, "structured")
    ax.scatter(c[s], r[s], s=1.6 * scale, c="white", edgecolors="black", linewidths=0.25 * scale, zorder=3)
    if probes:
        p = fr.get(sid, "probe")
        ax.scatter(c[p], r[p], s=9 * scale, marker="x", c="#F0E442", linewidths=0.9 * scale, zorder=4,
                   path_effects=[pe.withStroke(linewidth=1.8 * scale, foreground="black")])


def label(ax, text: str, loc: str = "lower left"):
    x, ha = (0.03, "left") if "left" in loc else (0.97, "right")
    y, va = (0.04, "bottom") if "lower" in loc else (0.96, "top")
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=5.2, color="black",
            bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "none", "alpha": 0.85})


def coverage_hist(ax, fr: Frames, cell: str, n_probes: int, alpha: float):
    bins = np.linspace(0, 1, 41)
    for meth, color, name in (("A-local", LOCAL_C, "A-local"), ("A-dual", DUAL_C, "A-dual")):
        v = fr.per_img[f"cov|{cell}|{meth}"].dropna()
        ax.hist(v, bins=bins, density=True, histtype="stepfilled", color=color, alpha=0.25, lw=0)
        ax.hist(v, bins=bins, density=True, histtype="step", color=color, lw=0.8, label=name)
    r = int(np.ceil((n_probes + 1) * (1 - alpha)))
    x = np.linspace(0.3, 1, 300)
    ax.plot(x, beta_dist(r, n_probes + 1 - r).pdf(x), color="k", lw=0.6, ls=(0, (2, 1)),
            label=f"law Beta({r}, {n_probes + 1 - r})")
    ax.axvline(1 - alpha, color="0.35", lw=0.5, ls=":")
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("per-image coverage $\\mathrm{Cov}_i$", fontsize=5.5, labelpad=1)
    ax.tick_params(labelsize=5, pad=1)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=4.6, frameon=False, loc="upper left", handlelength=1.3, borderpad=0.1, labelspacing=0.2)


def save(fig, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.01)
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print(f"wrote {out}")


def teaser(fr: Frames, sid: str, cell: str, n_probes: int, alpha: float, out: Path):
    h, w = fr.shape(sid)
    ratios = [w / h] * 3 + [1.7]
    fig = plt.figure(figsize=(6.9, 6.9 / (sum(ratios) * 1.04) + 0.12))
    gs = fig.add_gridspec(1, 4, width_ratios=ratios, wspace=0.04)
    axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
    show_background(axes[0], fr, sid)
    draw_anchors(axes[0], fr, sid)
    axes[0].set_title("(a) anchors $\\bullet$, probes $\\times$", fontsize=5.8, pad=2)
    for ax, meth, tag in ((axes[1], "A-local", "(b) A-local: anchors calibrate"),
                          (axes[2], "A-dual", "(c) A-dual: probes calibrate")):
        m = fr.miss(sid, cell, meth)
        show_background(ax, fr, sid, fade=0.35)
        overlay_miss(ax, fr, sid, m)
        draw_anchors(ax, fr, sid, probes=meth == "A-dual", scale=0.8)
        ax.set_title(tag, fontsize=5.8, pad=2)
        label(ax, f"coverage {m['cov']:.2f}   width {100 * m['width']:.0f}%")
    pos = axes[3].get_position()
    axes[3].set_position([pos.x0 + 0.03, pos.y0 + 0.14 * pos.height, pos.width - 0.03, 0.86 * pos.height])
    coverage_hist(axes[3], fr, cell, n_probes, alpha)
    axes[3].set_title(f"(d) coverage over all {len(fr.per_img)} test images", fontsize=5.8, pad=2)
    save(fig, out)


def sensor(fr: Frames, sids: list[str], cells: tuple[str, str], out: Path):
    have_rgb = all(fr.background(s)[1] for s in sids)
    cols = (["RGB"] if have_rgb else []) + ["laser depth\n(truth)", "log(iPad / laser)\nblue: iPad nearer",
                                            "A-dual: iPad anchors,\nlaser probes", "A-dual: iPad anchors,\niPad probes"]
    fig, axes = plt.subplots(len(sids), len(cols), figsize=(6.9, 6.9 / len(cols) * 0.75 * len(sids) + 0.15),
                             squeeze=False, gridspec_kw={"wspace": 0.03, "hspace": 0.04})
    for i, sid in enumerate(sids):
        row = list(axes[i])
        if have_rgb:
            show_background(row.pop(0), fr, sid)
        show_background(row[0], fr, sid, force_depth=True)
        h, w = fr.shape(sid)
        lid, gt = fr.get(sid, "lidar"), fr.get(sid, "gt")
        err = np.where(lid > 0, np.log(np.where(lid > 0, lid, 1) / gt), np.nan)
        e = fr.grid(sid, err)
        row[1].set_facecolor("0.85")
        row[1].imshow(e, extent=(0, w, h, 0), cmap="RdBu_r", vmin=-0.15, vmax=0.15, interpolation="nearest")
        row[1].set_xticks([])
        row[1].set_yticks([])
        label(row[1], f"median |err| {100 * np.nanmedian(np.abs(err)):.1f}%")
        for ax, cell in zip(row[2:], cells):
            m = fr.miss(sid, cell, "A-dual")
            show_background(ax, fr, sid, fade=0.35)
            overlay_miss(ax, fr, sid, m)
            draw_anchors(ax, fr, sid, scale=0.8)
            label(ax, f"coverage {m['cov']:.2f}")
        for ax in axes[i]:
            ax.set_aspect("auto")
    for ax, c in zip(axes[0], cols):
        ax.set_title(c, fontsize=6, pad=2)
    save(fig, out)


LAYOUTS = ("uniform", "rows", "cluster", "edges", "near")


def layouts(run_prefix: str, frames_dir: Path, sid: str, cell: str, out: Path):
    frs = {lay: Frames(results_dir("qual_dual") / f"{run_prefix}__{lay}", frames_dir) for lay in LAYOUTS}
    h, w = frs["uniform"].shape(sid)
    fig, axes = plt.subplots(3, len(LAYOUTS), figsize=(6.9, 6.9 / len(LAYOUTS) * h / w * 3 + 0.2), squeeze=False,
                             gridspec_kw={"wspace": 0.03, "hspace": 0.05})
    rows = ("anchors $\\bullet$, probes $\\times$", "A-local misses", "A-dual misses")
    for j, lay in enumerate(LAYOUTS):
        fr = frs[lay]
        show_background(axes[0, j], fr, sid)
        draw_anchors(axes[0, j], fr, sid, scale=0.8)
        axes[0, j].set_title(lay, fontsize=6.5, pad=2)
        for i, meth in ((1, "A-local"), (2, "A-dual")):
            m = fr.miss(sid, cell, meth)
            show_background(axes[i, j], fr, sid, fade=0.35)
            overlay_miss(axes[i, j], fr, sid, m)
            draw_anchors(axes[i, j], fr, sid, probes=meth == "A-dual", scale=0.6)
            label(axes[i, j], f"cov {m['cov']:.2f}  w {100 * m['width']:.0f}%")
    for i, r in enumerate(rows):
        axes[i, 0].set_ylabel(r, fontsize=6, labelpad=2)
    save(fig, out)


class Prepared:
    """Per-image lattice of a prepared table (no torch): row, col, prediction and laser truth."""

    def __init__(self, dataset: str, model: str):
        root = Path(os.environ.get("SPACAL_PREPARED_ROOT", "prepared_new"))
        z = np.load((root if root.is_absolute() else ROOT / root) / f"{dataset}__{model}.npz", allow_pickle=True)
        self.ids = {s: i for i, s in enumerate(z["img__sample_id"])}
        self.off = z["offsets"]
        self.hw = np.stack([z["img__height"], z["img__width"]], axis=1)
        self.cols = {k: z[f"pix__{k}"] for k in ("row", "col", "pred", "gt")}

    def get(self, sid: str) -> dict:
        i = self.ids[sid]
        return {k: v[self.off[i]:self.off[i + 1]] for k, v in self.cols.items()}

    def shape(self, sid: str) -> tuple[int, int]:
        h, w = self.hw[self.ids[sid]]
        return int(h), int(w)


def models(frames_dir: Path, sids: list[str], model_names: dict[str, str], out: Path):
    preds = {name: Prepared("arkitscenes", m) for m, name in model_names.items()}
    first = next(iter(preds))
    cols = ["RGB", "laser depth (truth)", "iPad LiDAR", *preds, f"|log error|, {first}"]
    fig, axes = plt.subplots(len(sids), len(cols), figsize=(6.9, 6.9 / len(cols) * 0.75 * len(sids) + 0.15),
                             squeeze=False, gridspec_kw={"wspace": 0.03, "hspace": 0.04})
    cmap = plt.get_cmap("turbo").copy()
    cmap.set_bad("0.85")
    for i, sid in enumerate(sids):
        h, w = preds[first].shape(sid)
        axes[i, 0].imshow(Image.open(frames_dir / f"{sid}_rgb.png").convert("RGB"), extent=(0, w, h, 0))
        gt = np.asarray(Image.open(frames_dir / f"{sid}_gt.png")).astype(float) / 1000
        lid = np.asarray(Image.open(frames_dir / f"{sid}_lidar.png")).astype(float) / 1000
        gt[gt <= 0], lid[lid <= 0] = np.nan, np.nan
        vmin, vmax = np.nanpercentile(gt, [2, 98])
        kw = {"extent": (0, w, h, 0), "cmap": cmap, "vmin": vmin, "vmax": vmax}
        axes[i, 1].imshow(gt, interpolation="antialiased", **kw)
        axes[i, 2].imshow(lid, interpolation="antialiased", **kw)
        label(axes[i, 1], f"{vmin:.1f}–{vmax:.1f} m")
        err = None
        for j, (name, pt) in enumerate(preds.items()):
            d = pt.get(sid)
            grid = np.full(((h + STRIDE - 1) // STRIDE, (w + STRIDE - 1) // STRIDE), np.nan)
            grid[d["row"] // STRIDE, d["col"] // STRIDE] = d["pred"]
            axes[i, 3 + j].imshow(grid, interpolation="nearest", **kw)
            label(axes[i, 3 + j], f"AbsRel {np.mean(np.abs(d['pred'] - d['gt']) / d['gt']):.3f}")
            if err is None:
                err = np.full_like(grid, np.nan)
                err[d["row"] // STRIDE, d["col"] // STRIDE] = np.abs(np.log(d["pred"] / d["gt"]))
        emap = plt.get_cmap("magma").copy()
        emap.set_bad("0.85")
        im = axes[i, -1].imshow(err, extent=(0, w, h, 0), cmap=emap, vmin=0, vmax=0.3, interpolation="nearest")
        label(axes[i, -1], f"median {np.nanmedian(err):.2f}")
        for ax in axes[i]:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("auto")
            for s in ax.spines.values():
                s.set_linewidth(0.4)
    for ax, c in zip(axes[0], cols):
        ax.set_title(c, fontsize=6, pad=2)
    pos = axes[-1, -1].get_position()
    cax = fig.add_axes([pos.x0 + 0.1 * pos.width, pos.y0 - 0.035, 0.8 * pos.width, 0.012])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[0, 0.1, 0.2, 0.3])
    cb.ax.tick_params(labelsize=5, pad=1, length=1.5)
    cb.outline.set_linewidth(0.4)
    save(fig, out)


def sheet(fr: Frames, cell: str, last: str, kinds: list[str], out: Path):
    rows = fr.cand[fr.cand.kind.isin(kinds)].drop_duplicates("sample_id")
    fig, axes = plt.subplots(len(rows), 3, figsize=(6, 1.55 * len(rows)), squeeze=False,
                             gridspec_kw={"wspace": 0.03, "hspace": 0.25})
    for (_, r), row in zip(rows.iterrows(), axes):
        sid = r.sample_id
        for ax, (c, meth) in zip(row, ((cell, "A-local"), (cell, "A-dual"), (last, "A-dual"))):
            m = fr.miss(sid, c, meth)
            show_background(ax, fr, sid, fade=0.35)
            overlay_miss(ax, fr, sid, m)
            draw_anchors(ax, fr, sid, probes=meth == "A-dual", scale=0.6)
            ax.set_aspect("auto")
            ax.set_title(f"{c} {meth} {m['cov']:.2f} w{100 * m['width']:.0f}%", fontsize=4.5, pad=1)
        row[0].set_ylabel(f"{r.kind}\n{sid.removeprefix('arkitscenes_')}", fontsize=4.5)
    save(fig, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("sheet", "teaser", "sensor", "layouts", "models"))
    ap.add_argument("--run", default="arkitscenes__unidepthv2-large__cluster")
    ap.add_argument("--frames", default=str(ROOT / "results" / "qual_frames"))
    ap.add_argument("--id")
    ap.add_argument("--ids", nargs="+")
    ap.add_argument("--cell", default="clean/clean")
    ap.add_argument("--sensor-cells", nargs=2, default=["lidar/clean", "lidar/lidar"])
    ap.add_argument("--n-probes", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--kinds", nargs="+", default=["teaser", "sensor", "requested"], help="sheet: candidate kinds")
    ap.add_argument("--out")
    args = ap.parse_args()

    run_dir = results_dir("qual_dual") / args.run
    fr = Frames(run_dir, Path(args.frames))
    plt.rcParams.update(STYLE)
    if args.what == "sheet":
        sheet(fr, args.cell, args.sensor_cells[-1], args.kinds, Path(args.out or run_dir / "sheet.pdf"))
    elif args.what == "layouts":
        layouts(args.run.rsplit("__", 1)[0], Path(args.frames), args.id, args.cell,
                Path(args.out or ROOT / "report_neurips/figures/layouts_qual.pdf"))
    elif args.what == "models":
        models(Path(args.frames), args.ids, {"unidepthv2-large": "UniDepthV2", "dav2-indoor-large": "DA-V2"},
               Path(args.out or ROOT / "report_neurips/figures/models_qual.pdf"))
    elif args.what == "teaser":
        sid = args.id or fr.cand[fr.cand.kind == "teaser"].iloc[0].sample_id
        teaser(fr, sid, args.cell, args.n_probes, args.alpha, Path(args.out or ROOT / "paper/figures/teaser_qual.pdf"))
    else:
        sids = args.ids or list(fr.cand[fr.cand.kind == "sensor"].sample_id[:3])
        sensor(fr, sids, tuple(args.sensor_cells), Path(args.out or ROOT / "paper/figures/sensor_qual.pdf"))


if __name__ == "__main__":
    main()
