"""Render the three PR #588 figures from the benchmark result files.

    python make_figures.py            # all three
    python make_figures.py fig1       # just one

Reads ``results/*.json`` written by ``bench_interventional.py`` and ``bench_depth.py``.
A trailing ``--tag NAME`` renders the ``*_NAME.json`` result files into ``*_NAME.png``.
"""

from __future__ import annotations

import sys
import textwrap

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from bench_common import (
    FIGURES,
    load_results as _load_results,
)
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator

# set by ``--tag NAME``: reads ``results/<name>_NAME.json`` and writes ``figures/<fig>_NAME.png``
TAG = ""


def load_results(name: str) -> dict:
    return _load_results(f"{name}_{TAG}" if TAG else name)


# --- palette (validated categorical slots; see the dataviz palette reference) ----------------
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8983"
GRID = "#e6e5e0"
SURFACE = "#fcfcfb"

COLORS = {
    "quadrature": "#2a78d6",  # slot 1 blue -- the new default
    "linear": "#eb6834",  # slot 2 orange
    "treeshapiq": "#1baf7a",  # slot 3 aqua
    "shap": "#4a3aa7",  # slot 7 violet
    "woodelf": "#2a78d6",
    "shapiq": "#1baf7a",
}
LABELS = {
    "quadrature": "Quadrature-TreeSHAP",
    "linear": "LinearTreeSHAP",
    "treeshapiq": "TreeSHAP-IQ",
    "shap": "shap TreeSHAP",
    "shapiq": "shapiq TreeSHAP-IQ",
    "woodelf": "Woodelf",
}
ORDER_STYLE = {1: "-", 2: (0, (5, 2))}

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.linewidth": 1.0,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "lines.linewidth": 2.0,
        "lines.solid_capstyle": "round",
        "figure.dpi": 160,
    }
)


def style_axes(ax) -> None:
    ax.grid(visible=True, which="major", axis="both", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def fmt_time(seconds: float) -> str:
    if seconds <= 0:
        return "0"
    if seconds < 1e-3:
        return f"{seconds * 1e6:g} µs"
    if seconds < 1:
        return f"{seconds * 1e3:g} ms"
    return f"{seconds:g} s"


def time_axis(ax, label: str = "runtime, log scale") -> None:
    ax.set_yscale("log")
    ax.set_ylabel(label)
    ax.yaxis.set_major_locator(LogLocator(base=10.0))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: fmt_time(v)))
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda *_: ""))


class EndLabels:
    """Collect right-hand series labels and push them apart before drawing.

    Direct labels are what let a four-series chart be read without hopping to the legend, but
    the ends of log-scale runtime curves bunch together. Labels are laid out in display space
    so the separation is a constant number of pixels regardless of the scale.
    """

    def __init__(self, ax, *, pad_frac: float = 0.02, min_gap_px: float = 22.0) -> None:
        self.ax = ax
        self.items: list[tuple[float, float, str, str]] = []
        self.pad_frac = pad_frac
        self.min_gap_px = min_gap_px

    def add(self, x: float, y: float, text: str, color: str) -> None:
        self.items.append((x, y, text, color))

    def draw(self) -> None:
        if not self.items:
            return
        ax = self.ax
        ys_px = [ax.transData.transform((x, y))[1] for x, y, _t, _c in self.items]
        order = np.argsort(ys_px)
        placed = list(ys_px)
        last = -np.inf
        for idx in order:  # single upward pass is enough for <= 8 series
            placed[idx] = max(ys_px[idx], last + self.min_gap_px)
            last = placed[idx]
        x0, x1 = ax.get_xlim()
        span = (x1 - x0) if ax.get_xscale() == "linear" else (np.log10(x1) - np.log10(x0))
        for (x, _y, text, color), y_px in zip(self.items, placed, strict=False):
            x_lab = (
                x + span * self.pad_frac
                if ax.get_xscale() == "linear"
                else 10 ** (np.log10(x) + span * self.pad_frac)
            )
            y_data = ax.transData.inverted().transform((0, y_px))[1]
            ax.annotate(
                text,
                xy=(x_lab, y_data),
                fontsize=9,
                fontweight="bold",
                color=color,
                va="center",
                ha="left",
                clip_on=False,
                annotation_clip=False,
            )


def widen_right(ax, frac: float = 0.30) -> None:
    """Reserve room inside the axes for the direct labels."""
    x0, x1 = ax.get_xlim()
    if ax.get_xscale() == "log":
        ax.set_xlim(x0, 10 ** (np.log10(x1) + (np.log10(x1) - np.log10(x0)) * frac))
    else:
        ax.set_xlim(x0, x1 + (x1 - x0) * frac)


def cross(ax, x, y, color):
    ax.plot(
        [x],
        [y],
        marker="X",
        color=color,
        markersize=10,
        markeredgecolor=SURFACE,
        markeredgewidth=1.4,
        linestyle="none",
        zorder=6,
    )


def header(fig, title: str, caption: str) -> None:
    """Title + caption, wrapped to the figure width and given room above the axes.

    ``bbox_inches="tight"`` grows the canvas to fit any text that overflows, which would
    silently stretch the figure and squeeze the panels; wrapping to the actual width keeps
    the panels the size they were laid out at.
    """
    width_in, height_in = fig.get_size_inches()
    lines = textwrap.wrap(" ".join(caption.split()), width=int(width_in * 16.5))
    x = 0.11 / width_in
    fig.suptitle(title, fontsize=15, x=x, ha="left", y=1.0, color=INK)
    fig.text(
        x,
        1.0 - 0.40 / height_in,
        "\n".join(lines),
        fontsize=9,
        color=INK_2,
        ha="left",
        va="top",
        linespacing=1.45,
    )
    fig.subplots_adjust(top=1.0 - (0.78 + 0.20 * len(lines)) / height_in)


def save_fig(fig, name: str, *, tagged: bool = True) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    if TAG and tagged:
        name = f"{name}_{TAG}"
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote figures/{name}.png")


# ============================================================================================
# Figure 1 -- interventional: explained instances vs. runtime
# ============================================================================================
def fig1() -> None:
    data = load_results("interventional")
    meta, records = data["meta"], data["records"]
    backgrounds = sorted({r["n_background"] for r in records})
    n_max = max(r["n_explain"] for r in records)

    fig, axes = plt.subplots(
        1, len(backgrounds), figsize=(4.9 * len(backgrounds), 4.6), sharey=True
    )
    axes = np.atleast_1d(axes)
    pending: list[EndLabels] = []

    for ax, m in zip(axes, backgrounds, strict=False):
        style_axes(ax)
        labels = EndLabels(ax)
        for backend in ("shapiq", "woodelf", "shap"):
            pts = sorted(
                (
                    (r["n_explain"], r["median_s"])
                    for r in records
                    if r["n_background"] == m
                    and r["backend"] == backend
                    and r.get("status") == "ok"
                ),
                key=lambda p: p[0],
            )
            if not pts:
                continue
            xs, ys = zip(*pts, strict=False)
            color = COLORS[backend]
            ax.plot(xs, ys, color=color, marker="o", markersize=4.5, zorder=3)
            if xs[-1] != n_max:  # curve cut off by the measurement budget
                cross(ax, xs[-1], ys[-1], color)
            labels.add(xs[-1], ys[-1], LABELS[backend], color)

        ax.set_xscale("log")
        ax.set_xlim(0.8, n_max * 1.25)
        widen_right(ax, 0.42)
        ax.set_xticks([1, 10, 100, 1000])
        ax.set_xticklabels(["1", "10", "100", "1000"])
        ax.set_xlabel("explained instances $n$")
        ax.set_title(f"background dataset $m$ = {m}", fontsize=11, pad=10, loc="left")

        # the shipped auto-routing cut-off: n * m >= 100 sends the batch to Woodelf
        cutoff = 100 / m
        if 1 <= cutoff <= n_max:
            ax.axvline(cutoff, color=MUTED, linestyle=(0, (2, 3)), linewidth=1.2, zorder=1)
            ax.annotate(
                "shapiq switches\nto Woodelf here",
                xy=(cutoff, 0.02),
                xycoords=("data", "axes fraction"),
                xytext=(5, 0),
                textcoords="offset points",
                fontsize=8,
                color=MUTED,
                va="bottom",
            )
        elif cutoff < 1:  # n * m >= 100 holds for every n on this background
            ax.annotate(
                "shapiq routes to Woodelf\nfor every $n$ here",
                xy=(0.02, 0.02),
                xycoords="axes fraction",
                fontsize=8,
                color=MUTED,
                va="bottom",
            )
        pending.append(labels)

    time_axis(axes[0])  # log scale first: EndLabels lays out in display coordinates
    for ax in axes[1:]:
        ax.set_ylabel("")
    for labels in pending:
        labels.draw()

    checks = meta["agreement"]
    if "max_abs_dev_shapiq_woodelf" in checks:  # older result files: a single check
        checks = {"m": checks}
    dev = max(
        max(c["max_abs_dev_shapiq_woodelf"], c["max_abs_dev_shapiq_shap"]) for c in checks.values()
    )
    header(
        fig,
        "Interventional TreeSHAP: two regimes, one explainer",
        f"Shapley values for a {meta['model']} on {meta['dataset']} "
        f"({meta['n_train']:,}×{meta['n_features']}). End-to-end wall clock — explainer "
        f"construction plus the explanation of $n$ instances — median of {meta['repeats']} runs, "
        f"single thread. ✕ = curve cut off at the {meta['stop_after_s']:.0f} s measurement "
        f"budget. The three backends return the same values (max deviation {dev:.0e}).",
    )
    fig.subplots_adjust(left=0.065, right=0.995, bottom=0.115, wspace=0.10)
    save_fig(fig, "fig1_interventional")


# ============================================================================================
# Figures 2 & 3 -- path-dependent: tree depth vs. runtime
# ============================================================================================
def _series(records, dataset, method, order):
    pts = [
        r
        for r in records
        if r["dataset"] == dataset
        and r["method"] == method
        and r["order"] == order
        and r.get("status") == "ok"
    ]
    pts.sort(key=lambda r: r["depth"])
    return pts


def _refusals(records, dataset, method, order):
    return sorted(
        (
            r
            for r in records
            if r["dataset"] == dataset
            and r["method"] == method
            and r["order"] == order
            and r.get("status") == "refused"
        ),
        key=lambda r: r["depth"],
    )


def method_legend(fig, methods, *, ncol, extra=(), y=0.012):
    handles = [Line2D([], [], color=COLORS[m], linewidth=2.4, label=LABELS[m]) for m in methods]
    handles += [
        Line2D([], [], color=INK_2, linewidth=2.0, linestyle=ORDER_STYLE[1], label="order 1 (SV)"),
        Line2D(
            [], [], color=INK_2, linewidth=2.0, linestyle=ORDER_STYLE[2], label="order 2 (k-SII)"
        ),
        *extra,
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, y),
        ncol=ncol,
        fontsize=9,
        labelcolor=INK_2,
        columnspacing=1.6,
        handlelength=2.2,
    )


def fig2() -> None:
    data = load_results("depth_real")
    meta, records = data["meta"], data["records"]
    datasets = [d for d in meta["datasets"] if any(r["dataset"] == d for r in records)]
    titles = {
        "superconductivity": "superconductivity · 17k×81 · regression",
        "heloc": "heloc · 8.4k×23 · classification",
        "bioresponse": "bioresponse · 3k×1776 · sparse binary",
    }
    methods = ("quadrature", "linear", "treeshapiq", "shap")

    fig, axes = plt.subplots(1, len(datasets), figsize=(5.3 * len(datasets), 5.0), sharey=True)
    axes = np.atleast_1d(axes)

    pending: list[EndLabels] = []
    for ax, ds in zip(axes, datasets, strict=False):
        style_axes(ax)
        labels = EndLabels(ax)
        depths = sorted({r["depth"] for r in records if r["dataset"] == ds})
        for method in methods:
            for order in (1, 2):
                pts = _series(records, ds, method, order)
                if not pts:
                    continue
                xs = [p["depth"] for p in pts]
                ys = [p["median_s"] for p in pts]
                ax.plot(
                    xs,
                    ys,
                    color=COLORS[method],
                    linestyle=ORDER_STYLE[order],
                    marker="o" if order == 1 else "s",
                    markersize=4 if order == 1 else 3.5,
                    zorder=3,
                )
                refused_deeper = [
                    r for r in _refusals(records, ds, method, order) if r["depth"] > xs[-1]
                ]
                if xs[-1] != max(depths) and not refused_deeper:
                    cross(ax, xs[-1], ys[-1], COLORS[method])  # cut off by the time budget
                if order == 1 or not _series(records, ds, method, 1):
                    labels.add(xs[-1], ys[-1], LABELS[method], COLORS[method])

        # where the shipped numerical guard refuses the polynomial explainers outright
        refused = [
            r["depth"] for r in records if r["dataset"] == ds and r.get("status") == "refused"
        ]
        if refused:
            ax.axvline(min(refused), color=MUTED, linestyle=(0, (2, 3)), linewidth=1.2, zorder=1)
            ax.annotate(
                "LinearTreeSHAP and TreeSHAP-IQ\nrefused from here (30 features/path)",
                xy=(min(refused), 0.02),
                xycoords=("data", "axes fraction"),
                xytext=(-6, 0),
                textcoords="offset points",
                fontsize=8,
                color=MUTED,
                va="bottom",
                ha="right",
            )
        ax.set_xlim(min(depths) - 1, max(depths) + 1)
        widen_right(ax, 0.34)
        ax.set_xticks(depths[:: max(1, round(len(depths) / 7))])
        ax.set_xlabel("tree depth")
        ax.set_title(titles.get(ds, ds), fontsize=10.5, pad=10, loc="left")
        pending.append(labels)

    time_axis(axes[0])  # log scale first: EndLabels lays out in display coordinates
    for ax in axes[1:]:
        ax.set_ylabel("")
    for labels in pending:
        labels.draw()

    header(
        fig,
        "Path-dependent TreeSHAP: single-explanation runtime by tree depth",
        "One sklearn decision tree per depth on TabArena datasets, one explained instance. "
        f"Median of ≤{meta['repeats']} runs, warm-up and explainer construction excluded, "
        "single thread. LinearTreeSHAP computes Shapley values only; shap's order 2 is its "
        "pairwise interaction matrix. ✕ = curve cut off at the 20 s measurement budget.",
    )
    method_legend(fig, methods, ncol=6)
    fig.subplots_adjust(left=0.062, right=0.995, bottom=0.175, wspace=0.09)
    save_fig(fig, "fig2_depth_real")


def _degraded_from(pts, rel_threshold=1e-4):
    """Number of leading points whose order-1 efficiency error is still below the threshold."""
    for i, p in enumerate(pts):
        err = p.get("efficiency_error")
        scale = max(p.get("prediction_scale") or 1.0, 1e-12)
        if err is not None and err / scale > rel_threshold:
            return i
    return len(pts)


def fig3() -> None:
    data = load_results("depth_synthetic")
    meta, records = data["meta"], data["records"]
    methods = ("quadrature", "linear", "treeshapiq", "shap")
    ds = "synthetic"
    x_label = "tree depth   ( = distinct features per decision path )"

    fig, (ax, ax_err) = plt.subplots(1, 2, figsize=(14.0, 5.4))
    style_axes(ax)
    style_axes(ax_err)

    cut = {m: _degraded_from(_series(records, ds, m, 1)) for m in methods}
    labels = EndLabels(ax)
    crash_x = {
        m: (_refusals(records, ds, m, 1)[0]["depth"] if _refusals(records, ds, m, 1) else None)
        for m in methods
    }

    for method in methods:
        color = COLORS[method]
        for order in (1, 2):
            pts = _series(records, ds, method, order)
            if not pts:
                continue
            xs = [p["depth"] for p in pts]
            ys = [p["median_s"] for p in pts]
            k = min(cut[method], len(xs))
            ax.plot(
                xs[:k],
                ys[:k],
                color=color,
                linestyle=ORDER_STYLE[order],
                marker="o" if order == 1 else "s",
                markersize=4 if order == 1 else 3.5,
                zorder=3,
            )
            if k < len(xs):  # still runs, but the values are no longer usable
                ax.plot(
                    xs[max(k - 1, 0) :],
                    ys[max(k - 1, 0) :],
                    color=color,
                    linestyle=ORDER_STYLE[order],
                    marker="o" if order == 1 else "s",
                    markersize=4 if order == 1 else 3.5,
                    alpha=0.25,
                    zorder=2,
                )
                if order == 1:
                    ax.plot(
                        [xs[k - 1]],
                        [ys[k - 1]],
                        marker="o",
                        markerfacecolor=SURFACE,
                        markeredgecolor=color,
                        markeredgewidth=2.2,
                        markersize=9,
                        linestyle="none",
                        zorder=6,
                    )
            if order == 1:
                labels.add(max(xs[-1], crash_x[method] or 0), ys[-1], LABELS[method], color)

        crash = _refusals(records, ds, method, 1)
        run = _series(records, ds, method, 1)
        if crash and run:
            x_c, y_c = crash[0]["depth"], run[-1]["median_s"]
            ax.plot(
                [run[-1]["depth"], x_c],
                [y_c, y_c],
                color=color,
                linestyle=(0, (1, 2)),
                linewidth=1.4,
                alpha=0.35,
                zorder=2,
            )
            cross(ax, x_c, y_c, color)

    guard = [
        r["depth"]
        for r in records
        if r["dataset"] == ds and r["max_features_per_path"] >= 30 and r["order"] == 1
    ]
    if guard:
        ax.axvline(min(guard), color=MUTED, linestyle=(0, (2, 3)), linewidth=1.2, zorder=1)
        ax.annotate(
            "shapiq now refuses the\npolynomial explainers here",
            xy=(min(guard), 0.02),
            xycoords=("data", "axes fraction"),
            xytext=(6, 0),
            textcoords="offset points",
            fontsize=8,
            color=MUTED,
            va="bottom",
        )

    depths = sorted({r["depth"] for r in records if r["dataset"] == ds})
    ticks = list(range(0, max(depths) + 1, 20))
    ax.set_xlim(0, max(depths) + 2)
    widen_right(ax, 0.30)
    ax.set_xticks(ticks)
    ax.set_xlabel(x_label)
    ax.set_title("runtime", fontsize=11, pad=10, loc="left")
    time_axis(ax)

    # --- right panel: what the runtime alone cannot show ------------------------------------
    err_labels = EndLabels(ax_err)
    for method in methods:
        pts = [p for p in _series(records, ds, method, 1) if p.get("efficiency_error") is not None]
        if not pts:
            continue
        xs = [p["depth"] for p in pts]
        ys = [
            max(p["efficiency_error"] / max(p.get("prediction_scale") or 1.0, 1e-12), 1e-17)
            for p in pts
        ]
        ax_err.plot(xs, ys, color=COLORS[method], marker="o", markersize=4, zorder=3)
        err_labels.add(xs[-1], ys[-1], LABELS[method], COLORS[method])
    ax_err.axhline(1e-4, color=MUTED, linestyle=(0, (2, 3)), linewidth=1.2, zorder=1)
    ax_err.annotate(
        "above this line the values are no longer usable",
        xy=(0.50, 1e-4),
        xycoords=("axes fraction", "data"),
        xytext=(0, -6),
        textcoords="offset points",
        fontsize=8,
        color=MUTED,
        va="top",
    )
    ax_err.set_yscale("log")
    ax_err.set_xlim(0, max(depths) + 2)
    widen_right(ax_err, 0.30)
    ax_err.set_xticks(ticks)
    ax_err.set_xlabel(x_label)
    ax_err.set_ylabel("relative efficiency error, log scale")
    ax_err.set_title(
        "accuracy of the same computation (Shapley values)", fontsize=11, pad=10, loc="left"
    )
    err_labels.draw()
    labels.draw()

    header(
        fig,
        "Deep trees: only the quadrature kernel is still standing",
        f"Synthetic rare-indicator features (shapiq issue #545; {meta['n_samples']:,}×"
        f"{meta['n_features']}, rate {meta['indicator_rate']}), where every root-to-leaf path "
        "uses as many distinct features as the tree is deep. One instance, median of "
        f"≤{meta['repeats']} runs, single thread. Faded = still runs, but the values are wrong; "
        "✕ = LinAlgError; order-2 curves that stop early hit the 20 s measurement budget. "
        "LinearTreeSHAP computes Shapley values only.",
    )
    method_legend(
        fig,
        methods,
        ncol=7,
        extra=(
            Line2D(
                [],
                [],
                color=MUTED,
                marker="o",
                linestyle="none",
                markerfacecolor=SURFACE,
                markeredgewidth=2.2,
                markersize=8,
                label="last trustworthy depth",
            ),
        ),
    )
    fig.subplots_adjust(left=0.058, right=0.995, bottom=0.185, wspace=0.30)
    save_fig(fig, "fig3_depth_synthetic")


# ============================================================================================
# Figure 4 -- what PR #590's two improvements are worth
# ============================================================================================
def fig4() -> None:
    """Before/after for the two changes in PR #590, against main at 2dc08d1."""
    before = _load_results("interventional")
    after = _load_results("interventional_pr590")
    ksii = _load_results("ksii_isolated")["cases"]
    m_panel = 100

    fig, (ax, ax_b) = plt.subplots(1, 2, figsize=(13.6, 5.0))
    style_axes(ax)
    style_axes(ax_b)

    def curve(data, backend):
        pts = sorted(
            (r["n_explain"], r["median_s"])
            for r in data["records"]
            if r["backend"] == backend and r["n_background"] == m_panel and r.get("median_s")
        )
        return [p[0] for p in pts], [p[1] for p in pts]

    labels = EndLabels(ax)
    xs, ys = curve(before, "shapiq")
    ax.plot(
        xs,
        ys,
        color=COLORS["shapiq"],
        linestyle=(0, (5, 2)),
        marker="o",
        markersize=4,
        alpha=0.45,
        zorder=2,
    )
    if xs[-1] != 1000:
        cross(ax, xs[-1], ys[-1], COLORS["shapiq"])
    labels.add(xs[-1], ys[-1], "shapiq on main", COLORS["shapiq"])
    for backend, label in (("shapiq", "shapiq + PR 590"), ("woodelf", "Woodelf"), ("shap", "shap")):
        xs, ys = curve(after, backend)
        ax.plot(xs, ys, color=COLORS[backend], marker="o", markersize=4.5, zorder=3)
        labels.add(xs[-1], ys[-1], label, COLORS[backend])
    ax.set_xscale("log")
    ax.set_xlim(0.8, 1250)
    widen_right(ax, 0.42)
    ax.set_xticks([1, 10, 100, 1000])
    ax.set_xticklabels(["1", "10", "100", "1000"])
    ax.set_xlabel("explained instances $n$")
    ax.set_title(
        f"1 · interventional TreeSHAP-IQ, background $m$ = {m_panel}",
        fontsize=11,
        pad=10,
        loc="left",
    )
    time_axis(ax)
    ax.axvline(1, color=MUTED, linestyle=(0, (2, 3)), linewidth=1.2, zorder=1)
    ax.annotate(
        "auto routes to Woodelf from $n$ = 1 —\nnow up to 198× slower than staying on shapiq",
        xy=(0.97, 0.04),
        xycoords="axes fraction",
        fontsize=8,
        color=MUTED,
        va="bottom",
        ha="right",
    )
    labels.draw()

    # --- right panel: the k-SII aggregation, timed on its own -------------------------------
    names = list(ksii)
    idx = np.arange(len(names))
    w = 0.27
    series = (
        ("main_ms", "main", COLORS["treeshapiq"]),
        ("pr590_ms", "PR 590", COLORS["quadrature"]),
        ("pr590_compact_ms", "PR 590 + compact ids", COLORS["linear"]),
    )
    for k, (key, label, color) in enumerate(series):
        vals = [ksii[n][key] for n in names]
        ax_b.bar(idx + (k - 1) * w, vals, w * 0.9, color=color, label=label, zorder=3)
        for i, v in enumerate(vals):
            ax_b.annotate(
                f"{v:.1f}" if v >= 1 else f"{v:.2f}",
                xy=(i + (k - 1) * w, v),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=7.5,
                color=INK_2,
            )
    ax_b.set_yscale("log")
    ax_b.set_xticks(idx)
    ax_b.set_xticklabels(
        [n.replace(" d", "\ndepth ") + f"\n{ksii[n]['n_base']} interactions" for n in names],
        fontsize=8.5,
    )
    ax_b.set_ylabel("aggregation time, log scale")
    ax_b.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: fmt_time(v / 1e3)))
    ax_b.set_title("2 · SII → k-SII aggregation, timed on its own", fontsize=11, pad=10, loc="left")
    top = max(c[k] for c in ksii.values() for k in ("main_ms", "pr590_ms", "pr590_compact_ms"))
    ax_b.set_ylim(top=top * 6)  # headroom so the legend clears the tallest bar
    ax_b.legend(fontsize=9, labelcolor=INK_2, loc="upper left", ncol=3, columnspacing=1.2)

    header(
        fig,
        "PR #590: both improvements, measured against main",
        "Left: end-to-end cost of explaining $n$ instances against a 100-row background "
        "(heloc, RandomForest 20 × depth 8), median of 3 runs. Right: "
        "aggregate_base_attributions timed alone on frozen SII inputs, median of 9 runs. "
        "Values are unchanged in both cases — the interventional path matches a brute-force "
        "oracle to 1.7e-16 and the aggregation is bit-identical. Single thread; shap and "
        "Woodelf are untouched by the PR and act as the control (1.01× / 1.04×).",
    )
    fig.subplots_adjust(left=0.062, right=0.995, bottom=0.145, wspace=0.26)
    save_fig(fig, "fig4_pr590", tagged=False)  # already names both versions


FIGS = {"fig1": fig1, "fig2": fig2, "fig3": fig3, "fig4": fig4}

if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--tag" in argv:
        i = argv.index("--tag")
        TAG = argv[i + 1]
        argv = argv[:i] + argv[i + 2 :]
    for key in argv or list(FIGS):
        FIGS[key]()
