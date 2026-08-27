"""Render the post figures as one plot per panel, from the existing ``results/*.json``.

No measurement happens here -- everything is read back from the result files, so the figures
can be restyled as often as needed without re-running a benchmark::

    python make_panels.py                 # main
    python make_panels.py --tag pr590     # PR 590 results

Conventions, kept identical across every panel:

* one colour per algorithm -- blue for the new kernels (Quadrature-TreeSHAP, Woodelf), orange
  for shapiq's polynomial family, violet for shap;
* ``LinearTreeSHAP`` and ``TreeSHAP-IQ`` share the orange, because order 1 of TreeSHAP-IQ *is*
  LinearTreeSHAP -- solid is order 1, dashed is order 2, so one colour reads as one algorithm;
* every series is named ``"<library> - <algorithm>"``;
* text is kept to a title, a one-line subtitle and the legend.
"""

from __future__ import annotations

import sys

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from bench_common import (
    FIGURES,
    load_results as _load_results,
)
from matplotlib.ticker import FuncFormatter, LogLocator

TAG = ""
PANELS = FIGURES / "panels"

# Below this relative efficiency error the values still carry roughly six significant digits.
# Past it an explainer is still fast, but it is no longer computing the quantity it claims to.
TOL = 1e-6

INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8983", "#e6e5e0", "#fcfcfb"
QUAD, POLY, SHAP = "#2a78d6", "#eb6834", "#4a3aa7"  # validated all-pairs, light + dark

DASH = (0, (5, 2))
DOT = (0, (1, 2.2))

# (result key, order) -> (colour, linestyle, legend name)
STYLE = {
    ("quadrature", 1): (QUAD, "-", "shapiq - Quadrature-TreeSHAP"),
    ("quadrature", 2): (QUAD, DASH, "shapiq - Quadrature-TreeSHAP, order 2"),
    ("linear", 1): (POLY, "-", "shapiq - LinearTreeSHAP"),
    ("treeshapiq", 1): (POLY, DASH, "shapiq - TreeSHAP-IQ"),
    ("treeshapiq", 2): (POLY, DASH, "shapiq - TreeSHAP-IQ, order 2"),
    ("shap", 1): (SHAP, "-", "shap - TreeSHAP"),
    ("shap", 2): (SHAP, DASH, "shap - TreeSHAP, order 2"),
    # interventional panel
    ("woodelf", 1): (QUAD, "-", "shapiq - Woodelf"),
    ("shapiq", 1): (POLY, "-", "shapiq - TreeSHAP-IQ"),
}

# Direct labels sit on the line, so a series that shares its colour *and* its algorithm with
# the line above it only has to say which order it is.
DIRECT = {("quadrature", 2): "order 2", ("shap", 2): "order 2"}


def direct_label(method: str, order: int) -> str:
    return DIRECT.get((method, order), STYLE[(method, order)][2])


plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.frameon": False,
        "lines.linewidth": 2.0,
        "lines.solid_capstyle": "round",
        "figure.dpi": 200,
    }
)


def load_results(name: str) -> dict:
    return _load_results(f"{name}_{TAG}" if TAG else name)


def accuracy_map(suite: str) -> dict:
    """``(dataset, depth, method, order) -> relative efficiency error``, if it was measured."""
    try:
        data = _load_results(f"accuracy_{suite}")
    except FileNotFoundError:
        return {}
    return {
        (r["dataset"], r["depth"], r["method"], r["order"]): r["rel_error"]
        for r in data["records"]
        if r.get("rel_error") is not None
    }


def first_unreliable(xs, acc: dict, dataset: str, method: str, order: int) -> int | None:
    """Index of the first depth whose values have lost more than ``TOL`` of their accuracy."""
    for i, depth in enumerate(xs):
        err = acc.get((dataset, depth, method, order))
        if err is not None and err > TOL:
            return i
    return None


def new_panel(size=(8.6, 4.4)):
    """A panel with a fixed layout.

    The margins are set here rather than left to ``bbox_inches="tight"``: the direct labels
    are drawn outside the data area, and a tight bounding box grows the canvas to swallow
    them, so otherwise every panel comes out a different width. ``EndLabels`` then grows the
    x-limit until the label column fits *inside* the axes, where these margins can hold it.
    """
    fig, ax = plt.subplots(figsize=size)
    fig.subplots_adjust(left=0.115, right=0.988, top=0.845, bottom=0.135)
    ax.grid(visible=True, which="major", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    return fig, ax


def titles(ax, title: str, subtitle: str = "") -> None:
    """Title above the axes, with the provenance as small lines beneath it -- no floating text.

    ``subtitle`` may hold several lines. Keeping them short matters: ``bbox_inches="tight"``
    stretches the whole canvas to fit the widest piece of text, so one long caption silently
    turns a 8-inch panel into a 13-inch one.
    """
    lines = subtitle.count("\n") + 1 if subtitle else 0
    ax.set_title(title, fontsize=11.5, loc="left", color=INK, pad=8 + 11.5 * lines)
    if subtitle:
        ax.text(
            0,
            1.02,
            subtitle,
            transform=ax.transAxes,
            fontsize=8.5,
            color=MUTED,
            va="bottom",
            linespacing=1.35,
        )


class EndLabels:
    """Collect right-hand series labels and push them apart before drawing.

    Direct labels are what let a six-series chart be read without hopping to a legend, but the
    ends of log-scale runtime curves bunch together. The layout runs in display space, so the
    separation is a fixed number of text lines no matter how the axes are scaled.
    """

    def __init__(self, ax, *, size: float = 8.5, pad_frac: float = 0.025) -> None:
        self.ax = ax
        self.size = size
        self.pad_frac = pad_frac
        self.items: list[tuple[float, float, str, str]] = []

    def add(self, x: float, y: float, text: str, color: str) -> None:
        self.items.append((x, y, text, color))

    def draw(self) -> None:
        """Place the labels, then widen the axes until every one of them fits inside it.

        The width a label needs is only knowable once it has been laid out, so the reserve is
        measured rather than guessed: each label hangs off its line end by a fixed offset in
        points, and the x-limit grows until nothing sticks out past the axes. Guessing a
        fraction instead leaves the longest name clipped on exactly the panels that need it.
        """
        if not self.items:
            return
        ax = self.ax
        fig = ax.figure
        px_per_line = self.size * 1.35 * fig.dpi / 72.0
        gaps = [px_per_line * (t.count("\n") + 1) for _x, _y, t, _c in self.items]
        ys = [ax.transData.transform((x, y))[1] for x, y, _t, _c in self.items]
        order = np.argsort(ys)
        placed, last = list(ys), -np.inf
        for idx in order:  # one upward pass is enough for a handful of series
            placed[idx] = max(ys[idx], last)
            last = placed[idx] + gaps[idx]
        # the upward pass can push the top label off the axes; drop the whole stack back down
        excess = (
            max(y + g / 2 for y, g in zip(placed, gaps, strict=False)) - ax.get_window_extent().y1
        )
        if excess > 0:
            placed = [y - excess for y in placed]
        anns = [
            ax.annotate(
                text,
                xy=(x, ax.transData.inverted().transform((0, y_px))[1]),
                xytext=(6, 0),
                textcoords="offset points",
                fontsize=self.size,
                color=color,
                va="center",
                ha="left",
                linespacing=1.35,
                clip_on=False,
                annotation_clip=False,
                zorder=6,
                # a curve that stops early is labelled where it stops, which can be on top of
                # another line -- the patch keeps the text readable without a leader
                bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1.2, "alpha": 0.85},
            )
            for (x, _y, text, color), y_px in zip(self.items, placed, strict=False)
        ]
        for _ in range(6):
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            box = ax.get_window_extent()
            overflow = max(a.get_window_extent(renderer=renderer).x1 for a in anns) - box.x1
            if overflow <= 1.0:
                break
            widen_right(ax, (overflow + 10.0) / box.width)


def widen_right(ax, frac: float) -> None:
    """Reserve room inside the axes for the direct labels."""
    x0, x1 = ax.get_xlim()
    if ax.get_xscale() == "log":
        ax.set_xlim(x0, 10 ** (np.log10(x1) + (np.log10(x1) - np.log10(x0)) * frac))
    else:
        ax.set_xlim(x0, x1 + (x1 - x0) * frac)


def fmt_time(seconds: float) -> str:
    """Human units, decade by decade: below a minute stays in s/ms/µs, above it becomes min/h."""
    if seconds <= 0:
        return "0"
    if seconds < 1e-3:
        return f"{seconds * 1e6:.3g} µs"
    if seconds < 1:
        return f"{seconds * 1e3:.3g} ms"
    if seconds < 1000:
        return f"{seconds:.3g} s"
    if seconds < 3600 * 2:
        return f"{seconds / 60:.3g} min"
    return f"{seconds / 3600:.3g} h"


def time_axis(ax) -> None:
    ax.set_yscale("log")
    ax.set_ylabel("runtime")
    ax.yaxis.set_major_locator(LogLocator(base=10.0))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: fmt_time(v)))
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda *_: ""))


def save(fig, name: str) -> None:
    PANELS.mkdir(parents=True, exist_ok=True)
    stem = f"{name}_{TAG}" if TAG else name
    for ext in ("png", "pdf"):
        fig.savefig(PANELS / f"{stem}.{ext}")
    plt.close(fig)
    print(f"wrote figures/panels/{stem}.png")


# ============================================================================================
# interventional: one panel per background size
# ============================================================================================
def count_ticks(ax, values) -> None:
    ax.set_xscale("log")
    ax.set_xticks(values)
    ax.set_xticklabels([f"{v // 1000}k" if v >= 1000 else str(v) for v in values])
    ax.xaxis.set_minor_formatter(FuncFormatter(lambda *_: ""))


def extrapolate(xs, ys, x_max):
    """Continue a curve past the last measured point as ``t(n) = setup + rate * n``.

    The interventional cost is a fixed setup plus a per-instance cost, and the measured points
    say so to within a few percent, so the affine fit is the honest extension -- but it *is* an
    extension: every extrapolated segment is drawn dotted and labelled as such.
    """
    xs, ys = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    keep = xs >= min(10, xs.max())  # drop the points where the fixed setup dominates
    rate, setup = np.polyfit(xs[keep], ys[keep], 1)
    grid = np.logspace(np.log10(xs[-1]), np.log10(x_max), 32)
    return grid, setup + rate * grid, float(rate)


def interventional() -> None:
    data = load_results("interventional")
    meta = data["meta"]
    records = data["records"]
    x_max = max(r["n_explain"] for r in records)
    ticks = [1, 10, 100, 1000, 10_000]

    for m in sorted({r["n_background"] for r in records}):
        fig, ax = new_panel()
        labels = EndLabels(ax, size=9)
        rates = {}
        for backend in ("shap", "shapiq", "woodelf"):
            pts = sorted(
                {
                    r["n_explain"]: r["median_s"]
                    for r in records
                    if r["n_background"] == m
                    and r["backend"] == backend
                    and r.get("status") == "ok"
                }.items()
            )
            if not pts:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            color, style, label = STYLE[(backend, 1)]
            ax.plot(xs, ys, color=color, linestyle=style, marker="o", markersize=3.5, zorder=3)
            end_x, end_y, note = xs[-1], ys[-1], ""
            if end_x < x_max:  # measurement was cut off -- show where the line goes, dotted
                gx, gy, rate = extrapolate(xs, ys, x_max)
                ax.plot(gx, gy, color=color, linestyle=DOT, linewidth=1.8, zorder=2)
                end_x, end_y, note = gx[-1], gy[-1], " (extrapolated)"
                rates[label] = rate
            labels.add(end_x, end_y, f"{label}\n{fmt_time(end_y)}{note}", color)
        count_ticks(ax, ticks)
        ax.set_xlim(0.8, x_max)
        time_axis(ax)
        ax.set_xlabel("explained instances $n$")
        titles(
            ax,
            f"interventional TreeSHAP  ·  background $m$ = {m}",
            f"{meta['model']} on {meta['dataset']}, Shapley values, end-to-end, single thread",
        )
        labels.draw()
        save(fig, f"panel_interventional_m{m}")
        for label, rate in rates.items():
            print(f"    {label:34s} m={m:<5d} {rate * 1e3:8.2f} ms per instance (fitted)")


# ============================================================================================
# path-dependent: one panel per dataset, plus the synthetic pair
# ============================================================================================
SERIES = (
    ("quadrature", 1),
    ("quadrature", 2),
    ("linear", 1),
    ("treeshapiq", 2),
    ("shap", 1),
    ("shap", 2),
)


def _depth_panel(records, dataset, title, subtitle, acc, xticks=None):
    """Runtime against depth. The part of a curve whose values are no longer accurate is faded.

    Timing a wrong answer is not a comparison: past ``TOL`` the polynomial explainers keep
    returning quickly but have lost every significant digit, so their curve continues as a
    hairline rather than stopping or being drawn as if it were still a result.
    """
    fig, ax = new_panel()
    labels, faded, over_budget = EndLabels(ax), False, False
    depths = sorted({r["depth"] for r in records if r["dataset"] == dataset})
    for method, order in SERIES:
        pts = sorted(
            (r["depth"], r["median_s"])
            for r in records
            if r["dataset"] == dataset
            and r["method"] == method
            and r["order"] == order
            and r.get("status") == "ok"
        )
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        color, style, _ = STYLE[(method, order)]
        marker, size = ("o", 3.5) if order == 1 else ("s", 3)
        cut = first_unreliable(xs, acc, dataset, method, order)
        good = slice(None) if cut is None else slice(0, cut)
        ax.plot(
            xs[good],
            ys[good],
            color=color,
            linestyle=style,
            marker=marker,
            markersize=size,
            zorder=3,
        )
        if cut is not None:
            faded = True
            tail = slice(max(cut - 1, 0), None)
            ax.plot(
                xs[tail],
                ys[tail],
                color=color,
                linestyle=style,
                linewidth=1.0,
                alpha=0.35,
                marker=marker,
                markersize=size - 1,
                markerfacecolor="none",
                zorder=2,
            )
        if xs[-1] < depths[-1] and not any(
            r["dataset"] == dataset
            and r["method"] == method
            and r["order"] == order
            and r["depth"] > xs[-1]
            and r.get("status") == "refused"
            for r in records
        ):
            over_budget = True
            ax.plot(
                xs[-1], ys[-1], marker="x", markersize=7, markeredgewidth=2, color=color, zorder=4
            )
        labels.add(xs[-1], ys[-1], direct_label(method, order), color)
    ax.set_xlabel("tree depth")
    ax.set_xticks(xticks or depths[:: max(1, round(len(depths) / 7))])
    time_axis(ax)
    notes = [n for n, on in ((FADE_NOTE, faded), (BUDGET_NOTE, over_budget)) if on]
    titles(ax, title, subtitle + ("\n" + "  ·  ".join(notes) if notes else ""))
    labels.draw()
    return fig, ax


# The accuracy panel is order 1 only: k-SII is efficient at every order, so an algorithm's
# order-2 curve lies on top of its order-1 curve and drawing both just doubles the ink.
SERIES_ACC = (("quadrature", 1), ("linear", 1), ("treeshapiq", 1), ("shap", 1))
ACC_LABEL = {"treeshapiq": "shapiq - TreeSHAP-IQ"}

FADE_NOTE = "faded = fewer than six correct digits left"
BUDGET_NOTE = "✕ = over the 20 s budget"


TITLES = {
    "superconductivity": "path-dependent TreeSHAP  ·  superconductivity (17k × 81)",
    "heloc": "path-dependent TreeSHAP  ·  heloc (8.4k × 23)",
    "bioresponse": "path-dependent TreeSHAP  ·  bioresponse (3k × 1776)",
}
ORDER2_NOTE = (
    "one tree per depth, one explained instance, single thread  ·  order 2 = k-SII (shap: SII)"
)


def depth_real() -> None:
    records = load_results("depth_real")["records"]
    acc = accuracy_map("real")
    for dataset in ("superconductivity", "heloc", "bioresponse"):
        fig, ax = _depth_panel(records, dataset, TITLES[dataset], ORDER2_NOTE, acc)
        refused = [
            r["depth"] for r in records if r["dataset"] == dataset and r.get("status") == "refused"
        ]
        if refused:
            ax.axvline(min(refused), color=MUTED, linestyle=DASH, linewidth=1.2, zorder=1)
            ax.annotate(
                "polynomial explainers refuse\n(features per path > 30)",
                xy=(min(refused), 0.02),
                xycoords=("data", "axes fraction"),
                xytext=(-6, 0),
                textcoords="offset points",
                fontsize=8,
                color=MUTED,
                va="bottom",
                ha="right",
            )
        save(fig, f"panel_depth_{dataset}")


def synthetic() -> None:
    records = load_results("depth_synthetic")["records"]
    fig, ax = _depth_panel(
        records,
        "synthetic",
        "path-dependent TreeSHAP  ·  synthetic deep trees",
        "rare-indicator features (shapiq issue #545): distinct features per path = tree depth",
        accuracy_map("synthetic"),
        xticks=[4, 20, 40, 60, 80, 100],
    )
    ax.set_xlabel("tree depth  ( = distinct features per decision path )")
    save(fig, "panel_synthetic_runtime")

    # accuracy -- the reason the synthetic depth sweep exists at all
    fig, ax = new_panel()
    labels = EndLabels(ax)
    for method, order in SERIES_ACC:
        pts = sorted(
            (r["depth"], max(r["rel_error"], 1e-17))
            for r in _load_results("accuracy_synthetic")["records"]
            if r["method"] == method and r["order"] == order and r.get("rel_error") is not None
        )
        if not pts:
            continue
        color, style, _ = STYLE[(method, order)]
        ax.plot(
            [p[0] for p in pts],
            [p[1] for p in pts],
            color=color,
            linestyle=style,
            marker="o" if order == 1 else "s",
            markersize=3.5 if order == 1 else 3,
            zorder=3,
        )
        labels.add(
            pts[-1][0], pts[-1][1], ACC_LABEL.get(method, direct_label(method, order)), color
        )
    ax.axhline(TOL, color=MUTED, linestyle=DASH, linewidth=1.2, zorder=1)
    ax.annotate(
        "fewer than six correct digits above this line",
        xy=(0.02, TOL),
        xycoords=("axes fraction", "data"),
        xytext=(0, 4),
        textcoords="offset points",
        fontsize=8,
        color=MUTED,
    )
    ax.set_yscale("log")
    ax.set_xlabel("tree depth  ( = distinct features per decision path )")
    ax.set_xticks([4, 20, 40, 60, 80, 100])
    ax.set_ylabel("relative efficiency error")
    titles(
        ax,
        "path-dependent TreeSHAP  ·  accuracy on the same trees",
        "|Σ values + baseline − prediction| / |prediction − mean|; exact arithmetic gives 0"
        "\norder 2 tracks order 1 exactly: k-SII is efficient at every order",
    )
    labels.draw()
    save(fig, "panel_synthetic_accuracy")


PANEL_GROUPS = {"interventional": interventional, "depth": depth_real, "synthetic": synthetic}

if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--tag" in argv:
        i = argv.index("--tag")
        TAG = argv[i + 1]
        argv = argv[:i] + argv[i + 2 :]
    for group in argv or PANEL_GROUPS:
        PANEL_GROUPS[group]()
