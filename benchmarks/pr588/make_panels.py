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
* every curve is named at its own line end; the library and release come from the colour key,
  except on the interventional panel, which carries no key and names them on the sticker;
* text is kept to a title, a short subtitle and the key.
"""

from __future__ import annotations

import itertools
import sys

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from bench_common import (
    FIGURES,
    load_results as _load_results,
)
from matplotlib.legend_handler import HandlerPatch, HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch
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

# The release a method ships in, not the build it was timed on -- every curve here is measured
# against the same working tree. The tag says "this is what you get when you upgrade".
V_NEW, V_OLD = "v1.7.0", "v1.6.0"
SEP = " ("  # where an over-long sticker wraps: the algorithm, then its order

# (result key, order) -> (colour, linestyle, name). The stickers carry the algorithm alone;
# which library and release it belongs to is the colour key's job, and repeating it on every
# sticker only widened the reserve the labels take out of the plotting area.
STYLE = {
    ("quadrature", 1): (QUAD, "-", "Quadrature-TreeSHAP"),
    ("quadrature", 2): (QUAD, DASH, "Quadrature-TreeSHAP (order 2)"),
    ("linear", 1): (POLY, "-", "LinearTreeSHAP"),
    ("treeshapiq", 1): (POLY, DASH, "TreeSHAP-IQ"),
    ("treeshapiq", 2): (POLY, DASH, "TreeSHAP-IQ (order 2)"),
    ("shap", 1): (SHAP, "-", "TreeSHAP"),
    ("shap", 2): (SHAP, DASH, "TreeSHAP (order 2)"),
    # interventional panel
    ("woodelf", 1): (QUAD, "-", "Woodelf"),
    ("shapiq", 1): (POLY, "-", "Interventional TreeSHAP-IQ"),
}

# What v1.7.0 brings. These curves are drawn heavier and carry a thin surface-coloured halo, so
# they stay legible where they cross an older one -- the halo is a path effect on the real line
# rather than a second wider line underneath, which keeps dashes, markers and z-order honest.
NEW = {"quadrature", "woodelf", "shapiq"}
LW_NEW, LW_OLD, HALO = 2.4, 1.8, 2.2


def series(method: str, order: int) -> tuple[dict, dict, str, str]:
    """Line kwargs, marker kwargs, colour and name, with v1.7.0 given the heavier treatment.

    Line and markers are two artists on purpose. A path effect on a single artist strokes the
    markers as well, and matplotlib draws the whole marker pass after the whole line pass -- so
    the halo of each marker lands *on top of* the line it is meant to protect and beads it into
    a dashed-looking curve. Stroking only the line and laying plain markers over it keeps the
    curve continuous.
    """
    color, style, label = STYLE[(method, order)]
    is_new = method in NEW
    width = LW_NEW if is_new else LW_OLD
    zorder = 4 if is_new else 3
    line = {"color": color, "linestyle": style, "linewidth": width, "zorder": zorder}
    if is_new:
        line["path_effects"] = [
            pe.Stroke(linewidth=width + HALO, foreground=SURFACE),
            pe.Normal(),
        ]
    marks = {
        "color": color,
        "linestyle": "none",
        "marker": "o" if order == 1 else "s",
        "markersize": (4.2 if is_new else 3.4) if order == 1 else (3.8 if is_new else 3.0),
        "zorder": zorder + 0.4,
    }
    return line, marks, color, label


def plot_series(ax, xs, ys, line: dict, marks: dict) -> None:
    ax.plot(xs, ys, **line)
    ax.plot(xs, ys, **marks)


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


def new_panel(*, legend: bool = True):
    """A panel with a fixed layout.

    The margins are set here rather than left to ``bbox_inches="tight"``: the direct labels
    are drawn outside the data area, and a tight bounding box grows the canvas to swallow
    them, so otherwise every panel comes out a different width. ``EndLabels`` then grows the
    x-limit until the label column fits *inside* the axes, where these margins can hold it.
    A panel without a key reclaims the band the key would have used, instead of carrying an
    empty strip under its x label.
    """
    bottom = 0.295 if legend else 0.135
    fig, ax = plt.subplots(figsize=(8.6, 5.0 if legend else 4.4))
    fig.subplots_adjust(left=0.115, right=0.988, top=0.845, bottom=bottom)
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

    A label is written on one line whenever one line fits. What it has to fit inside is a
    budget: the reserve is taken out of the plotting area, so an unbounded longest label buys
    a wide empty margin for every panel. Only the labels that break the budget wrap, at the
    separator between the library and the algorithm.
    """

    def __init__(
        self,
        ax,
        *,
        size: float = 8.5,
        pad_frac: float = 0.025,
        budget: float = 0.30,
        always_wrap: bool = False,
    ):
        self.ax = ax
        self.size = size
        self.pad_frac = pad_frac
        self.budget = budget  # fraction of the axes width one label may claim
        self.always_wrap = always_wrap  # break at SEP even when the one-line form would fit
        self.items: list[tuple[float, float, str, str]] = []

    def add(self, x: float, y: float, text: str, color: str) -> None:
        self.items.append((x, y, text, color))

    def _wrap(self, ann, limit: float, renderer) -> None:
        """Break one over-long label at its separator, longest-piece-first."""
        text = ann.get_text()
        if "\n" in text or SEP not in text:
            return
        if not self.always_wrap and ann.get_window_extent(renderer=renderer).width <= limit:
            return
        ann.set_text(text.replace(SEP, "\n(", 1))

    def _relax(self, ys: list[float], heights: list[float]) -> list[float]:
        """Separate overlapping stickers by moving both neighbours, not just the upper one.

        A single upward pass anchors the bottom sticker on its line and shifts everything else
        up, which drags the whole stack away from the curves it labels. Pushing each colliding
        pair apart by half the overlap instead spreads the displacement, so the bottom sticker
        can drop below its line end when that is what makes room -- and every sticker ends up
        as close to its own curve as the crowding allows.
        """
        box = self.ax.get_window_extent()
        order = list(np.argsort(ys))
        placed = list(ys)
        for _ in range(60):
            worst = 0.0
            for lower, upper in itertools.pairwise(order):
                need = (heights[lower] + heights[upper]) / 2
                overlap = need - (placed[upper] - placed[lower])
                if overlap > 0.5:
                    placed[lower] -= overlap / 2
                    placed[upper] += overlap / 2
                    worst = max(worst, overlap)
            for i, height in enumerate(heights):  # never leave the axes
                placed[i] = min(max(placed[i], box.y0 + height / 2), box.y1 - height / 2)
            if worst <= 0.5:
                break
        return placed

    def draw(self) -> None:
        """Lay the labels out, then widen the axes just enough to hold the widest one.

        The width a label needs is only knowable once it has been laid out, so both the wrap
        decision and the reserve are measured rather than guessed.
        """
        if not self.items:
            return
        ax = self.ax
        fig = ax.figure
        anns = [
            ax.annotate(
                text,
                xy=(x, y),
                xytext=(5, 0),
                textcoords="offset points",
                fontsize=self.size,
                color=color,
                va="center",
                ha="left",
                linespacing=1.3,
                clip_on=False,
                annotation_clip=False,
                zorder=6,
                # a curve that stops early is labelled where it stops, which can be on top of
                # another line -- the patch keeps the text readable without a leader
                bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1.2, "alpha": 0.85},
            )
            for x, y, text, color in self.items
        ]
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        limit = ax.get_window_extent().width * self.budget
        for ann in anns:
            self._wrap(ann, limit, renderer)

        px_per_line = self.size * 1.3 * fig.dpi / 72.0
        heights = [px_per_line * (a.get_text().count("\n") + 1) for a in anns]
        ys = [ax.transData.transform((x, y))[1] for x, y, _t, _c in self.items]
        placed = self._relax(ys, heights)
        inverse = ax.transData.inverted()
        for ann, (x, _y, _t, _c), y_px in zip(anns, self.items, placed, strict=False):
            ann.xy = (x, inverse.transform((0, y_px))[1])

        for _ in range(6):
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            box = ax.get_window_extent()
            overflow = max(a.get_window_extent(renderer=renderer).x1 for a in anns) - box.x1
            if overflow <= 1.0:
                break
            widen_right(ax, (overflow + 14.0) / box.width)


def widen_right(ax, frac: float) -> None:
    """Reserve room inside the axes for the direct labels."""
    x0, x1 = ax.get_xlim()
    if ax.get_xscale() == "log":
        ax.set_xlim(x0, 10 ** (np.log10(x1) + (np.log10(x1) - np.log10(x0)) * frac))
    else:
        ax.set_xlim(x0, x1 + (x1 - x0) * frac)


class RoundedSwatch(HandlerPatch):
    """Draw a legend patch as a rounded box rather than a sharp rectangle."""

    def create_artists(
        self,
        legend,  # noqa: ARG002
        orig_handle,
        xdescent,
        ydescent,
        width,
        height,
        fontsize,  # noqa: ARG002
        trans,
    ):
        # The legend's handlelength is set for the dashed order keys, which need the room to
        # show a dash pattern. A swatch stretched to that length reads as a thick line, so the
        # box takes its own size here and sits centred in the handle slot.
        box_h = height * 1.7
        box_w = box_h * 1.9
        box = FancyBboxPatch(
            (-xdescent + (width - box_w) / 2, -ydescent + (height - box_h) / 2),
            box_w,
            box_h,
            boxstyle=f"round,pad=0,rounding_size={box_h * 0.38:.2f}",
            facecolor=orig_handle.get_facecolor(),
            edgecolor="none",
            mutation_aspect=1.0,
        )
        box.set_transform(trans)
        return [box]


def key_patch(color: str, label: str) -> Patch:
    """A colour swatch for the legend -- rendered as a rounded box by ``RoundedSwatch``."""
    return Patch(facecolor=color, edgecolor="none", label=label)


def key_line(**kwargs) -> Line2D:
    """A neutral proxy line for the encoding legend -- grey, so it reads as a key, not a series."""
    return Line2D([], [], color=INK_2, **kwargs)


def row_major(entries: list, ncol: int) -> list:
    """Reorder handles so matplotlib's column-major fill lays them out row by row.

    ``Axes.legend`` has no row-major option: with three columns it puts the first two handles
    in column one. Transposing here is what lets the key read as "the three releases, then the
    three encodings" instead of interleaving them.
    """
    rows = -(-len(entries) // ncol)
    return [
        entries[row * ncol + col]
        for col in range(ncol)
        for row in range(rows)
        if row * ncol + col < len(entries)
    ]


def encoding_legend(ax, entries, ncol: int = 3, labels: list[str] | None = None) -> None:
    """A key for the visual variables, not a list of series -- the series name their own lines.

    What the reader cannot infer from a direct label is what the *encoding* means: which order a
    dash stands for, that the heavy outlined curves are the new release, and why part of a line
    is faded. That is what goes here, and nothing else. It sits below the axes -- placed inside
    them it lands on exactly the curves it is there to explain.
    """
    if not entries:
        return
    ax.legend(
        handles=entries,
        **({"labels": labels} if labels else {}),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.235),
        ncol=ncol,
        fontsize=8.5,
        labelcolor=INK_2,
        handlelength=2.6,
        handletextpad=0.7,
        columnspacing=2.0,
        borderaxespad=0.0,
        frameon=False,
        handler_map={Patch: RoundedSwatch(), tuple: HandlerTuple(ndivide=None, pad=0.4)},
    )


# Colour carries the release, so it is the first thing the key explains -- as a rounded swatch,
# which reads as "this colour" rather than as one more line style to decode.
COLOUR_KEYS = (
    key_patch(QUAD, f"shapiq {V_NEW}"),
    key_patch(POLY, f"shapiq {V_OLD}"),
    key_patch(SHAP, "shap"),
)
# The interventional panel carries no key, so its stickers name the library themselves.
LIBRARY = {"woodelf": f"shapiq {V_NEW}", "shapiq": f"shapiq {V_NEW}", "shap": "shap"}
ORDER_KEYS = (
    key_line(
        linestyle="-", linewidth=1.6, marker="o", markersize=4, label="order 1 (Shapley values)"
    ),
    key_line(
        linestyle=(0, (4, 1.8)),
        linewidth=1.6,
        marker="s",
        markersize=3.5,
        label="order 2 (k-SII)",
    ),
)
FADE_KEY = key_line(
    linewidth=1.0,
    alpha=0.5,
    marker="o",
    markersize=3,
    markerfacecolor="none",
    label="numerically instable",
)


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
        fig, ax = new_panel(legend=False)
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
            line, marks, color, label = series(backend, 1)
            plot_series(ax, xs, ys, line, marks)
            end_x, end_y = xs[-1], ys[-1]
            if end_x < x_max:  # measurement was cut off -- show where the line goes, dotted
                gx, gy, rate = extrapolate(xs, ys, x_max)
                ax.plot(gx, gy, color=color, linestyle=DOT, linewidth=1.8, zorder=2)
                # with no key on this panel, the dotted tail says what it is where it runs
                mid = len(gx) // 2
                ax.annotate(
                    "extrapolated",
                    xy=(gx[mid], gy[mid]),
                    xytext=(0, -13),
                    textcoords="offset points",
                    fontsize=8,
                    color=color,
                    alpha=0.8,
                    ha="center",
                    va="top",
                )
                end_x, end_y = gx[-1], gy[-1]
                rates[label] = rate
            labels.add(end_x, end_y, f"{LIBRARY[backend]}\n{label}", color)
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
    labels, faded = EndLabels(ax, always_wrap=True), False
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
        line, marks, color, label = series(method, order)
        cut = first_unreliable(xs, acc, dataset, method, order)
        good = slice(None) if cut is None else slice(0, cut)
        plot_series(ax, xs[good], ys[good], line, marks)
        if cut is not None:
            faded = True
            tail = slice(max(cut - 1, 0), None)
            ax.plot(
                xs[tail],
                ys[tail],
                color=color,
                linestyle=line["linestyle"],
                linewidth=1.0,
                alpha=0.35,
                marker=marks["marker"],
                markersize=marks["markersize"] - 1,
                markerfacecolor="none",
                zorder=2,
            )
        labels.add(xs[-1], ys[-1], label, color)
    ax.set_xlabel("tree depth")
    ax.set_xticks(xticks or depths[:: max(1, round(len(depths) / 7))])
    time_axis(ax)
    titles(ax, title, subtitle)
    keys = [*COLOUR_KEYS, *ORDER_KEYS, *([FADE_KEY] if faded else [])]
    encoding_legend(ax, row_major(keys, 3), ncol=3)
    labels.draw()
    return fig, ax


# The accuracy panel is order 1 only: k-SII is efficient at every order, so an algorithm's
# order-2 curve lies on top of its order-1 curve and drawing both just doubles the ink.
SERIES_ACC = (("quadrature", 1), ("linear", 1), ("treeshapiq", 1), ("shap", 1))


TITLES = {
    "superconductivity": "path-dependent TreeSHAP  ·  superconductivity (17k × 81)",
    "heloc": "path-dependent TreeSHAP  ·  heloc (8.4k × 23)",
    "bioresponse": "path-dependent TreeSHAP  ·  bioresponse (3k × 1776)",
}
ORDER2_NOTE = "one tree per depth, one explained instance, single thread  ·  order 2 = k-SII"


# Where to spell out the speed-up on the panel itself: (slower series, faster series, depth).
RATIO_ARROW = {"superconductivity": (("shap", 2), ("quadrature", 2), 28)}


def ratio_arrow(ax, records, dataset: str, slower, faster, at: int | None = None) -> None:
    """A two-headed arrow between two curves at the deepest tree, labelled with their ratio.

    The number is read back out of the result file rather than written into the figure, so it
    cannot drift away from the measurement it claims to summarize. ``at`` picks the depth to
    anchor on; without it, or when that depth has no measurement on both curves, the arrow
    falls back to the deepest tree they share.
    """

    def curve(series):
        method, order = series
        return {
            r["depth"]: r["median_s"]
            for r in records
            if r["dataset"] == dataset
            and r["method"] == method
            and r["order"] == order
            and r.get("status") == "ok"
        }

    hi, lo = curve(slower), curve(faster)
    shared = set(hi) & set(lo)
    if not shared:
        return
    depth = at if at in shared else max(shared)
    y_hi, y_lo = hi[depth], lo[depth]
    ax.annotate(
        "",
        xy=(depth, y_lo),
        xytext=(depth, y_hi),
        arrowprops={
            "arrowstyle": "<->",
            "color": INK_2,
            "linewidth": 1.4,
            "shrinkA": 4,
            "shrinkB": 4,
        },
        zorder=6,
    )
    ax.annotate(
        f"{y_hi / y_lo:.0f}× faster",
        xy=(depth, (y_hi * y_lo) ** 0.5),
        xytext=(-7, 0),
        textcoords="offset points",
        ha="right",
        va="center",
        fontsize=9.5,
        color=INK,
        fontweight="bold",
        zorder=7,
        bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1.5, "alpha": 0.85},
    )


def depth_real() -> None:
    records = load_results("depth_real")["records"]
    acc = accuracy_map("real")
    for dataset in ("superconductivity", "heloc", "bioresponse"):
        fig, ax = _depth_panel(records, dataset, TITLES[dataset], ORDER2_NOTE, acc)
        if dataset in RATIO_ARROW:
            slower, faster, at = RATIO_ARROW[dataset]
            ratio_arrow(ax, records, dataset, slower, faster, at=at)
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
        line, marks, color, label = series(method, order)
        plot_series(ax, [p[0] for p in pts], [p[1] for p in pts], line, marks)
        labels.add(pts[-1][0], pts[-1][1], label, color)
    ax.axhline(TOL, color=MUTED, linestyle=DASH, linewidth=1.2, zorder=1)
    ax.annotate(
        "numerically instable above this line",
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
    encoding_legend(ax, list(COLOUR_KEYS), ncol=3)
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
