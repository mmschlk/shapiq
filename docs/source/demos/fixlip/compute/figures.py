"""Draw the figures of the FIxLIP demo's page: ``../figures/<name>-<theme>.webp``.

The page explains the demo step by step on its first image-text pair, the baby elephant and the
egret, as SigLIP 2 sees it. Every figure is drawn twice, for the light and the dark theme of the
docs. The figures read the demo's data (``../assets/data.js``), its images, and the model's
predictions for the masks they show (``figures.json``, written by evaluate.py). The pointing game's
numbers are those of the FIxLIP paper (Table 1). Nothing here runs a model.

Usage::

    python figures.py            # every figure
    python figures.py game cross # only these

The figures are laid out in points for the Inter typeface, which has to be installed.
"""

from __future__ import annotations

import argparse
import io
import json
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, PathPatch, Rectangle
from matplotlib.path import Path as MplPath
from pack import read_data
from PIL import Image
from scipy.stats import spearmanr

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

HERE = Path(__file__).resolve().parent
ASSETS = HERE.parent / "assets"
OUT = HERE.parent / "figures"
PHOTO = "elephant"
WIDTH = 520.0  # of a figure, in points. The page shows it about 1.4 times as large.
DPI = 200
RED, BLUE = "#ff0d57", "#1e88e5"  # the demo's colors of what raises and lowers the similarity
MASKED = (128, 128, 128)  # a masked patch, the mean color of SigLIP 2's normalization
THEMES = {
    "light": {
        "ink": "#1b1e27",
        "muted": "#5b6275",
        "line": "#d5d9e6",
        "raised": "#eef0f7",
        "masked": "#c4c7d1",
        "surface": "#ffffff",  # the background of the page
        "accent": "#6a57cd",
        "second": "#0f9d8a",
        "gray": "#8b91a1",
    },
    "dark": {
        "ink": "#e6e8ee",
        "muted": "#a3a9ba",
        "line": "#3a3f4e",
        "raised": "#2a2e3a",
        "masked": "#4a4f5e",
        "surface": "#131416",
        "accent": "#8f7fe8",
        "second": "#1fa896",
        "gray": "#7d8496",
    },
}
# FIxLIP's Table 1: pointing game recognition of CLIP (ViT-B/32) on ImageNet-1k, 1 to 4 objects
TABLE_1 = {
    "GAME": [0.61, 0.43, 0.33, 0.28],
    "Grad-ECLIP": [0.68, 0.45, 0.33, 0.28],
    "Shapley values": [0.70, 0.56, 0.46, 0.37],
    "exCLIP": [0.73, 0.88, 0.89, 0.92],
    "FIxLIP": [0.81, 0.80, 0.81, 0.83],  # weighted Banzhaf interactions, p = 0.5
}
# the four images of the pointing game's sketch and the class names of its text
MOSAIC = {
    "elephant": "elephant",
    "giraffe-tree": "giraffe",
    "clock-tower": "clock",
    "pineapple": "pineapple",
}

Theme = dict[str, str]
Tint = tuple[Any, float]  # a color and its opacity
Point = tuple[float, float]


class Data:
    """The demo's data and the model's predictions, as the figures read them."""

    def __init__(self) -> None:
        """Read ``data.js``, ``figures.json`` and the image."""
        self.demo = read_data(ASSETS)
        self.values = json.loads((HERE / "figures.json").read_text(encoding="utf-8"))
        self.image = self.photo(PHOTO)
        self.models = {model["id"]: model for model in self.demo["models"]}

    @staticmethod
    def photo(name: str) -> np.ndarray:
        """Return one of the demo's images."""
        return np.asarray(Image.open(ASSETS / f"{name}.jpg").convert("RGB"))

    def explanation(self, model: str = "siglip2") -> dict[str, Any]:
        """Return an explanation of the pair with its interactions as a matrix."""
        e = dict(self.demo["explanations"][f"{PHOTO}/{model}"])
        e["first"] = np.array(e["first"])
        n = len(e["first"])
        pairs = np.zeros((n, n))
        pairs[np.triu_indices(n, 1)] = e["pairs"]
        e["pairs"] = pairs + pairs.T
        e["words"] = [text for text, _ in e["tokens"]]
        e["n_patches"] = e["grid"] ** 2
        return e


# -- drawing -----------------------------------------------------------------------------------


def canvas(height: float) -> tuple[Figure, Axes]:
    """Return a figure with one axes that covers it, in points, with y pointing down."""
    fig = plt.figure(figsize=(WIDTH / 72, height / 72))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(height, 0)
    ax.axis("off")
    ax.set_autoscale_on(False)
    return fig, ax


def trim(fig: Figure, ax: Axes, height: float) -> None:
    """Cut the canvas to a height, once it is known where the drawing ends."""
    fig.set_size_inches(WIDTH / 72, height / 72)
    ax.set_ylim(height, 0)


def axes_at(fig: Figure, x: float, y: float, w: float, h: float) -> Axes:
    """Add the axes of a chart at a place in points, from the canvas's upper left corner."""
    height = fig.get_figheight() * 72
    return fig.add_axes((x / WIDTH, 1 - (y + h) / height, w / WIDTH, h / height))


def text_width(ax: Axes, text: str, **kwargs: Any) -> float:
    """Return the width of a text in points."""
    artist = ax.text(0, 0, text, **kwargs)
    width = artist.get_window_extent(ax.figure.canvas.get_renderer()).width * 72 / ax.figure.dpi
    artist.remove()
    return width


def mix(color: str, share: float, theme: Theme) -> tuple[float, ...]:
    """Return a color at a share of its strength over the page's background, without opacity."""
    back = np.array(mpl.colors.to_rgb(theme["surface"]))
    return tuple(back + share * (np.array(mpl.colors.to_rgb(color)) - back))


def rounded(
    ax: Axes, x: float, y: float, w: float, h: float, radius: float = 3, **kwargs: Any
) -> FancyBboxPatch:
    """Draw a box with rounded corners."""
    box = FancyBboxPatch(
        (x + radius, y + radius),
        w - 2 * radius,
        h - 2 * radius,
        boxstyle=f"round,pad={radius},rounding_size={radius}",
        **({"linewidth": 0} | kwargs),
    )
    ax.add_patch(box)
    return box


def draw_photo(
    ax: Axes,
    x: float,
    y: float,
    size: float,
    image: np.ndarray,
    grid: int,
    *,
    shown: Iterable[int] | None = None,
    tints: dict[int, Tint] | None = None,
    veil: Tint | None = None,
    lines: str | None = None,
) -> Callable[[int], Point]:
    """Draw the image, its absent patches masked, and return where the center of a patch is.

    ``shown`` are the patches that are present (all without it), ``tints`` a color and an opacity
    for patches, ``veil`` a color and an opacity over the patches that are not shown instead of the
    mask, and ``lines`` the color of the lines of the grid.
    """
    pixels = image.copy()
    cell = pixels.shape[0] // grid
    present = set(range(grid * grid)) if shown is None else set(shown)
    overlay = np.zeros((grid, grid, 4))
    for k in range(grid * grid):
        row, col = divmod(k, grid)
        if k not in present and veil is None:
            pixels[row * cell : (row + 1) * cell, col * cell : (col + 1) * cell] = MASKED
        elif k not in present:
            overlay[row, col] = (*mpl.colors.to_rgb(veil[0]), veil[1])
        if tints and k in tints:
            overlay[row, col] = (*mpl.colors.to_rgb(tints[k][0]), tints[k][1])
    extent = (x, x + size, y + size, y)
    clip = rounded(ax, x, y, size, size, radius=4, facecolor="none")
    for layer, interpolation in ((pixels, "antialiased"), (overlay, "nearest")):
        shown_layer = ax.imshow(layer, extent=extent, interpolation=interpolation, aspect="auto")
        shown_layer.set_clip_path(clip)
    if lines:
        for k in range(1, grid):
            at = k * size / grid
            ax.plot([x + at, x + at], [y, y + size], color=lines, linewidth=0.4, alpha=0.7)
            ax.plot([x, x + size], [y + at, y + at], color=lines, linewidth=0.4, alpha=0.7)
    return lambda k: (x + (k % grid + 0.5) * size / grid, y + (k // grid + 0.5) * size / grid)


def draw_tokens(
    ax: Axes,
    x: float,
    y: float,
    words: list[str],
    theme: Theme,
    *,
    shown: Iterable[int] | None = None,
    tints: dict[int, Tint] | None = None,
    column: bool = False,
    wrap: float | None = None,
    size: float = 8,
    step: float | None = None,
    bold: Iterable[int] = (),
    outlined: Iterable[int] = (),
) -> list[tuple[float, float, float]]:
    """Draw the text's tokens as chips and return the left, the right and the middle of each.

    The chips stand in a row from ``(x, y)``, broken at the width ``wrap``, or in a ``column``.
    A token that is not ``shown`` is a gray chip without its text, as a masked patch. ``step`` is
    the distance from one row of chips to the next.
    """
    present = set(range(len(words))) if shown is None else set(shown)
    pad, gap = 3.0, (2.5 if step is None else 1.6)
    height = size + 5 if step is None else step - gap
    places, left, top = [], x, y
    widths = [
        text_width(ax, word, fontsize=size, fontweight="semibold" if k in bold else "regular")
        for k, word in enumerate(words)
    ]
    for k, word in enumerate(words):
        width = (max(widths) if column else widths[k]) + 2 * pad
        if not column and wrap is not None and left + width > x + wrap + 0.5:
            left, top = x, top + height + gap
        chip = theme["raised"] if k in present else theme["masked"]
        rounded(ax, left, top, width, height, radius=2.5, facecolor=chip)
        if tints and k in tints and k in present:  # the tint lies on the chip, so a weak one shows
            rounded(ax, left, top, width, height, 2.5, facecolor=tints[k][0], alpha=tints[k][1])
        if k in outlined:
            rounded(ax, left, top, width, height, radius=2.5, fill=False, ec=theme["ink"], lw=1.0)
        if k in present:
            ax.text(
                left + pad if column else left + width / 2,
                top + height / 2,
                word,
                fontsize=size,
                fontweight="semibold" if k in bold else "regular",
                color=theme["ink"],
                ha="left" if column else "center",
                va="center_baseline",
            )
        places.append((left, left + width, top + height / 2))
        if column:
            top += height + gap
        else:
            left += width + gap
    return places


def tint_of(value: float, largest: float, strength: float = 0.85, floor: float = 0.0) -> Tint:
    """Return the color and the opacity of a value, red or blue by its sign, as the demo's.

    The opacity grows from ``floor`` to ``strength`` with the value's share of the largest one.
    """
    share = min(abs(value) / largest, 1) if largest > 0 else 0
    return (RED if value >= 0 else BLUE), (floor + (strength - floor) * share if value else 0)


def edge(
    ax: Axes,
    a: Point,
    b: Point,
    value: float,
    largest: float,
    theme: Theme,
    bend: float = 0.0,
    reach: float | None = None,
) -> None:
    """Draw an interaction as a line between two players, wider and stronger the larger it is.

    The line bends by the share ``bend`` of its length, or reaches ``reach`` points to the right.
    Its color has no opacity, so that a red and a blue line do not blend where they cross.
    """
    share = min(abs(value) / largest, 1)
    (x1, y1), (x2, y2) = a, b
    mid = ((x1 + x2) / 2 + bend * (y2 - y1), (y1 + y2) / 2 - bend * (x2 - x1))
    path = MplPath([a, mid, b], [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3])
    if reach is not None:  # an arc between two points of the same x
        out = x1 + reach / 0.75
        path = MplPath([a, (out, y1), (out, y2), b], [MplPath.MOVETO] + [MplPath.CURVE4] * 3)
    ax.add_patch(
        PathPatch(
            path,
            facecolor="none",
            edgecolor=mix(RED if value >= 0 else BLUE, 0.4 + 0.6 * share, theme),
            linewidth=0.6 + 2.6 * share,
            capstyle="round",
            zorder=3 + share,
        )
    )


def label(ax: Axes, x: float, y: float, text: str, theme: Theme, **kwargs: Any) -> None:
    """Write a text in the secondary color."""
    ax.text(x, y, text, **({"fontsize": 8, "color": theme["muted"], "va": "center"} | kwargs))


def heading(ax: Axes, x: float, y: float, text: str, theme: Theme, **kwargs: Any) -> None:
    """Write the title of a panel."""
    options = {"fontsize": 9, "color": theme["ink"], "fontweight": "semibold", "va": "center"}
    ax.text(x, y, text, **(options | kwargs))


def swatch(ax: Axes, x: float, y: float, color: str, text: str, theme: Theme) -> float:
    """Draw an entry of a legend, a square of a color and its text, and return its right end."""
    ax.add_patch(Rectangle((x, y - 4), 8, 8, color=color, lw=0))
    label(ax, x + 12, y, text, theme)
    return x + 12 + text_width(ax, text, fontsize=8)


def arrow(ax: Axes, a: Point, b: Point, theme: Theme) -> None:
    """Draw an arrow from one point to another."""
    ax.add_patch(
        FancyArrowPatch(
            a,
            b,
            arrowstyle="-|>,head_length=3.2,head_width=1.8",
            color=theme["muted"],
            linewidth=0.9,
            shrinkA=0,
            shrinkB=0,
            mutation_scale=1,
        )
    )


def number(value: float, digits: int = 3) -> str:
    """Return a number with a true minus sign."""
    return f"{value:.{digits}f}".replace("-", "\N{MINUS SIGN}")


def signed(value: float, digits: int = 3) -> str:
    """Return a number with its sign."""
    return ("+" if value >= 0 else "\N{MINUS SIGN}") + f"{abs(value):.{digits}f}"


def chart(ax: Axes, theme: Theme, *, grid: str | None = "y") -> None:
    """Style the axes of a chart: hairlines in the theme's line color, no box."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme["line"])
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=theme["muted"], labelsize=8, length=0, pad=4)
    ax.set_facecolor("none")
    if grid:
        ax.grid(axis=grid, color=theme["line"], linewidth=0.6, alpha=0.7)
        ax.set_axisbelow(True)
    ax.xaxis.label.set_color(theme["muted"])
    ax.yaxis.label.set_color(theme["muted"])
    ax.xaxis.label.set_size(8)
    ax.yaxis.label.set_size(8)


def players_of(data: Data, players: Iterable[int]) -> tuple[list[int], list[int]]:
    """Split players into the patches and the tokens (counted from 0) among them."""
    n_patches = data.values["grid"] ** 2
    players = list(players)
    return [k for k in players if k < n_patches], [k - n_patches for k in players if k >= n_patches]


def card(
    ax: Axes,
    x: float,
    y: float,
    size: float,
    data: Data,
    theme: Theme,
    mask: dict[str, Any],
    title: str,
    note: str | None = None,
) -> float:
    """Draw a mask: the masked image, the masked text under it, and the model's similarity.

    Returns the y of the card's lower end.
    """
    patches, tokens = players_of(data, mask["players"])
    heading(ax, x, y + 6, title, theme)
    y += 16
    if note:
        label(ax, x, y + 3, note, theme)
        y += 12
    draw_photo(ax, x, y, size, data.image, data.values["grid"], shown=patches)
    places = draw_tokens(ax, x, y + size + 6, data.values["tokens"], theme, shown=tokens, wrap=size)
    below = places[-1][2] + 16
    label(ax, x, below, "similarity", theme)
    ax.text(
        x + size,
        below,
        number(mask["value"]),
        fontsize=9,
        fontweight="semibold",
        color=theme["ink"],
        ha="right",
        va="center",
    )
    return below + 7


def interactions_of(
    e: dict[str, Any], count: int, *, cross: bool = False
) -> list[tuple[int, int, float]]:
    """Return the strongest interactions of an explanation, all or only the cross-modal ones."""
    n, n_patches = len(e["first"]), e["n_patches"]
    found = [
        (i, j, e["pairs"][i, j])
        for i in range(n)
        for j in range(i + 1, n)
        if not cross or (i < n_patches <= j)
    ]
    return sorted(found, key=lambda item: -abs(item[2]))[:count]


def panel(
    ax: Axes,
    x: float,
    y: float,
    size: float,
    data: Data,
    theme: Theme,
    e: dict[str, Any],
    *,
    values: np.ndarray | None = None,
    edges: list[tuple[int, int, float]] | None = None,
    members: list[int] | None = None,
    largest: float | None = None,
    token_size: float = 8,
    chosen: int | None = None,
    apart: bool = False,
) -> float:
    """Draw an explanation: the image, the text's tokens in a column, and interactions.

    ``values`` tints every player (its main effect or a conditioned interaction), relative to the
    ``largest`` value or, with ``apart``, to the largest of its own modality. ``edges`` are the
    interactions to draw, ``members`` the players of a clique, the others stepping back, and
    ``chosen`` a player to frame. Returns the y of the panel's lower end.
    """
    n_patches, n_words = e["n_patches"], len(e["words"])
    step = max(token_size + 4.5, min(size / n_words, token_size + 7))
    height = max(size, step * n_words - 1.6)
    top = y + (height - size) / 2
    tints_p, tints_t = {}, {}
    if values is not None:
        scale = largest or np.abs(values).max()
        scale_p = np.abs(values[:n_patches]).max() if apart else scale
        scale_t = np.abs(values[n_patches:]).max() if apart else scale
        tints_p = {k: tint_of(values[k], scale_p, strength=0.6) for k in range(n_patches)}
        tints_t = {
            k - n_patches: tint_of(values[k], scale_t, floor=0.12)
            for k in range(n_patches, len(values))
        }
    shown = None if members is None else [k for k in members if k < n_patches]
    veil = (theme["surface"], 0.72)
    center = draw_photo(
        ax, x, top, size, data.image, e["grid"], shown=shown, tints=tints_p, veil=veil
    )
    bold = [] if members is None else [k - n_patches for k in members if k >= n_patches]
    places = draw_tokens(
        ax,
        x + size + 8,
        y + (height - step * n_words + 1.6) / 2,
        e["words"],
        theme,
        tints=tints_t,
        column=True,
        size=token_size,
        step=step,
        bold=bold,
        outlined=[] if chosen is None or chosen < n_patches else [chosen - n_patches],
    )
    cell = size / e["grid"]
    framed = list(shown or []) + ([chosen] if chosen is not None and chosen < n_patches else [])
    for k in framed:
        cx, cy = center(k)
        frame = Rectangle((cx - cell / 2, cy - cell / 2), cell, cell, fill=False, ec=theme["ink"])
        frame.set_linewidth(1.0)
        ax.add_patch(frame)
    if members is not None:  # the tokens outside the clique step back
        for k, (left, right, middle) in enumerate(places):
            if k not in bold:
                box = (left - 0.5, middle - step / 2, right - left + 1, step)
                rounded(ax, *box, facecolor=theme["surface"], alpha=0.6)
    edges = edges or []
    scale = largest or max((abs(value) for _, _, value in edges), default=1.0)
    # the ends of the arcs between two tokens, spread over the height of a chip
    ends: dict[int, list[int]] = {}
    for i, j, _ in edges:
        if i >= n_patches:
            ends.setdefault(i, []).append(j)
            ends.setdefault(j, []).append(i)

    def end(k: int, other: int) -> float:
        others = sorted(ends[k])
        share = (others.index(other) + 0.5) / len(others) - 0.5
        return places[k - n_patches][2] + share * 0.6 * (step - 1.6)

    for i, j, value in edges:
        if i < n_patches <= j:  # cross-modal: from the patch to the token's left end
            left, _, middle = places[j - n_patches]
            edge(ax, center(i), (left, middle), value, scale, theme)
        elif j < n_patches:  # between two patches
            edge(ax, center(i), center(j), value, scale, theme, bend=0.18)
        else:  # between two tokens: an arc on the right of the column
            right = places[0][1] + 1.5
            a, b = end(i, j), end(j, i)
            edge(
                ax, (right, a), (right, b), value, scale, theme, reach=min(3 + 0.2 * abs(b - a), 11)
            )
    return y + height


# -- the figures -------------------------------------------------------------------------------


def encoders(data: Data, theme: Theme) -> Figure:
    """How a vision-language encoder scores an image-text pair."""
    fig, ax = canvas(204)
    e = data.explanation()
    top, bottom = 72.0, 176.0  # the middle of the image's row and of the text's row
    draw_photo(ax, 0, top - 62, 124, data.image, e["grid"], lines="#ffffff")
    label(ax, 62, 143, "64 image patches", theme, ha="center")
    places = draw_tokens(ax, 0, bottom - 14.25, e["words"], theme, wrap=140)
    right = max(place[1] for place in places)
    label(ax, right / 2, bottom + 23, "10 text tokens", theme, ha="center")
    for y, name in ((top, "Vision encoder"), (bottom, "Language encoder")):
        rounded(ax, 262, y - 17, 94, 34, radius=5, facecolor=theme["raised"])
        ax.text(309, y - 4.5, name, fontsize=9, color=theme["ink"], ha="center", va="center")
        label(ax, 309, y + 7, "ViT-B/32" if y == top else "transformer", theme, ha="center")
    arrow(ax, (132, top), (256, top), theme)
    arrow(ax, (right + 8, bottom), (256, bottom), theme)
    rng = np.random.default_rng(5)
    for y, name in ((top, "image embedding"), (bottom, "text embedding")):
        arrow(ax, (360, y), (378, y), theme)
        for k, shade in enumerate(rng.uniform(0.2, 1.0, 8)):
            color = mix(theme["accent"], shade, theme)
            ax.add_patch(Rectangle((383 + k * 8.5, y - 4), 7.5, 8, color=color, lw=0))
        label(ax, 383 + 34, y + (15 if y == bottom else -15), name, theme, ha="center")
    middle = (top + bottom) / 2
    arrow(ax, (417, top + 9), (452, middle - 16), theme)
    arrow(ax, (417, bottom - 9), (452, middle + 16), theme)
    ax.text(
        486,
        middle - 5,
        number(e["similarity"]),
        fontsize=15,
        fontweight="semibold",
        color=theme["ink"],
        ha="center",
        va="center",
    )
    label(ax, 486, middle + 10, "cosine similarity", theme, ha="center")
    return fig


def scale(data: Data, theme: Theme) -> Figure:
    """The similarity of the pair on each model's scale."""
    fig, canvas_ax = canvas(152)
    kinds = {
        "matching": (theme["accent"], 0.14, "200 matching pairs"),
        "unrelated": (theme["gray"], -0.14, "39,800 unrelated pairs"),
    }
    x = 0.0
    for color, _, text in kinds.values():
        x = swatch(canvas_ax, x, 7, color, text, theme) + 18
    ax = axes_at(fig, 50, 22, 462, 92)
    chart(ax, theme, grid="x")
    rows = {"siglip2": 1.0, "clip": 0.0}
    for model, y in rows.items():
        norms = data.models[model]["norms"]
        for kind, (color, shift, _) in kinds.items():
            q = norms[kind]  # the 0th, 5th, ..., 100th percentiles
            ax.plot([q[1], q[19]], [y + shift] * 2, color=mix(color, 0.45, theme), linewidth=2)
            ax.plot([q[5], q[15]], [y + shift] * 2, color=color, linewidth=6, solid_capstyle="butt")
            gap = [y + shift - 0.09, y + shift + 0.09]
            ax.plot([q[10]] * 2, gap, color=theme["surface"], linewidth=1.4)
        value = data.demo["explanations"][f"{PHOTO}/{model}"]["similarity"]
        ax.plot([value], [y + 0.14], "o", color=theme["ink"], markersize=6, mec=theme["surface"])
        ax.annotate(
            f"elephant pair, {number(value)}",
            (value, y + 0.14),
            xytext=(0, 9),
            textcoords="offset points",
            fontsize=8,
            color=theme["ink"],
            ha="center",
        )
    ax.set_yticks(list(rows.values()), ["SigLIP 2", "CLIP"])
    ax.tick_params(axis="y", labelsize=9, labelcolor=theme["ink"])
    ax.set_ylim(-0.45, 1.62)
    ax.set_xlim(-0.2, 0.42)
    ax.set_xlabel("cosine similarity")
    ax.spines["left"].set_visible(False)
    return fig


def modalities(data: Data, theme: Theme) -> Figure:
    """The similarity needs both modalities, and the words go with their parts of the image."""
    fig, ax = canvas(300)
    presets = data.values["presets"]
    size, gap = 86.0, 8.0
    cards = {"empty": "No tokens", "image": "Image only", "text": "Text only", "full": "All tokens"}
    below = [
        card(ax, k * (size + gap), 0, size, data, theme, presets[key], title)
        for k, (key, title) in enumerate(cards.items())
    ]
    # the regions of the image against the words that name them
    thumb, cell = 36.0, 44.0
    left = WIDTH - thumb - 6 - 2 * cell - 5
    heading(ax, left, 6, "Regions and words", theme)
    words = {"elephant": "elephant", "bird": "white bird"}
    regions = ("elephant", "egret")
    values = np.array([[presets[f"{r}-{w}"]["value"] for w in words] for r in regions])
    low, high = data.values["empty"], values.max()
    top = 40.0
    for col, text in enumerate(words.values()):
        x = left + thumb + 6 + (col + 0.5) * (cell + 3) - 1.5
        ax.text(x, top - 9, text, fontsize=8, color=theme["ink"], ha="center", va="center")
    for row, region in enumerate(regions):
        y = top + row * (cell + 3)
        shown = data.values[region]
        draw_photo(ax, left, y + (cell - thumb) / 2, thumb, data.image, 8, shown=shown)
        for col in range(2):
            x = left + thumb + 6 + col * (cell + 3)
            share = max(values[row, col] - low, 0) / (high - low)
            fill = mix(theme["accent"], 0.08 + 0.52 * share, theme)
            rounded(ax, x, y, cell, cell, radius=4, facecolor=fill)
            ax.text(
                x + cell / 2,
                y + cell / 2,
                number(values[row, col]),
                fontsize=9,
                fontweight="semibold",
                color=theme["ink"],
                ha="center",
                va="center",
            )
    for k, text in enumerate(("similarity of one region", "with these words")):
        label(ax, left, top + 2 * cell + 16 + k * 11, text, theme)
    trim(fig, ax, max(below))
    return fig


def related(data: Data, theme: Theme) -> Figure:
    """What a saliency map, a cross-modal method and FIxLIP show of the same explanation."""
    fig, ax = canvas(300)
    e = data.explanation()
    order1 = e["first"] + 0.5 * e["pairs"].sum(axis=1)  # the first-order conversion, p = 1/2
    size, width = 104.0, 173.6
    titles = (
        ("First-order attributions", "a saliency map"),
        ("Cross-modal interactions", "a bipartite graph"),
        ("FIxLIP", "a complete graph"),
    )
    for k, (title, text) in enumerate(titles):
        heading(ax, k * width, 6, title, theme)
        label(ax, k * width, 18, text, theme)
    largest = np.abs(e["pairs"]).max()
    panel(ax, 0, 30, size, data, theme, e, values=order1, apart=True)
    edges = interactions_of(e, 12, cross=True)
    panel(ax, width, 30, size, data, theme, e, edges=edges, largest=largest)
    edges = interactions_of(e, 16)
    below = panel(ax, 2 * width, 30, size, data, theme, e, values=e["first"], edges=edges)
    trim(fig, ax, below + 3)
    return fig


def game(data: Data, theme: Theme) -> Figure:
    """Masks of the image-text pair and the similarity that the model predicts for them."""
    fig, ax = canvas(300)
    presets = data.values["presets"]
    cards = [
        ("All tokens", presets["full"]),
        ("No tokens", presets["empty"]),
        ("No elephant", presets["no-elephant"]),
        ("No egret", presets["no-egret"]),
        ("A random mask", data.values["examples"]["0.5"][0]),
    ]
    size = 92.0
    gap = (WIDTH - 2 - len(cards) * size) / (len(cards) - 1)
    below = [
        card(ax, k * (size + gap), 0, size, data, theme, mask, title)
        for k, (title, mask) in enumerate(cards)
    ]
    trim(fig, ax, max(below))
    return fig


def interaction(data: Data, theme: Theme) -> Figure:
    """A patch and a word, alone and together, on average over random masks of the others."""
    fig, canvas_ax = canvas(150)
    together = data.values["together"]
    panels = (
        ("egret", "The patch of the egret and “bird”"),
        ("mismatch", "The patch of the egret and “elephant”"),
    )
    for k, (key, title) in enumerate(panels):
        heading(canvas_ax, k * 266, 6, title, theme)
        ax = axes_at(fig, k * 266 + 70, 20, 178, 96)
        chart(ax, theme, grid="x")
        t = together[key]
        alone = [t["patch"] - t["neither"], t["word"] - t["neither"]]
        both = t["both"] - t["neither"]
        excess = both - sum(alone)
        ax.barh([2, 1, 0], [*alone, sum(alone)], height=0.5, color=theme["gray"])
        ax.barh([0], [excess], left=[sum(alone)], height=0.5, color=RED if excess >= 0 else BLUE)
        for y, value in zip((2, 1, 0), (*alone, both), strict=True):
            weight = "semibold" if y == 0 else "regular"
            at = max(value, sum(alone) if y == 0 else 0, 0) + 0.003
            ax.text(
                at, y, signed(value), fontsize=8, color=theme["ink"], va="center", weight=weight
            )
        # the part that the two add beyond the sum of what each adds alone
        middle = sum(alone) + excess / 2
        ax.plot([middle, middle], [-0.3, -0.52], color=theme["muted"], linewidth=0.8)
        note = f"interaction {signed(excess)}"
        side = "left" if middle < 0.06 else "right"  # away from the ends of the axis
        at = middle + (-0.002 if side == "left" else 0.002)
        ax.text(at, -0.6, note, fontsize=8, color=theme["ink"], ha=side, va="top")
        ax.set_yticks([2, 1, 0], ["only the patch", "only the word", "both"])
        ax.tick_params(axis="y", labelsize=8, labelcolor=theme["ink"])
        ax.set_xlim(-0.004, 0.118)
        ax.set_ylim(-1.25, 2.6)
        ax.set_xticks([0, 0.05, 0.1])
        ax.spines["left"].set_visible(False)
        ax.axvline(0, color=theme["muted"], linewidth=0.8)
        ax.set_xlabel("change of the similarity")
    return fig


def explanation(data: Data, theme: Theme) -> Figure:
    """The demo's explanation: main effects as tints, the strongest interactions as lines."""
    fig, ax = canvas(300)
    e = data.explanation()
    edges = interactions_of(e, 20)
    below = panel(ax, 0, 2, 206, data, theme, e, values=e["first"], edges=edges, token_size=9.5)
    x = 312.0
    heading(ax, x, 10, "Strongest main effects", theme)
    words = e["words"]
    together = data.values["together"]
    patches = {together[key]["players"][0]: f"{key} patch" for key in ("egret", "trunk")}

    def name(k: int) -> str:
        if k >= e["n_patches"]:
            return f"“{words[k - e['n_patches']]}”"
        row, col = divmod(k, e["grid"])
        return patches.get(k, f"patch {row + 1}, {col + 1}")

    def row(y: float, text: str, value: float) -> None:
        ax.text(x, y, text, fontsize=8, color=theme["ink"], va="center")
        ax.text(
            WIDTH - 2, y, signed(value), fontsize=8, color=theme["ink"], va="center", ha="right"
        )

    y = 26.0
    for k in np.argsort(-np.abs(e["first"]))[:3]:
        row(y, name(k), e["first"][k])
        y += 13
    heading(ax, x, y + 10, "Strongest interactions", theme)
    y += 26
    for i, j, value in interactions_of(e, 8):
        row(y, f"{name(i)} and {name(j)}", value)
        y += 13
    end = swatch(ax, x, y + 11, RED, "raises the similarity", theme)
    swatch(ax, end + 14, y + 11, BLUE, "lowers it", theme)
    trim(fig, ax, max(below + 2, y + 20))
    return fig


def conditioning(data: Data, theme: Theme) -> Figure:
    """The interactions of one token with all others, as heatmaps."""
    fig, ax = canvas(300)
    e = data.explanation()
    n_patches, size, width = e["n_patches"], 104.0, 173.6
    egret = data.values["together"]["egret"]["players"][0]
    chosen = (
        ("“elephant”", n_patches + e["words"].index("elephant")),
        ("“bird”", n_patches + e["words"].index("bird")),
        ("the patch of the egret", egret),
    )
    largest = max(np.abs(e["pairs"][k]).max() for _, k in chosen)
    for at, (name, k) in enumerate(chosen):
        heading(ax, at * width, 6, f"Conditioned on {name}", theme)
        values = e["pairs"][k]
        below = panel(
            ax, at * width, 20, size, data, theme, e, values=values, largest=largest, chosen=k
        )
    trim(fig, ax, below + 3)
    return fig


def clique_edges(e: dict[str, Any], members: list[int]) -> list[tuple[int, int, float]]:
    """Return the interactions inside a clique, without those too weak to see."""
    largest = np.abs(e["pairs"]).max()
    edges = [(i, j, e["pairs"][i, j]) for i in members for j in members if i < j]
    return [item for item in edges if abs(item[2]) > 0.08 * largest]


def cliques(data: Data, theme: Theme) -> Figure:
    """The demo's cliques of 2, 6 and 10 players, and what the model predicts for them alone."""
    fig, ax = canvas(300)
    e = data.explanation()
    size, width = 104.0, 173.6
    largest = np.abs(e["pairs"]).max()
    for at, count in enumerate((2, 6, 10)):
        members, value = e["cliques"][str(count)]
        heading(ax, at * width, 6, f"Clique of {count}", theme)
        edges = clique_edges(e, members)
        below = panel(
            ax, at * width, 20, size, data, theme, e, members=members, edges=edges, largest=largest
        )
        gain = data.values["cliques"][str(count)] - data.values["empty"]
        rows = (("value in the explanation", value), ("gain measured on the model", gain))
        for k, (text, shown) in enumerate(rows):
            y = below + 13 + k * 13
            label(ax, at * width, y, text, theme)
            ax.text(
                at * width + 160,
                y,
                signed(shown),
                fontsize=8.5,
                fontweight="semibold",
                color=theme["ink"],
                ha="right",
                va="center",
            )
    trim(fig, ax, below + 34)
    return fig


def masking(data: Data, theme: Theme) -> Figure:
    """Masks at p = 0.3, 0.5 and 0.7 and the similarities the model predicts for them."""
    fig, ax = canvas(222)
    width, size = 173.6, 72.0
    low, high = data.values["empty"], data.values["presets"]["full"]["value"]
    bins = np.linspace(-0.02, 0.24, 27)
    holdout = data.values["holdout"]
    tallest = max(np.histogram(h["value"], bins)[0].max() for h in holdout.values())
    for at, (p, masks) in enumerate(holdout.items()):
        x = at * width
        heading(ax, x, 6, f"p = {p}", theme)
        # of the four masks of every p, one that shows what p does
        mask = data.values["examples"][p][1 if p == "0.5" else 0]
        patches, tokens = players_of(data, mask["players"])
        draw_photo(ax, x, 18, size, data.image, data.values["grid"], shown=patches)
        places = draw_tokens(
            ax, x + size + 8, 18, data.values["tokens"], theme, shown=tokens, wrap=88
        )
        below = max(places[-1][2] + 16, 18 + size - 4.5)
        label(ax, x + size + 8, below, "similarity", theme)
        ax.text(
            x + size + 96,
            below,
            number(mask["value"]),
            fontsize=9,
            fontweight="semibold",
            color=theme["ink"],
            ha="right",
            va="center",
        )
        axes = axes_at(fig, x + 2, 122, 162, 50)
        chart(axes, theme, grid=None)
        values, kept = np.array(masks["value"]), np.array(masks["elephant"], dtype=bool)
        colors = [theme["accent"], theme["gray"]]
        axes.hist([values[kept], values[~kept]], bins, stacked=True, color=colors, rwidth=0.84)
        for value, text in ((low, "no tokens"), (high, "all tokens")):
            axes.axvline(value, color=theme["muted"], linewidth=0.8)
            axes.text(
                value,
                1.2 * tallest,
                text,
                fontsize=8,
                color=theme["muted"],
                ha="center",
                va="bottom",
                clip_on=False,
            )
        axes.set_xlim(-0.02, 0.24)
        axes.set_ylim(0, 1.12 * tallest)
        axes.set_yticks([])
        axes.set_xticks([0, 0.1, 0.2])
        axes.spines["left"].set_visible(False)
        axes.set_xlabel("similarity of 512 random masks")
    end = swatch(ax, 0, 215, theme["accent"], "masks that keep “elephant”", theme)
    swatch(ax, end + 18, 215, theme["gray"], "masks without it", theme)
    return fig


def cross(data: Data, theme: Theme) -> Figure:
    """Cross-modal sampling: every combination of 6 masked images and 6 masked texts."""
    fig, ax = canvas(240)
    c = data.values["cross"]
    values = np.array(c["values"])
    cell, thumb, left, top = 30.0, 27.0, 185.0, 31.0
    low, high = values.min(), values.max()
    words = data.values["tokens"]
    heading(ax, left, 6, "6 masked texts", theme)
    label(ax, left, 18, "6 passes of the language encoder", theme)
    for col, tokens in enumerate(c["captions"]):  # a text as two rows of its ten tokens
        x = left + col * (cell + 2)
        for k, word in enumerate(words):
            color = theme["accent"] if k in tokens else theme["masked"]
            square = Rectangle(
                (x + 1.7 + (k % 5) * 5.4, top + (k // 5) * 5.4), 4.4, 4.4, color=color
            )
            square.set_linewidth(0)
            ax.add_patch(square)
            if word == "elephant":  # the word that the similarity depends on the most
                frame = Rectangle((x + 0.9 + (k % 5) * 5.4, top - 0.8), 6.0, 6.0, fill=False)
                frame.set_edgecolor(theme["ink"])
                frame.set_linewidth(0.7)
                ax.add_patch(frame)
    grid_top = top + 15
    for row, patches in enumerate(c["photos"]):
        y = grid_top + row * (cell + 2)
        at = (left - thumb - 8, y + (cell - thumb) / 2)
        draw_photo(ax, *at, thumb, data.image, data.values["grid"], shown=patches)
        for col in range(6):
            x = left + col * (cell + 2)
            share = (values[row, col] - low) / (high - low)
            fill = mix(theme["accent"], 0.08 + 0.52 * share, theme)
            rounded(ax, x, y, cell, cell, radius=3, facecolor=fill)
            ax.text(
                x + cell / 2,
                y + cell / 2,
                number(values[row, col]),
                fontsize=8,
                color=theme["ink"],
                ha="center",
                va="center",
            )
    middle = grid_top + 3 * (cell + 2) - 1
    notes = (
        (left - thumb - 18, "right", "6 masked images", ("6 passes of the", "vision encoder")),
        (
            left + 6 * (cell + 2) + 12,
            "left",
            "36 similarities",
            ("from 12 passes", "instead of 72"),
        ),
    )
    for x, side, title, lines in notes:
        heading(ax, x, middle - 13, title, theme, ha=side)
        for k, text in enumerate(lines):
            label(ax, x, middle + k * 12, text, theme, ha=side)
    return fig


def r2(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Return the R^2 of a prediction, with the best constant added to it (as explain.py)."""
    y_hat = y_hat + np.mean(y - y_hat)
    return float(1 - np.sum((y - y_hat) ** 2) / np.sum((y - y.mean()) ** 2))


def faithfulness(data: Data, theme: Theme) -> Figure:
    """What the explanation predicts for 512 held-out masks against what the model predicts."""
    fig, canvas_ax = canvas(256)
    holdout = data.values["holdout"]["0.5"]
    y = np.array(holdout["value"])
    panels = (
        ("order1", "First order", "attributions only", theme["gray"]),
        ("order2", "Second order", "main effects and interactions", theme["accent"]),
    )
    for k, (key, title, text, color) in enumerate(panels):
        heading(canvas_ax, k * 266, 6, title, theme)
        label(canvas_ax, k * 266, 18, text, theme)
        ax = axes_at(fig, k * 266 + 44, 32, 190, 190)
        chart(ax, theme, grid="both")
        y_hat = np.array(holdout[key])
        y_hat = y_hat + np.mean(y - y_hat)  # the constant is fit to these masks, as in the demo
        ax.plot([0, 0.24], [0, 0.24], color=theme["muted"], linewidth=0.8)
        ax.scatter(y_hat, y, s=9, color=color, alpha=0.55, linewidths=0)
        ax.set_xlim(0, 0.24)
        ax.set_ylim(0, 0.24)
        ax.set_xticks([0, 0.1, 0.2])
        ax.set_yticks([0, 0.1, 0.2])
        ax.set_xlabel("similarity predicted by the explanation")
        if k == 0:
            ax.set_ylabel("similarity predicted by SigLIP 2")
        rho = spearmanr(y, y_hat)[0]
        note = f"$R^2$ = {r2(y, y_hat):.2f}\nrank correlation = {rho:.2f}"
        ax.text(0.232, 0.012, note, fontsize=8, color=theme["ink"], ha="right", va="bottom")
    return fig


def curves(data: Data, theme: Theme) -> Figure:
    """Insertion and deletion curves of the elephant pair, and two of the subsets behind them."""
    fig, canvas_ax = canvas(300)
    found, subsets = data.values["curves"], data.values["subsets"]
    n = len(found["order2-max"])
    low, high = data.values["empty"], data.values["presets"]["full"]["value"]
    # two subsets that the second-order explanation finds, as masks
    shown = {"max": 10, "min": n - 5}
    titles = {"max": "Valued the most", "min": "Valued the least"}
    below = []
    for k, (kind, size) in enumerate(shown.items()):
        mask = {
            "players": subsets[f"order2-{kind}"][size - 1],
            "value": found[f"order2-{kind}"][size - 1],
        }
        x = WIDTH - 178 + k * 94
        below.append(
            card(canvas_ax, x, 0, 82, data, theme, mask, titles[kind], f"of {size} tokens")
        )
    trim(fig, canvas_ax, max(*below, 226))
    heading(canvas_ax, 0, 6, "Insertion and deletion curves", theme)
    ax = axes_at(fig, 44, 34, 246, 158)
    chart(ax, theme)
    sizes = np.arange(n + 1)
    series = (
        ("order1", theme["gray"], "first order", (0, (4, 2))),
        ("order2", theme["accent"], "second order", "solid"),
    )
    x = 0.0
    for key, color, name, style in series:
        top = np.array([low, *found[f"{key}-max"]])
        bottom = np.array([low, *found[f"{key}-min"]])
        if key == "order2":
            ax.fill_between(sizes, bottom, top, color=color, alpha=0.1, linewidth=0)
        for curve in (top, bottom):
            ax.plot(sizes, curve, color=color, linewidth=1.5, linestyle=style)
        # the paper's AID, the sum of the differences of the two curves
        text = f"{name}, AID {np.sum(top - bottom):.1f}"
        canvas_ax.plot([x, x + 14], [20, 20], color=color, linewidth=1.5, linestyle=style)
        label(canvas_ax, x + 18, 20, text, theme)
        x += 18 + text_width(canvas_ax, text, fontsize=8) + 16
    for kind, size in shown.items():
        value = found[f"order2-{kind}"][size - 1]
        ax.plot([size], [value], "o", color=theme["ink"], markersize=5, mec=theme["surface"])
    for value, text in ((high, "all tokens"), (low, "no tokens")):
        ax.axhline(value, color=theme["muted"], linewidth=0.8)
        ax.text(n + 1.5, value, text, fontsize=8, color=theme["muted"], va="center", clip_on=False)
    for value, text in ((0.243, "subsets valued the most"), (0.082, "subsets valued the least")):
        ax.text(34, value, text, fontsize=8, color=theme["ink"], ha="center", va="center")
    ax.set_xlim(0, n)
    ax.set_ylim(0, 0.26)
    ax.set_yticks([0, 0.1, 0.2])
    ax.set_xticks([0, 20, 40, 60, n])
    ax.set_xlabel("tokens in the subset")
    ax.set_ylabel("similarity predicted by SigLIP 2")
    return fig


def pointing(data: Data, theme: Theme) -> Figure:
    """A sketch of the pointing game, and the paper's results for it."""
    fig, canvas_ax = canvas(206)
    # the sketch: four images in one, and a text of their class names
    heading(canvas_ax, 0, 6, "Pointing game", theme)
    label(canvas_ax, 0, 18, "what counts as correct for “elephant”", theme)
    size, top = 124.0, 32.0
    half = size / 2
    centers = []
    for k, name in enumerate(MOSAIC):
        x, y = (k % 2) * (half + 1.5), top + (k // 2) * (half + 1.5)
        tint = tint_of(1 if k == 0 else -1, 1, strength=0.3)
        draw_photo(canvas_ax, x, y, half - 1.5, data.photo(name), 1, tints={0: tint})
        centers.append((x + half / 2, y + half / 2))
    column = (size + 34, top + (size - 4 * 16.5 + 1.6) / 2)
    places = draw_tokens(
        canvas_ax, *column, list(MOSAIC.values()), theme, column=True, step=16.5, outlined=[0]
    )
    word = (places[0][0], places[0][2])
    for k, center in enumerate(centers):
        edge(canvas_ax, center, word, 0.6 if k == 0 else -0.6, 1, theme)
    end = swatch(canvas_ax, 0, top + size + 14, RED, "a positive interaction", theme)
    swatch(canvas_ax, end + 12, top + size + 14, BLUE, "a negative one", theme)
    # the results of the paper
    left = 280.0
    heading(canvas_ax, left, 6, "Pointing game recognition", theme)
    label(canvas_ax, left, 18, "CLIP (ViT-B/32), Table 1 of the paper", theme)
    ax = axes_at(fig, left + 32, 32, 122, 138)
    chart(ax, theme)
    colors = {"exCLIP": theme["second"], "FIxLIP": theme["accent"]}
    objects = [1, 2, 3, 4]
    names: dict[float, list[str]] = {}  # the methods by where their lines end
    for name, scores in TABLE_1.items():
        ax.plot(
            objects,
            scores,
            color=colors.get(name, theme["gray"]),
            linewidth=1.5,
            marker="o",
            markersize=4.5,
            mec=theme["surface"],
            mew=0.8,
            zorder=3 if name in colors else 2,
        )
        names.setdefault(scores[-1], []).append(name)
    for value, methods in names.items():
        text = ", ".join(reversed(methods))
        at = value + (0.03 if value == TABLE_1["Shapley values"][-1] else 0)
        ax.text(4.2, at, text, fontsize=8, color=theme["ink"], va="center", clip_on=False)
    ax.axhline(0.25, color=theme["muted"], linewidth=0.8)
    ax.text(1, 0.235, "random", fontsize=8, color=theme["muted"], va="top")
    ax.set_xlim(0.85, 4.1)
    ax.set_ylim(0, 1)
    ax.set_xticks(objects)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xlabel("class names in the text")
    return fig


def models(data: Data, theme: Theme) -> Figure:
    """The cliques of 6 of CLIP and SigLIP 2 for the same image-text pair."""
    fig, ax = canvas(300)
    size, width = 130.0, 270.0
    for at, (model, name) in enumerate((("clip", "CLIP"), ("siglip2", "SigLIP 2"))):
        e = data.explanation(model)
        members, value = e["cliques"]["6"]
        heading(ax, at * width, 6, f"{name} (ViT-B/32)", theme)
        # the lines are relative to the model's own strongest interaction
        largest = np.abs(e["pairs"]).max()
        edges = clique_edges(e, members)
        below = panel(
            ax, at * width, 20, size, data, theme, e, members=members, edges=edges, largest=largest
        )
        note = f"similarity {number(e['similarity'])}, value of the clique {signed(value)}"
        label(ax, at * width, below + 13, f"{note}, $R^2$ = {e['r2']:.2f}", theme)
    trim(fig, ax, below + 22)
    return fig


FIGURES = {
    figure.__name__: figure
    for figure in (
        encoders,
        scale,
        modalities,
        related,
        game,
        interaction,
        explanation,
        conditioning,
        cliques,
        masking,
        cross,
        faithfulness,
        curves,
        pointing,
        models,
    )
}


def main() -> None:
    """Draw the figures in both themes."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("figures", nargs="*", metavar="figure", help=", ".join(FIGURES))
    args = parser.parse_args()
    if any(font.name == "Inter" for font in font_manager.fontManager.ttflist):
        mpl.rcParams["font.family"] = "Inter"
    else:
        warnings.warn("Inter is not installed, so texts will not fit their places.", stacklevel=1)
    mpl.rcParams["mathtext.default"] = "regular"
    OUT.mkdir(exist_ok=True)
    data = Data()
    for name in args.figures or FIGURES:
        for theme_name, theme in THEMES.items():
            fig = FIGURES[name](data, theme)
            buffer = io.BytesIO()
            fig.savefig(buffer, dpi=DPI, transparent=True, format="png")
            plt.close(fig)
            # WebP keeps the images and the transparent background in a fifth of a PNG's size
            Image.open(buffer).save(OUT / f"{name}-{theme_name}.webp", quality=88, method=6)
        print(name)


if __name__ == "__main__":
    main()
