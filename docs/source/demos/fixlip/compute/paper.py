"""Read the data of the plots of the FIxLIP paper that the page shows again.

The page re-renders four results of the paper (Baniecki et al., NeurIPS 2025) in its own visual
language. The data of the three plots is read from the vector figures of the paper's LaTeX
sources, in which every box and every curve is a path with exact coordinates. The axes are
calibrated on the grid lines and the ticks. The scores of the pointing game are those of Table 1.
All of it is written to ``../assets/figures-data.js`` under ``paper``.

- ``correlation`` and ``r2``: the box plots of the p-faithfulness correlation (p = 0.7) and of
  R^2 (p = 0.5) for CLIP (ViT-B/32) on MS COCO, as the quartiles and the whiskers of each method.
- ``curves``: the insertion and deletion curves for CLIP (ViT-B/32) on MS COCO, as 51 points for
  each method, from all tokens to none, with the area between the curves.
- ``runtime``: the seconds that the model-agnostic and the cross-modal estimator take for SigLIP 2
  (ViT-B/32), for budgets from 2^15 to 2^21 masks.
- ``pointing``: the pointing game recognition for CLIP (ViT-B/32) on ImageNet-1k.

Usage::

    python paper.py <the folder "figures" of the paper's sources>

It needs PyMuPDF (``pip install pymupdf``).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pymupdf
from pack import read_figures, write_figures

# the methods of the box plots from top to bottom, and of the curves by their color
BOXES = {
    "correlation": (
        "faithfulness/plot_faithfulness_spearman_correlation_methods_0.7_rotated.pdf",
        0.7,
    ),
    "r2": ("faithfulness/plot_faithfulness_r2_methods_0.5_rotated.pdf", 0.5),
}
METHODS = {
    "correlation": ["GAME", "Grad-ECLIP", "exCLIP", "Shapley values", "Banzhaf values"],
    "r2": ["Shapley values", "Banzhaf values"],
}
FIXLIP = ["FIxLIP (SI)", "FIxLIP (p = 0.3)", "FIxLIP (p = 0.5)", "FIxLIP (p = 0.7)"]
CURVES = "insertion_deletion_openai-clip-vit-base-patch32.pdf"
COLORS = {
    (0.92, 0.26, 0.21): "FIxLIP (p = 0.7)",
    (0.89, 0.45, 0.0): "FIxLIP (p = 0.5)",
    (0.98, 0.74, 0.02): "FIxLIP (p = 0.3)",
    (0.26, 0.52, 0.96): "Shapley values",
    (0.2, 0.66, 0.33): "Grad-ECLIP",
    (0.6, 0.63, 0.65): "GAME",
    (0.8, 0.53, 0.6): "exCLIP",
}
# the area between the curves and its standard deviation, as the legend of the figure states them
AID = {
    "FIxLIP (p = 0.7)": (0.94, 0.18),
    "FIxLIP (p = 0.5)": (1.02, 0.18),
    "FIxLIP (p = 0.3)": (0.95, 0.18),
    "Shapley values": (0.77, 0.15),
    "Grad-ECLIP": (0.37, 0.16),
    "GAME": (0.33, 0.16),
    "exCLIP": (0.25, 0.18),
}
RUNTIME = "time_google-siglip2-base-patch32-256.pdf"
ESTIMATORS = {(0.26, 0.52, 0.96): "model-agnostic", (0.92, 0.26, 0.21): "cross-modal"}
# the speedups that the figure notes at two budgets, for the whole explanation and for inference
SPEEDUPS = [[16, 5, 20], [18, 6, 45]]
# Table 1 of the paper, for texts that name one to four objects
POINTING = {
    "GAME": [0.61, 0.43, 0.33, 0.28],
    "Grad-ECLIP": [0.68, 0.45, 0.33, 0.28],
    "Shapley values": [0.70, 0.56, 0.46, 0.37],
    "Banzhaf values": [0.64, 0.52, 0.43, 0.35],
    "exCLIP": [0.73, 0.88, 0.89, 0.92],
    "FIxLIP (SI)": [0.83, 0.82, 0.84, 0.86],
    "FIxLIP (p = 0.3)": [0.78, 0.78, 0.80, 0.81],
    "FIxLIP (p = 0.5)": [0.81, 0.80, 0.81, 0.83],
    "FIxLIP (p = 0.7)": [0.83, 0.81, 0.83, 0.85],
}
GRAY, DARK = (0.8, 0.8, 0.8), (0.26, 0.26, 0.26)


def drawings(path: Path) -> list[dict[str, Any]]:
    """Return the paths of a figure, with their colors rounded."""
    paths = pymupdf.open(path)[0].get_drawings()
    for drawing in paths:
        for key in ("color", "fill"):
            drawing[key] = tuple(round(c, 2) for c in drawing[key] or ())
    return paths


def lines(drawing: dict[str, Any]) -> list[tuple[float, float, float, float]]:
    """Return the straight segments of a path as (x1, y1, x2, y2)."""
    return [(i[1].x, i[1].y, i[2].x, i[2].y) for i in drawing["items"] if i[0] == "l"]


def boxes(path: Path, methods: list[str]) -> list[dict[str, Any]]:
    """Return the five numbers of every box of a plot of horizontal boxes.

    A number outside the axis of the figure is ``None``, since the figure does not hold it.
    """
    paths = drawings(path)
    # the grid lines at 0, 0.2, ..., 1
    grid = sorted(
        {
            round(x1, 2)
            for d in paths
            if d["color"] == GRAY
            for x1, y1, x2, y2 in lines(d)
            if x1 == x2
        }
    )
    scale = lambda x: (x - grid[0]) / (grid[-1] - grid[0])  # noqa: E731
    strokes = [line for d in paths if d["color"] == DARK and d["type"] == "s" for line in lines(d)]
    filled = [
        d["rect"]
        for d in paths
        if d["fill"] not in ((), (1, 1, 1), (0.5, 0.5, 0.5), (0.15, 0.15, 0.15))
    ]
    filled = sorted((r for r in filled if r.height < 20), key=lambda r: r.y0)
    out = []
    for name, box in zip(methods, filled, strict=True):
        middle = (box.y0 + box.y1) / 2
        whiskers = [s for s in strokes if s[1] == s[3] and abs(s[1] - middle) < 1]
        ends = [x for x1, _, x2, _ in whiskers for x in (x1, x2)]
        median = [s[0] for s in strokes if s[0] == s[2] and abs(min(s[1], s[3]) - box.y0) < 0.1]
        numbers = {
            "low": min(ends),
            "q1": box.x0,
            "median": median[0] if median else None,
            "q3": box.x1,
            "high": max(ends),
        }
        # the figure cuts what lies left of its axis
        out.append(
            {"name": name}
            | {k: None if x is None or x < 0 else round(scale(x), 3) for k, x in numbers.items()}
        )
    return out


def curves(path: Path) -> list[dict[str, Any]]:
    """Return the insertion and the deletion curve of every method, from all tokens to none."""
    paths = drawings(path)
    # the ticks of the y-axis at -0.5, -0.25, ..., 1
    ticks = sorted(
        {
            round(y1, 2)
            for d in paths
            for x1, y1, x2, y2 in lines(d)
            if y1 == y2 and abs(x1 - x2) == 3.5
        }
    )
    scale = lambda y: 1 - 1.5 * (y - ticks[0]) / (ticks[-1] - ticks[0])  # noqa: E731
    out: dict[str, dict[str, Any]] = {}
    for d in paths:
        if d["type"] != "s" or d["color"] not in COLORS or len(d["items"]) != 50:
            continue
        segments = lines(d)
        ys = [segments[0][1]] + [s[3] for s in segments]
        kind = "insertion" if d["dashes"].startswith("[ 8") else "deletion"
        out.setdefault(COLORS[d["color"]], {})[kind] = [round(scale(y), 3) for y in ys]
    return [{"name": name, "aid": AID[name][0], "sd": AID[name][1]} | out[name] for name in AID]


def runtime(path: Path) -> dict[str, Any]:
    """Return the seconds of both estimators for every budget."""
    paths = drawings(path)
    out: dict[str, Any] = {"budgets": list(range(15, 22))}
    for d in paths:
        if d["type"] != "s" or d["color"] not in ESTIMATORS or len(d["items"]) != 7:
            continue
        ys = [s[3] for s in lines(d)]  # the first point, 2^14, lies left of the axis
        # the ticks of the y-axis: 1 second at 133.73 pt and 10,000 seconds at 1.65 pt
        seconds = [10 ** (4 * (133.73 - y) / (133.73 - 1.65)) for y in ys]
        kind = "total" if d["dashes"].startswith("[ 8") else "inference"
        out.setdefault(ESTIMATORS[d["color"]], {})[kind] = [float(f"{s:.3g}") for s in seconds]
    out["speedups"] = SPEEDUPS
    return out


def main() -> None:
    """Read the plots and write them to ``figures-data.js``."""
    folder = Path(sys.argv[1])
    paper: dict[str, Any] = {}
    for key, (file, p) in BOXES.items():
        paper[key] = {"p": p, "methods": boxes(folder / file, METHODS[key] + FIXLIP)}
    paper["curves"] = curves(folder / CURVES)
    paper["runtime"] = runtime(folder / RUNTIME)
    paper["pointing"] = POINTING
    for method in paper["curves"]:
        area = float(np.mean(method["insertion"]) - np.mean(method["deletion"]))
        print(f"{method['name']}: area between the curves {area:.2f}, in the paper {method['aid']}")
    if "--dry" in sys.argv:
        print(paper)
        return
    figures = read_figures()
    figures["paper"] = paper
    write_figures(figures)


if __name__ == "__main__":
    main()
