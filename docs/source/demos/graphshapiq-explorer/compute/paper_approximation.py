"""Read Figure 4 (left) of the GraphSHAP-IQ paper from its vector graphic, for ``paper-approximation``.

The figure "2-SII for Mutagenicity with 2-layer GIN" (Muschalik et al., ICLR 2025, page 9) shows the
MSE of 2-SII estimates against the number of model evaluations, for GraphSHAP-IQ and five
model-agnostic baselines, each as a baseline (solid) and as its interaction-informed variant
(dashed), averaged over 10 molecules of Mutagenicity with 30 to 40 atoms, with a band (the
standard error of the mean). In the PDF every curve is a path with exact coordinates. This script
calibrates the axes on their tick marks (x: 0, 1000, ..., 8000, linear; y: 10^-9, 10^-7, ...,
10^-1, logarithmic), and reads the curves and the bands by their colors (matplotlib's colors of the
paper's legend) and their dashes.

A band is the mean plus and minus the standard error, so its middle is the curve (checked: equal to
0.05% wherever both are drawn). KernelSHAP-IQ's first point (about 300 evaluations) lies far above
the plot, and the PDF holds its line only from where it enters the plot (matplotlib clips it). Its
band is not clipped, so the script takes that point as the middle of the band (an MSE of about 17).

Writes ``parts/paper-approximation.json``::

    {"x": [min, max], "y": [min, max] (the axes of the paper), "series": [
        {"method", "variant": "GraphSHAP-IQ" | "baseline" | "interaction-informed",
         "points": [[evaluations, MSE], ...], "band": [[evaluations, low, high], ...] | null}]}

Usage (needs PyMuPDF, ``pip install pymupdf``)::

    python paper_approximation.py <the paper's PDF>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pymupdf

HERE = Path(__file__).resolve().parent
OUT = HERE / "parts" / "paper-approximation.json"
PAGE = 8  # page 9
PLOT = pymupdf.Rect(118, 60, 240, 210)  # around the left panel's plot (and a band that leaves it), in PDF points
COLORS = {  # matplotlib's colors of the figure, by the paper's legend
    (0.49, 0.808, 0.51): "GraphSHAP-IQ",
    (1.0, 0.435, 0.0): "KernelSHAP-IQ",
    (1.0, 0.729, 0.031): "Inc. KernelSHAP-IQ",
    (0.49, 0.325, 0.871): "Permutation Sampling",
    (0.937, 0.153, 0.651): "SHAP-IQ",
    (0.0, 0.706, 0.847): "SVARM-IQ",
}


def key(color: tuple[float, ...] | None) -> tuple[float, ...] | None:
    """Return a color rounded as in COLORS."""
    return tuple(round(c, 3) for c in color) if color else None


def main() -> None:
    """Read the figure and write the part."""
    page = pymupdf.open(sys.argv[1])[PAGE]
    drawings = [d for d in page.get_drawings() if PLOT.contains(d["rect"])]
    black = [d for d in drawings if key(d.get("color")) == (0.0, 0.0, 0.0) and len(d["items"]) == 1]
    # the tick marks: short black strokes below the x axis and left of the y axis
    xticks = sorted(d["items"][0][1].x for d in black if abs(d["items"][0][1].x - d["items"][0][2].x) < 1e-6 and d["rect"].height < 1.5)
    yticks = sorted((d["items"][0][1].y for d in black if abs(d["items"][0][1].y - d["items"][0][2].y) < 1e-6 and d["rect"].width < 1.5), reverse=True)
    if len(xticks) != 9 or len(yticks) != 5:
        msg = f"expected 9 x ticks and 5 y ticks, found {len(xticks)} and {len(yticks)}"
        raise ValueError(msg)
    fx = np.polyfit(xticks, np.arange(9) * 1000, 1)  # pixels to evaluations
    fy = np.polyfit(yticks, [-9, -7, -5, -3, -1], 1)  # pixels to log10(MSE)
    residual = max(np.abs(np.polyval(fy, yticks) - [-9, -7, -5, -3, -1]).max(), np.abs(np.polyval(fx, xticks) - np.arange(9) * 1000).max() / 1000)
    x = lambda px: float(np.polyval(fx, px))  # noqa: E731
    y = lambda py: float(10 ** np.polyval(fy, py))  # noqa: E731
    spines = [d for d in black if d["type"] == "s"]
    left = min(d["rect"].x0 for d in spines)
    bottom = max(d["rect"].y1 for d in spines)
    right = max(d["rect"].x1 for d in spines)
    top = min(d["rect"].y0 for d in spines)

    lines, bands = [], []
    for d in drawings:
        name = COLORS.get(key(d.get("color"))) or COLORS.get(key(d.get("fill")))
        if name is None:
            continue
        if d["type"] == "s" and len(d["items"]) > 2:  # a curve
            points = [d["items"][0][1]] + [item[-1] for item in d["items"]]
            dashed = d.get("dashes") not in (None, "[] 0")
            lines.append({"method": name, "dashed": dashed, "points": [[x(p.x), y(p.y)] for p in points]})
        elif d["type"] == "fs" and len(d["items"]) > 8 and (d.get("fill_opacity") or 1) < 0.5:  # a band
            points = [d["items"][0][1]] + [item[-1] for item in d["items"]]
            # the upper edge's first point, along the lower edge, then back along the upper one
            count = (len(points) - 1) // 2
            lower = points[1 : count + 1]
            upper = points[count + 1 : 2 * count + 1][::-1]
            bands.append({"method": name, "band": [[x(a.x), y(a.y), y(b.y)] for a, b in zip(lower, upper, strict=True)]})
    series = []
    for line in lines:
        variant = "GraphSHAP-IQ" if line["method"] == "GraphSHAP-IQ" else "interaction-informed" if line["dashed"] else "baseline"
        series.append({"method": line["method"], "variant": variant, "points": line["points"], "band": None})
    # each band belongs to the curve of its color that runs through its middle
    for band in bands:
        def distance(s: dict, band: dict = band) -> float:
            at = {round(px): py for px, py in s["points"]}
            mids = [(np.log10(lo) + np.log10(hi)) / 2 - np.log10(at[round(bx)]) for bx, lo, hi in band["band"] if round(bx) in at]
            return float(np.mean(np.abs(mids))) if mids else np.inf
        candidates = [s for s in series if s["method"] == band["method"] and s["band"] is None]
        best = min(candidates, key=distance)
        best["band"] = band["band"]
    # a curve that starts where it enters the plot: its first point is the middle of its band
    for s in series:
        if s["band"] and abs(s["points"][0][0] - s["band"][0][0]) > 1:
            bx, lo, hi = s["band"][0]
            print(f"{s['method']} ({s['variant']}): first point {s['points'][0]} replaced by the band's middle")
            s["points"][0] = [bx, (lo + hi) / 2]
    middle = [
        abs((lo + hi) / 2 / py - 1)
        for s in series if s["band"]
        for (px, py), (bx, lo, hi) in zip(s["points"], s["band"], strict=True)
        if abs(px - bx) < 1
    ]
    print(f"largest gap between a curve and the middle of its band: {max(middle):.1%}")
    order = list(COLORS.values())
    series.sort(key=lambda s: (order.index(s["method"]), s["variant"] == "interaction-informed"))
    for s in series:
        s["points"] = [[round(px), float(f"{py:.4g}")] for px, py in s["points"]]
        if s["band"]:
            s["band"] = [[round(bx), float(f"{lo:.4g}"), float(f"{hi:.4g}")] for bx, lo, hi in s["band"]]
        last = s["points"][-1]
        print(f"{s['method']} ({s['variant']}): {len(s['points'])} points, last {last[0]} {last[1]:.2e}, band {'yes' if s['band'] else 'no'}")
    part = {
        "x": [round(x(left)), round(x(right))],
        "y": [float(f"{y(bottom):.4g}"), float(f"{y(top):.4g}")],
        "series": series,
    }
    print(f"axes: x {part['x']}, y {part['y']}, calibration residual {residual:.1e}")
    OUT.write_text(json.dumps(part, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
