"""This module contains the configuration for the shapiq visualizations."""

from colour import Color

__all__ = [
    "RED",
    "BLUE",
    "NEUTRAL",
    "LINES",
    "COLORS_N_SII",
]

RED = Color("#ff0d57")
BLUE = Color("#1e88e5")
NEUTRAL = Color("#ffffff")
LINES = Color("#cccccc")

COLORS_N_SII = [
    "#D81B60",
    "#FFB000",
    "#1E88E5",
    "#FE6100",
    "#7F975F",
    "#74ced2",
    "#708090",
    "#9966CC",
    "#CCCCCC",
    "#800080",
]
COLORS_N_SII = COLORS_N_SII * (100 + (len(COLORS_N_SII)))  # repeat the colors list
