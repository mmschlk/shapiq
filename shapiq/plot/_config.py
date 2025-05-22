"""This module contains the configuration for the shapiq visualizations."""

from __future__ import annotations

from colour import Color

__all__ = ["BLUE", "COLORS_K_SII", "LINES", "NEUTRAL", "RED", "get_color"]

RED = Color("#ff0d57")
BLUE = Color("#1e88e5")
NEUTRAL = Color("#ffffff")
LINES = Color("#cccccc")

COLORS_K_SII = [
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
COLORS_K_SII = COLORS_K_SII * (100 + (len(COLORS_K_SII)))  # repeat the colors list


def get_color(value: float) -> str:
    """Returns blue color for negative values and red color for positive values.

    Args:
        value (float): The value to determine the color for.

    Returns:
        str: The color as a hex string.

    """
    if value >= 0:
        return RED.hex
    return BLUE.hex
