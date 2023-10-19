"""This module contains the configuration for the shapiq visualizations."""
from colour import Color

RED = Color("#ff0d57")
BLUE = Color("#1e88e5")
NEUTRAL = Color("#ffffff")

__all__ = [
    "RED",
    "BLUE",
    "NEUTRAL",
]


if __name__ == "__main__":

    red = [round(c * 255, 0) for c in RED.rgb]
    blue = [round(c * 255, 0) for c in BLUE.rgb]
    neutral = [round(c * 255, 0) for c in NEUTRAL.rgb]

    print("RED", red)
    print("BLUE", blue)
    print("NEUTRAL", neutral)