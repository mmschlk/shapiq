"""This utility module contains helper functions for plotting."""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable
from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from PIL.Image import Image

__all__ = ["abbreviate_feature_names", "add_image_in_center", "format_labels", "format_value"]


def format_value(
    s: float | str,
    format_str: str = "%.2f",
) -> str:
    """Strips trailing zeros and uses a unicode minus sign.

    Args:
        s: The value to be formatted.
        format_str: The format string to be used. Defaults to "%.2f".

    Returns:
        str: The formatted value.

    Examples:
        >>> format_value(1.0)
        "1"
        >>> format_value(1.234)
        "1.23"

    """
    if not issubclass(type(s), str):
        s = format_str % s
    s = re.sub(r"\.?0+$", "", s)
    if s[0] == "-":
        s = "\u2212" + s[1:]
    return str(s)


def format_labels(
    feature_mapping: dict[int, str],
    feature_tuple: tuple[int, ...],
) -> str:
    """Formats the feature labels for the plots.

    Args:
        feature_mapping: A dictionary mapping feature indices to feature names.
        feature_tuple: The feature tuple to be formatted.

    Returns:
        str: The formatted feature tuple.

    Example:
        >>> feature_mapping = {0: "A", 1: "B", 2: "C"}
        >>> format_labels(feature_mapping, (0, 1))
        "A x B"
        >>> format_labels(feature_mapping, (0,))
        "A"
        >>> format_labels(feature_mapping, ())
        "Base Value"

    """
    if len(feature_tuple) == 0:
        return "Base Value"
    if len(feature_tuple) == 1:
        return str(feature_mapping[feature_tuple[0]])
    return " x ".join([str(feature_mapping[f]) for f in feature_tuple])


def abbreviate_feature_names(feature_names: Iterable[str]) -> list[str]:
    """A rudimentary function to abbreviate feature names for plotting.

    Args:
        feature_names: The feature names to be abbreviated.

    Returns:
        list[str]: The abbreviated feature names.

    """
    abbreviated_names = []
    for _name in feature_names:
        name = str(_name)
        name = name.strip()
        capital_letters = sum(1 for c in name if c.isupper())
        seperator_chars = (" ", "_", "-", ".")
        is_seperator_in_name = any(c in seperator_chars for c in name[:-1])
        if is_seperator_in_name:
            for seperator in seperator_chars:
                name = name.replace(seperator, ".")
            name_parts = name.split(".")
            new_name = ""
            for part in name_parts:
                if part:
                    new_name += part[0].upper()
            abbreviated_names.append(new_name)
        elif capital_letters > 1:
            new_name = "".join([c for c in name if c.isupper()])
            abbreviated_names.append(new_name[0:3])
        else:
            abbreviated_names.append(name.strip()[0:3] + ".")
    return abbreviated_names


def add_image_in_center(
    axis: Axes,
    image: Image | np.ndarray,
    size: float = 0.4,
    n_features: int | None = None,
) -> None:
    """Adds an image in the center of the plot.

    Args:
        axis: The matplotlib axis to add the image to.

        image: The image to be added.

        size: The size of the image in the plot. Defaults to ``0.4``.

        n_features: The number of features in the plot. If provided, the image is divided into a
            grid containing n_features patches. Defaults to ``None``.
    """
    from PIL import Image

    # add the image in the center of the plot
    image_to_plot = Image.fromarray(np.asarray(deepcopy(image)))
    axis.imshow(image_to_plot, extent=(-size, size, -size, size), zorder=1e10)

    if n_features is None:
        return

    # add grid lines
    x = np.linspace(-size, size, int(math.sqrt(n_features) + 1))
    y = np.linspace(-size, size, int(math.sqrt(n_features) + 1))
    axis.vlines(
        x=x, ymin=-size, ymax=size, colors="white", linewidths=2, linestyles="solid", zorder=2e10
    )
    axis.hlines(
        y=y, xmin=-size, xmax=size, colors="white", linewidths=2, linestyles="solid", zorder=2e10
    )
