"""This module contains the sentence plot."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch, PathPatch
from matplotlib.textpath import TextPath

from ._config import BLUE, RED

if TYPE_CHECKING:
    from collections.abc import Sequence

    from shapiq.interaction_values import InteractionValues


def _get_color_and_alpha(max_value: float, value: float) -> tuple[str, float]:
    """Gets the color and alpha value for an interaction value."""
    color = RED.hex if value >= 0 else BLUE.hex
    ratio = abs(value / max_value)
    ratio = min(ratio, 1.0)  # make ratio at most 1
    return color, ratio


def sentence_plot(
    interaction_values: InteractionValues,
    words: Sequence[str],
    *,
    connected_words: Sequence[tuple[str, str]] | None = None,
    chars_per_line: int = 35,
    font_family: str = "sans-serif",
    show: bool = False,
    max_score: float | None = None,
) -> tuple[plt.Figure, plt.Axes] | None:
    """Plots the first order effects (attributions) of a sentence or paragraph.

    An example of the plot is shown below.

    .. image:: /_static/sentence_plot_example.png
        :width: 400
        :align: center

    Args:
        interaction_values: The interaction values as an interaction object.
        words: The words of the sentence or a paragraph of text.
        connected_words: A list of tuples with connected words. Defaults to ``None``. If two 'words'
            are connected, the plot will not add a space between them (e.g., the parts "enjoy" and
            "able" would be connected to "enjoyable" with potentially different attributions for
            each part).
        chars_per_line: The maximum number of characters per line. Defaults to ``35`` after which
            the text will be wrapped to the next line. Connected words receive a '-' in front of
            them.
        font_family: The font family used for the plot. Defaults to ``sans-serif``. For a list of
            available font families, see the matplotlib documentation of
            ``matplotlib.font_manager.FontProperties``. Note the plot is optimized for sans-serif.
        max_score: The maximum score for the attributions to scale the colors and alpha values. This
            is useful if you want to compare the attributions of different sentences and both plots
            should have the same color scale. Defaults to ``None``.
        show: Whether to show the plot. Defaults to ``False``.

    Returns:
        If ``show`` is ``True``, the function returns ``None``. Otherwise, it returns a tuple with
        the figure and the axis of the plot.

    Example:
        >>> import numpy as np
        >>> from shapiq.plot import sentence_plot
        >>> iv = InteractionValues(
        ...    values=np.array([0.45, 0.01, 0.67, -0.2, -0.05, 0.7, 0.1, -0.04, 0.56, 0.7]),
        ...    index="SV",
        ...    n_players=10,
        ...    min_order=1,
        ...    max_order=1,
        ...    estimated=False,
        ...    baseline_value=0.0,
        ... )
        >>> words = ["I", "really", "enjoy", "working", "with", "Shapley", "values", "in", "Python", "!"]
        >>> connected_words = [("Shapley", "values")]
        >>> fig, ax = sentence_plot(iv, words, connected_words, show=False, chars_per_line=100)
        >>> plt.show()

    .. image:: /_static/sentence_plot_connected_example.png
        :width: 300
        :align: center

    """
    # set all the size parameters
    fontsize = 20
    word_spacing = 15
    line_spacing = 10
    height_padding = 5
    width_padding = 5

    # clean the input
    connected_words = [] if connected_words is None else connected_words
    words = [word.strip() for word in words]
    attributions = [interaction_values[(i,)] for i in range(len(words))]

    # get the maximum score
    max_abs_attribution = max_score
    if max_score is None:
        max_abs_attribution = max([abs(value) for value in attributions])

    # create plot
    fig, ax = plt.subplots()

    max_x_pos = 0
    x_pos, y_pos = word_spacing, 0
    lines, chars_in_line = 0, 0
    for i, (_word, attribution) in enumerate(zip(words, attributions, strict=False)):
        word = _word
        # check if the word is connected
        is_word_connected_first = False
        is_word_connected_second = (words[i - 1], word) in connected_words
        with contextlib.suppress(IndexError):
            is_word_connected_first = (word, words[i + 1]) in connected_words

        # check if the line is too long and needs to be wrapped
        chars_in_line += len(word)
        if chars_in_line > chars_per_line:
            lines += 1
            chars_in_line = 0
            x_pos = word_spacing
            y_pos -= fontsize + line_spacing
            if is_word_connected_second:
                word = "-" + word

        # adjust the x position for connected words
        if is_word_connected_second:
            x_pos += 2

        # set the position of the word in the plot
        position = (x_pos, y_pos)

        # get the color and alpha value
        color, alpha = _get_color_and_alpha(max_abs_attribution, attribution)

        # get the text
        text_color = "black" if alpha < 2 / 3 else "white"
        fp = FontProperties(family=font_family, style="normal", size=fontsize, weight="normal")
        text_path = TextPath(position, word, prop=fp)
        text_path = PathPatch(text_path, facecolor=text_color, edgecolor="none")
        width_of_text = text_path.get_window_extent().width

        # get dimensions for the explanation patch
        height_patch = fontsize + height_padding
        width_patch = width_of_text + 1
        y_pos_patch = y_pos - height_padding
        x_pos_patch = x_pos + 1
        if is_word_connected_first:
            x_pos_patch -= width_padding / 2
            width_patch += width_padding / 2
        elif is_word_connected_second:
            width_patch += width_padding / 2
        else:
            x_pos_patch -= width_padding / 2
            width_patch += width_padding

        # create the explanation patch
        patch = FancyBboxPatch(
            xy=(x_pos_patch, y_pos_patch),
            width=width_patch,
            height=height_patch,
            color=color,
            alpha=alpha,
            zorder=-1,
            boxstyle="Round, pad=0, rounding_size=3",
        )

        # draw elements for the word
        ax.add_patch(patch)
        ax.add_artist(text_path)

        # update the x position
        x_pos += width_of_text + word_spacing
        max_x_pos = max(max_x_pos, x_pos)
        if is_word_connected_first:
            x_pos -= word_spacing

    # fix up the dimensions of the plot
    ax.set_xlim(0, max_x_pos)
    ax.set_ylim(y_pos - fontsize / 2, fontsize + fontsize / 2)
    width = max_x_pos
    height = fontsize + fontsize / 2 + abs(y_pos - fontsize / 2)
    fig.set_size_inches(width / 100, height / 100)

    # clean up the plot
    ax.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # draw the plot
    if not show:
        return fig, ax
    plt.show()
    return None
