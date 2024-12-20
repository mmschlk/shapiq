"""This test module contains all tests for the sentence plot."""

import matplotlib.pyplot as plt
import numpy as np

from shapiq.interaction_values import InteractionValues
from shapiq.plot import sentence_plot


def _text_values() -> tuple[list[str], InteractionValues]:
    words = ["I", "really", "enjoy", "working", "with", "Shapley", "values", "in", "Python", "!"]
    values = [0.45, 0.01, 0.67, -0.2, -0.05, 0.7, 0.1, -0.04, 0.56, 0.7]
    iv = InteractionValues(
        n_players=10,
        values=np.array(values),
        index="SV",
        min_order=1,
        max_order=1,
        estimated=False,
        baseline_value=0.0,
    )
    return words, iv


def test_sentence_plot():
    """Test the sentence plot function."""
    words, iv = _text_values()

    fig, ax = sentence_plot(iv, words, show=False)
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    plt.close(fig)

    fig, ax = iv.plot_sentence(
        words,
        show=False,
        connected_words=[("Shapley", "values")],
        max_score=0.5,  # max_score is intentionally lower than the attributions here
    )
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    plt.close(fig)

    output = sentence_plot(
        iv,
        words,
        chars_per_line=100,
        show=True,
        connected_words=[("Shapley", "values")],
        max_score=1.0,
    )
    assert output is None
    plt.close("all")
