"""This test module contains all tests regarding sentiment classifier benchmark game."""

import numpy as np
import pytest

from shapiq.games import SentimentClassificationGame


@pytest.mark.slow
@pytest.mark.parametrize("mask_strategy", ["remove", "mask"])
def test_basic_function(mask_strategy):
    """Tests the SentimentClassificationGame with a small input text."""
    input_text = "this is a six word sentence"
    n_players = 6
    game = SentimentClassificationGame(
        input_text=input_text, normalize=True, mask_strategy=mask_strategy
    )

    assert game.n_players == n_players
    assert game.original_input_text == input_text
    assert game.original_model_output == game._full_output

    assert game.normalization_value == game._empty_output
    assert game.normalize  # should be normalized

    # test value function
    test_coalition = np.array([[0, 0, 0, 0, 0, 0]], dtype=bool)
    assert game.value_function(test_coalition) == game._empty_output
    assert game(test_coalition) == game._empty_output - game.normalization_value

    test_coalition = np.array([[1, 1, 1, 1, 1, 1]], dtype=bool)
    assert game.value_function(test_coalition) == game._full_output
    assert game(test_coalition) == game._full_output - game.normalization_value

    test_coalition = np.array([[1, 0, 1, 0, 1, 0]], dtype=bool)
    assert game.value_function(test_coalition) == game(test_coalition) + game.normalization_value

    # test ValueError with wrong param
    with pytest.raises(ValueError):
        _ = SentimentClassificationGame(
            input_text=input_text, normalize=True, mask_strategy="undefined"
        )
