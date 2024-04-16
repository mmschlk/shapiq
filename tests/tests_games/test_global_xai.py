"""This test module contains all tests regarding the GlobalExplanation game."""

import numpy as np


def loss_function(y_pred, y_test):
    return np.mean(np.square(y_pred - y_test))


def test_basic_function(background_reg_dataset):
    """Tests the GlobalExplanation game with a small regression and classification dataset."""
    raise NotImplementedError("The game is not implemented yet.")


def test_california():
    """Test the GlobalExplanation game with the california housing dataset."""
    raise NotImplementedError("The game is not implemented yet.")


def test_adult_census():
    """Test the GlobalExplanation game with the adult census dataset."""
    raise NotImplementedError("The game is not implemented yet.")


def test_bike_sharing():
    """Test the GlobalExplanation game with the bike sharing dataset."""
    raise NotImplementedError("The game is not implemented yet.")
