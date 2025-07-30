"""This module contains fixtures for the tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from shapiq.imputer.marginal_imputer import MarginalImputer

if TYPE_CHECKING:
    from shapiq.game import Game


@pytest.fixture
def cooking_game() -> Game:
    """Return the cooking game object."""
    from shapiq.game import Game

    class CookingGame(Game):
        def __init__(self):
            self.characteristic_function = {
                (): 10,
                (0,): 4,
                (1,): 3,
                (2,): 2,
                (0, 1): 9,
                (0, 2): 8,
                (1, 2): 7,
                (0, 1, 2): 15,
            }
            super().__init__(
                n_players=3,
                player_names=["Alice", "Bob", "Charlie"],  # Optional list of names
                normalization_value=self.characteristic_function[()],  # 0
                normalize=False,
            )

        def value_function(self, coalitions: np.ndarray) -> np.ndarray:
            """Defines the worth of a coalition as a lookup in the characteristic function."""
            output = [
                self.characteristic_function[tuple(np.where(coalition)[0])]
                for coalition in coalitions
            ]
            return np.array(output)

    return CookingGame()


@pytest.fixture
def paper_game():
    """Return a simple game object."""
    from scipy.special import binom

    from shapiq.game import Game

    class PaperGame(Game):
        """A simple game with 11 players.

        A game with 11 players, where each coalition must contain at least 2 players and with
        probability 0.1 of two players not cooperating.
        """

        def __init__(self):
            super().__init__(
                n_players=11,
                player_names=None,  # Optional list of names
                normalization_value=0,  # 0
            )

        def value_function(self, coalitions: np.ndarray) -> np.ndarray:
            """Defines the worth of a coalition as a lookup in the characteristic function."""
            output = [
                sum(coalition) - 0.1 * binom(sum(coalition), 2) if sum(coalition) > 1 else 0
                for coalition in coalitions
            ]
            return np.array(output)

    return PaperGame()


@pytest.fixture
def cooking_game_pre_computed(cooking_game) -> Game:
    game = cooking_game
    game.precompute()
    return game


def get_california_housing_imputer() -> MarginalImputer:
    """Return a California housing imputer."""
    from .data import get_california_housing_train_test_explain
    from .models import get_california_housing_random_forest

    _, _, x_test, _, x_explain = get_california_housing_train_test_explain()
    model = get_california_housing_random_forest()

    imputer = MarginalImputer(
        model=model.predict,
        data=x_test,
        x=x_explain,
        random_state=42,
        normalize=False,
        sample_size=100,
        joint_marginal_distribution=True,
    )
    imputer_hash = hash(
        (
            imputer.sample_size,
            imputer.joint_marginal_distribution,
            imputer.normalize,
            imputer.random_state,
        )
    )
    assert imputer_hash == 9070456741283270540
    return imputer


@pytest.fixture
def california_housing_imputer():
    """Return a California housing imputer fixture."""
    return get_california_housing_imputer()
