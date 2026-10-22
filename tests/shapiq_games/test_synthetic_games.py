"""Tests for the synthetic games and their analytic structure."""

from __future__ import annotations

import numpy as np
import pytest

from shapiq.utils import powerset
from shapiq_games.synthetic import SOUM, DummyGame, RandomTableGame, UnanimityGame


def _all_coalitions(n: int) -> np.ndarray:
    return np.array([[i in s for i in range(n)] for s in powerset(range(n))], dtype=bool)


def test_dummy_game_values_and_counter() -> None:
    game = DummyGame(4, interaction=(1, 2))
    coalitions = np.array([[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 1, 0], [1, 1, 1, 1]], dtype=bool)
    np.testing.assert_allclose(game(coalitions), [0.0, 0.25, 1.5, 2.0])
    assert game.access_counter == 4


def test_unanimity_game() -> None:
    game = UnanimityGame(np.array([0, 1, 0, 1]))
    assert game.interaction == (1, 3)
    coalitions = np.array([[0, 0, 0, 0], [0, 1, 0, 0], [1, 1, 0, 1], [1, 1, 1, 1]], dtype=bool)
    np.testing.assert_array_equal(game(coalitions), [0.0, 0.0, 1.0, 1.0])


def test_soum_moebius_coefficients_reconstruct_the_game() -> None:
    game = SOUM(6, n_basis_games=12, max_interaction_size=4, random_state=5)
    moebius = game.moebius_coefficients
    coalitions = _all_coalitions(6)
    values = game(coalitions)
    for coalition, value in zip(coalitions, values, strict=True):
        members = set(np.flatnonzero(coalition))
        reconstructed = sum(
            moebius[interaction]
            for interaction in moebius.interaction_lookup
            if set(interaction) <= members
        )
        assert value == pytest.approx(reconstructed)


def test_soum_is_seeded_and_normalizable() -> None:
    np.testing.assert_array_equal(
        SOUM(5, 8, random_state=1).linear_coefficients,
        SOUM(5, 8, random_state=1).linear_coefficients,
    )
    assert not np.array_equal(
        SOUM(5, 8, random_state=1).linear_coefficients,
        SOUM(5, 8, random_state=2).linear_coefficients,
    )
    np.testing.assert_array_equal(SOUM(5, 8).linear_coefficients, SOUM(5, 8).linear_coefficients)
    game = SOUM(5, 8, min_interaction_size=0, random_state=0, normalize=True)
    assert game(game.empty_coalition)[0] == pytest.approx(0.0)


def test_random_table_game_is_a_lookup_table() -> None:
    game = RandomTableGame(4, low=-1.0, high=1.0, random_state=0)
    coalitions = _all_coalitions(4)
    values = game(coalitions)
    assert np.all((values >= -1.0) & (values < 1.0))
    # player i contributes 2**i to the index of a coalition
    indices = coalitions.astype(int) @ (2 ** np.arange(4))
    np.testing.assert_array_equal(values, game.values[indices])
    with pytest.raises(ValueError, match="supports 1 to 20 players"):
        RandomTableGame(21)
