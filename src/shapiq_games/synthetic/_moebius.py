"""The Möbius representation of the synthetic games, as sparse interaction values."""

from __future__ import annotations

import numpy as np

from shapiq.interaction_values import InteractionValues

__all__ = ["moebius_values"]


def moebius_values(coefficients: dict[tuple[int, ...], float], n_players: int) -> InteractionValues:
    """Return non-zero Möbius coefficients as interaction values of the index ``"Moebius"``.

    Args:
        coefficients: The coefficient of every interaction (a sorted tuple of players).
        n_players: The number of players.

    Returns:
        The coefficients; the coefficient of the empty set, if any, is also the baseline value.
    """
    lookup = {interaction: i for i, interaction in enumerate(coefficients)}
    return InteractionValues(
        values=np.array(list(coefficients.values()), dtype=float),
        index="Moebius",
        max_order=n_players,
        min_order=0,
        n_players=n_players,
        interaction_lookup=lookup,
        estimated=False,
        baseline_value=coefficients.get((), 0.0),
    )
