"""A collection of differently sized interaction values usefull for testing purposes."""

import copy
from typing import Literal

import pytest

from shapiq.interaction_values import InteractionValues


def _get_interaction_value(
    n_players: int,
    n_interactions: int | Literal["all"],
    min_order: int,
    max_order: int,
) -> InteractionValues:
    """Create an InteractionValue with the given parameters.

    Args:
        n_players (int): The number of players.
        n_interactions (int): The number of interactions.
        min_order (int): The minimum order of the interaction values.
        max_order (int): The maximum order of the interaction values.
    """


_iv_n_players_300_0_300 = _get_interaction_value(
    n_players=300,
    n_interactions=100,
    min_order=0,
    max_order=300,
)


@pytest.fixture
def iv_n_players_300():
    """Return an InteractionValue with 300 players (but only a fraction of them are non-zero)."""
    return copy.deepcopy(_iv_n_players_300_0_300)
