"""This module contains the base GraphSHAP-IQ xai game."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

from shapiq.game import Game


class GraphSHAPIQXAI(Game):
    """A GraphSHAP-IQ explanation game for graph networks.

    The game is based on the GraphSHAP-IQ algorithm and is used to explain the predictions of graph
    networks. GraphSHAP-IQ is used to compute Shapley interaction values for graph networks.
    """

    def __init__(self) -> None:
        """Docstring."""
        # TO DO

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        """Docstring."""
        # TO DO
        return coalitions
