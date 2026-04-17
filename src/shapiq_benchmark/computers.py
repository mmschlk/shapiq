"""Benchmark ground truth computers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from shapiq import ExactComputer
from shapiq import TreeExplainer
from shapiq.tree.interventional.explainer import InterventionalTreeExplainer
from shapiq.typing import IndexType

from .base import GroundTruthComputer

if TYPE_CHECKING:
    from shapiq import Game, InteractionValues

class InterventionalComputer(GroundTruthComputer[IndexType]):
    """Exact computer for interventional games using the InterventionalTreeExplainer."""

    def __init__(self, game: Game) -> None:
        self.game=game

    def exact_values(self, index: IndexType, order: int) -> InteractionValues:
        explainer = InterventionalTreeExplainer(
            model=self.game.model,
            index=index,
            max_order=order,
            data=self.game.data,
            debug=False,
            class_index=self.game.class_index,
        )
        return explainer.explain_function(x=self.game.target_instance[0])


class PathdependentComputer(GroundTruthComputer[IndexType]):
    """Exact computer for tree-based games using the TreeExplainer."""

    def __init__(self, game: Game) -> None:
        self.game=game

    def exact_values(self, index: IndexType, order: int) -> InteractionValues:
        explainer = TreeExplainer(
            model=self.game.model,
            index=index,
            max_order=order,
            class_index=self.game.class_label,
        )
        return explainer.explain_function(x=self.game.x_explain)


class LocalXAIComputer(GroundTruthComputer[IndexType]):
    """Exact computer for local explanation games using the ExactComputer."""

    def __init__(self, game: Game) -> None:
        self._computer = ExactComputer(
            game=game, n_players=game.n_players, evaluate_game=False
        )

    def exact_values(self, index: IndexType, order: int) -> InteractionValues:
        return self._computer(index=index, order=order)
