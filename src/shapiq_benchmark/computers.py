"""Benchmark ground truth computers."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from shapiq import ExactComputer
from shapiq.tree.explainer import TreeExplainer
from shapiq.explainer.tabpfn import TabPFNExplainer
from shapiq.tree.interventional.explainer import InterventionalTreeExplainer
from shapiq.tree.interventional.game import InterventionalGame
from shapiq.typing import IndexType
from shapiq.explainer.custom_types import ExplainerIndices
from shapiq.imputer.tabpfn_imputer import TabPFNImputer
from shapiq_games.benchmark.local_xai.base import LocalExplanation
from shapiq_games.benchmark.local_xai.benchmark_image import ImageClassifier
from shapiq_games.benchmark.treeshapiq_xai.base import TreeSHAPIQXAI

from .base import GroundTruthComputer

if TYPE_CHECKING:
    from shapiq import Game, InteractionValues


class InterventionalComputer(GroundTruthComputer[IndexType]):
    """Exact computer for interventional games using the InterventionalTreeExplainer."""

    def __init__(self, game: InterventionalGame) -> None:
        self.game = game
        self._computer = InterventionalTreeExplainer(
            model=self.game.model,
            data=self.game.data,
            debug=False,
            class_index=self.game.class_index,
        )

    def exact_values(
        self, index: IndexType, order: int, budget: int | None = None
    ) -> InteractionValues:
        """Compute exact interaction values using the InterventionalTreeExplainer.
        Args:
            index: The index for which to compute interaction values.
            order: The order of interactions to compute.
            budget: Optional budget for computation (not used in this computer).
        Returns:
            InteractionValues: The computed interaction values.
        """
        self._computer.index = index
        self._computer.max_order = order
        return self._computer.explain_function(x=self.game.target_instance[0])


class PathdependentComputer(GroundTruthComputer[IndexType]):
    """Exact computer for tree-based games using the TreeExplainer."""

    def __init__(self, game: TreeSHAPIQXAI) -> None:
        self.game = game
        self._computer = TreeExplainer(
            model=self.game.model,
            class_index=self.game.class_label,
        )

    def exact_values(
        self, index: IndexType, order: int, budget: int | None = None
    ) -> InteractionValues:
        """Compute exact interaction values using the TreeExplainer.
        Args:
            index: The index for which to compute interaction values.
            order: The order of interactions to compute.
            budget: Optional budget for computation (not used in this computer).
        Returns:
            InteractionValues: The computed interaction values.
        """
        return self._computer.explain_function(
            x=self.game.x_explain,
            index=index,
            max_order=order,
        )


class LocalXAIComputer(GroundTruthComputer[IndexType]):
    """Exact computer for local explanation games using the ExactComputer."""

    def __init__(self, game: LocalExplanation) -> None:
        self.game = game
        self._computer = ExactComputer(
            game=game, n_players=game.n_players, evaluate_game=False
        )

    def exact_values(
        self, index: IndexType, order: int, budget: int | None = None
    ) -> InteractionValues:
        """Compute exact interaction values using the ExactComputer.
        Args:
            index: The index for which to compute interaction values.
            order: The order of interactions to compute.
            budget: Optional budget for computation (not used in this computer).
        Returns:
            InteractionValues: The computed interaction values.
        """
        return self._computer(index=index, order=order)


class TabPFNComputer(GroundTruthComputer[IndexType]):
    """Exact computer for TabPFN imputers using the TabPFNExplainer."""

    def __init__(self, game: TabPFNImputer) -> None:
        self.game = game

    def exact_values(
        self, index: IndexType, order: int, budget: int | None = None
    ) -> InteractionValues:
        """Compute exact interaction values using the TabPFNExplainer.
        Args:
            index: The index for which to compute interaction values.
            order: The order of interactions to compute.
            budget: Budget for computation.
        Returns:
            InteractionValues: The computed interaction values.
        """
        if budget is None:
            msg = "budget must be provided for TabPFN explanations."
            raise ValueError(msg)
        explainer_index = cast(ExplainerIndices, index)
        explainer = TabPFNExplainer(
            model=self.game.model,
            data=self.game.x_train,
            labels=self.game.y_train,
            index=explainer_index,
            max_order=order,
        )
        return explainer.explain(x=self.game.x, budget=budget)


class ImageComputer(GroundTruthComputer[IndexType]):
    """Exact computer for image classifier games using the ExactComputer."""

    def __init__(self, game: ImageClassifier) -> None:
        self._computer = ExactComputer(
            game=game, n_players=game.n_players, evaluate_game=False
        )

    def exact_values(
        self, index: IndexType, order: int, budget: int | None = None
    ) -> InteractionValues:
        """Compute exact interaction values using the ExactComputer.
        Args:
            index: The index for which to compute interaction values.
            order: The order of interactions to compute.
            budget: Optional budget for computation (not used in this computer).
        Returns:
            InteractionValues: The computed interaction values.
        """
        return self._computer(index=index, order=order)
