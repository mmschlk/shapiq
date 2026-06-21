"""Benchmark ground truth computers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, TypeVar, runtime_checkable

from shapiq import ExactComputer, Game
from shapiq.imputer.tabpfn_imputer import TabPFNImputer
from shapiq.tree.explainer import TreeExplainer
from shapiq.tree.interventional.explainer import InterventionalTreeExplainer
from shapiq.typing import IndexType
from shapiq_games.benchmark.local_xai.benchmark_image import ImageClassifier

T_Index_contra = TypeVar("T_Index_contra", bound=IndexType, contravariant=True)

if TYPE_CHECKING:
    from shapiq import InteractionValues
    from shapiq.tree.interventional.game import InterventionalGame
    from shapiq_games.benchmark.local_xai.base import LocalExplanation
    from shapiq_games.benchmark.treeshapiq_xai.base import TreeSHAPIQXAI


@runtime_checkable
class GroundTruthComputer(Protocol[T_Index_contra]):
    """A protocol for ground truth computers that compute exact interaction values.

    This protocol defines the interface for any ground truth computer that can compute exact
    interaction values for a given game and index type.
    """

    def exact_values(
        self, index: T_Index_contra, order: int, **kwargs: object
    ) -> InteractionValues:
        """Compute the exact interaction values for a given index and order.

        Args:
            index: The index type for which to compute the interaction values.
            order: The order of interactions to compute.
            **kwargs: Additional keyword arguments for computation.

        Returns:
            InteractionValues: The computed interaction values for the specified index and order.
        """
        ...


class BruteForceComputer[In: Game, IndexT: IndexType](GroundTruthComputer[IndexT]):
    """A brute force computer for exact computation of interaction values."""

    def __init__(self, game: In) -> None:
        """Initialize a BruteForceComputer instance."""
        self.game = game
        self._computer = ExactComputer(game=game, n_players=game.n_players, evaluate_game=False)

    def exact_values(self, index: IndexT, order: int, **kwargs: object) -> InteractionValues:
        """Compute the exact values using brute force."""
        return self._computer(index=index, order=order, **kwargs)


class InterventionalComputer(GroundTruthComputer[IndexType]):
    """Exact computer for interventional games using the InterventionalTreeExplainer."""

    def __init__(self, game: InterventionalGame) -> None:
        """Initialize the interventional computer for a given game."""
        self.game = game
        self._computer = InterventionalTreeExplainer(
            model=self.game.model,
            data=self.game.data,
            debug=False,
            class_index=self.game.class_index,
        )

    def exact_values(self, index: IndexType, order: int, **kwargs: Any) -> InteractionValues:
        """Compute exact interaction values using the InterventionalTreeExplainer.

        Args:
            index: The index for which to compute interaction values.
            order: The order of interactions to compute.
            **kwargs: Additional keyword arguments for computation.

        Returns:
            InteractionValues: The computed interaction values.
        """
        self._computer.index = index
        self._computer.max_order = order
        return self._computer.explain_function(x=self.game.target_instance[0], **kwargs)


class PathdependentComputer(GroundTruthComputer[IndexType]):
    """Exact computer for tree-based games using the TreeExplainer."""

    def __init__(self, game: TreeSHAPIQXAI) -> None:
        """Initialize the pathdependent computer for a given game."""
        self.game = game
        self._computer = TreeExplainer(
            model=self.game.model,
            class_index=self.game.class_label,
        )

    def exact_values(self, index: IndexType, order: int, **kwargs: object) -> InteractionValues:
        """Compute exact interaction values using the TreeExplainer.

        Args:
            index: The index for which to compute interaction values.
            order: The order of interactions to compute.
            **kwargs: Additional keyword arguments for computation.

        Returns:
            InteractionValues: The computed interaction values.
        """
        return self._computer.explain_function(
            x=self.game.x_explain,
            index=index,
            max_order=order,
            **kwargs,
        )


class LocalXAIComputer(GroundTruthComputer[IndexType]):
    """Exact computer for local explanation games using the ExactComputer."""

    def __init__(self, game: LocalExplanation) -> None:
        """Initialize the local XAI computer for a given game."""
        self.game = game
        self._computer = ExactComputer(game=game, n_players=game.n_players, evaluate_game=False)

    def exact_values(self, index: IndexType, order: int, **kwargs: Any) -> InteractionValues:
        """Compute exact interaction values using the ExactComputer.

        Args:
            index: The index for which to compute interaction values.
            order: The order of interactions to compute.
            **kwargs: Additional keyword arguments for computation.

        Returns:
            InteractionValues: The computed interaction values.
        """
        return self._computer(
            index=index,
            order=order,
            **kwargs,
        )


class TabPFNComputer(BruteForceComputer[TabPFNImputer, IndexType]):
    """Exact computer for TabPFN imputers using the TabPFNExplainer."""


class ImageComputer(BruteForceComputer[ImageClassifier, IndexType]):
    """Exact computer for image classifier games using the ExactComputer."""
