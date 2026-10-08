"""Ground-truth computers: exact interaction values of a game.

A computer is a thin adapter between a game and an exact algorithm of core shapiq. It never
re-implements the algorithm. Every computer

- declares which indices and orders it supports, read from the declarations of the wrapped core
  algorithm (``valid_indices`` attributes and ``Literal`` index aliases),
- raises :class:`UnsupportedComputationError` instead of computing something else, and
- returns the values of the game *as the game evaluates it*: the interactions of order ``1`` to
  ``order`` and, as ``baseline_value``, the value of the empty coalition of the game. The order-0
  term is not part of the ground truth (core algorithms disagree on it, and it does not depend on
  the interactions).

Chain of trust: closed-form games validate :class:`BruteForceComputer`, and brute force validates
every structured computer on small games (see ``tests/shapiq_benchmark``). Only then are the
structured computers used as ground truth for large games.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, cast, get_args

import numpy as np

from shapiq import ExactComputer, Game, InteractionValues
from shapiq.explainer.custom_types import ValidNNExplainerIndices
from shapiq.explainer.nn import KNNExplainer, ThresholdNNExplainer, WeightedKNNExplainer
from shapiq.explainer.product_kernel import ProductKernelComputer as CoreProductKernelComputer
from shapiq.explainer.product_kernel.product_kernel import ProductKernelSHAPIQIndices
from shapiq.game_theory.moebius_converter import MoebiusConverter, ValidMoebiusConverterIndices
from shapiq.tree.explainer import TreeExplainer
from shapiq.tree.interventional import InterventionalTreeSHAPIQ
from shapiq.tree.interventional.computer import InterventionalTreeSHAPIQIndices
from shapiq.tree.quadrature.computer import QuadratureTreeSHAPIndices
from shapiq.tree.validation import validate_tree_model
from shapiq_games.kernel import ProductKernelGame
from shapiq_games.nn import KNNGame, ThresholdNNGame, WeightedKNNGame
from shapiq_games.nn._base import NNGameBase
from shapiq_games.synthetic import SOUM, DummyGame, UnanimityGame
from shapiq_games.tree import InterventionalTreeGame, PathDependentTreeGame

if TYPE_CHECKING:
    from shapiq.tree.explainer import TreeExplainerIndices
    from shapiq.typing import IndexType

__all__ = [
    "BruteForceComputer",
    "Computer",
    "InterventionalTreeComputer",
    "KNNComputer",
    "MoebiusComputer",
    "PathDependentTreeComputer",
    "ProductKernelComputer",
    "UnsupportedComputationError",
    "default_computer",
]

DEFAULT_MAX_PLAYERS = 25
"""The default player cap of brute-force computation (``2**25`` game evaluations), the size of the
largest games of the benchmarking paper. Brute force more than doubles its time and memory with
every player: at 22 players, the Shapley values of a cheap game take about 2.5 minutes and 2.6 GB."""

VALUE_INDICES: frozenset[IndexType] = frozenset({"SV", "BV", "ELC"})
"""Values (one number per player), not interactions: only defined for order 1."""


class UnsupportedComputationError(ValueError):
    """Raised when a computer cannot compute an index, order, or game exactly."""


def _standardize(
    values: InteractionValues,
    game: Game,
    index: IndexType,
    order: int,
) -> InteractionValues:
    """Return the interactions of order 1 to ``order`` and the game's empty value as baseline.

    Only the interactions the core algorithm stored are kept (an interaction that is not stored is
    0), so a sparse result stays sparse: a tree with 100 features has a few hundred nonzero
    interactions of order 4, not all ``C(100, 4) = 3921225``.
    """
    interactions = sorted(
        (
            (interaction, float(value))
            for interaction, value in values.dict_values.items()
            if 1 <= len(interaction) <= order
        ),
        key=lambda item: (len(item[0]), item[0]),  # by order, then lexicographic
    )
    return InteractionValues(
        values=np.array([value for _, value in interactions], dtype=float),
        index=index,
        max_order=order,
        min_order=1,
        n_players=game.n_players,
        interaction_lookup={interaction: i for i, (interaction, _) in enumerate(interactions)},
        estimated=False,
        estimation_budget=None,
        baseline_value=float(game(game.empty_coalition)[0]),
    )


class Computer[G: Game](ABC):
    """A ground-truth computer bound to one game of the family ``G`` it understands.

    Attributes:
        game: The game.
    """

    name: ClassVar[str]
    """A short, stable name of the computer (part of ground-truth cache keys)."""

    def __init__(self, game: G) -> None:
        """Bind the computer to a game.

        Raises:
            UnsupportedComputationError: If the computer cannot handle the game.
        """
        if not self.supports_game(game):
            msg = f"{type(self).__name__} cannot compute {type(game).__name__} games exactly."
            raise UnsupportedComputationError(msg)
        self.game: G = game

    @classmethod
    @abstractmethod
    def supports_game(cls, game: Game) -> bool:
        """Return whether the computer can compute this game exactly."""

    @classmethod
    @abstractmethod
    def supported_indices(cls) -> tuple[IndexType, ...]:
        """Return the indices the wrapped core algorithm declares."""

    def max_order(self) -> int:
        """Return the highest supported interaction order (the number of players by default)."""
        return self.game.n_players

    def supports(self, index: IndexType, order: int) -> bool:
        """Return whether the computer can compute ``index`` up to ``order`` for its game."""
        if index not in self.supported_indices():
            return False
        if index in VALUE_INDICES and order != 1:
            return False
        return 1 <= order <= self.max_order()

    def exact_values(self, index: IndexType, order: int) -> InteractionValues:
        """Compute the exact interaction values of order 1 to ``order``.

        Args:
            index: The interaction index, e.g. ``"k-SII"``.
            order: The highest interaction order.

        Returns:
            The interactions of order ``1`` to ``order``, with the game's empty value as baseline.

        Raises:
            UnsupportedComputationError: If ``(index, order)`` is not supported.
        """
        if not self.supports(index, order):
            msg = (
                f"{type(self).__name__} does not support index={index!r} with order={order} for "
                f"a {self.game.n_players}-player {type(self.game).__name__}."
            )
            raise UnsupportedComputationError(msg)
        return _standardize(self._compute(index, order), self.game, index, order)

    @abstractmethod
    def _compute(self, index: IndexType, order: int) -> InteractionValues:
        """Run the core algorithm."""


class BruteForceComputer(Computer[Game]):
    """Brute force: evaluates all ``2**n`` coalitions with :class:`~shapiq.ExactComputer`.

    Works for every game up to ``max_players`` players (default 25). The game values are evaluated
    once and reused for every index and order.
    """

    name = "brute_force"

    def __init__(self, game: Game, *, max_players: int = DEFAULT_MAX_PLAYERS) -> None:
        """Bind the computer to a game.

        Args:
            game: The game.
            max_players: The player cap. Defaults to ``25``.

        Raises:
            UnsupportedComputationError: If the game has more than ``max_players`` players.
        """
        if game.n_players > max_players:
            msg = (
                f"Brute force is capped at {max_players} players, but the game has "
                f"{game.n_players}. Raise max_players explicitly to compute it anyway."
            )
            raise UnsupportedComputationError(msg)
        super().__init__(game)
        self._exact: ExactComputer | None = None

    @classmethod
    def supports_game(cls, game: Game) -> bool:  # noqa: ARG003
        """Brute force supports every game (the player cap is checked separately)."""
        return True

    @classmethod
    def supported_indices(cls) -> tuple[IndexType, ...]:
        """The indices of :class:`~shapiq.ExactComputer`."""
        return tuple(ExactComputer.valid_indices)

    def _compute(self, index: IndexType, order: int) -> InteractionValues:
        if self._exact is None:
            self._exact = ExactComputer(game=self.game, n_players=self.game.n_players)
        return self._exact(index=index, order=order)


def moebius_representation(game: Game) -> InteractionValues | None:
    """Return the Möbius transform of a game that knows it (``moebius_coefficients``), or ``None``."""
    coefficients = getattr(game, "moebius_coefficients", None)
    return coefficients if isinstance(coefficients, InteractionValues) else None


class MoebiusComputer(Computer[SOUM | UnanimityGame | DummyGame]):
    """Synthetic games with a known Möbius representation, via :class:`~shapiq.MoebiusConverter`.

    Supports :class:`~shapiq_games.synthetic.SOUM`, :class:`~shapiq_games.synthetic.UnanimityGame`
    and :class:`~shapiq_games.synthetic.DummyGame`, for any number of players. Besides the indices
    of the converter, it returns the Möbius transform itself (``"Moebius"``), which is sparse.
    """

    name = "moebius"

    @classmethod
    def supports_game(cls, game: Game) -> bool:
        """Games whose Möbius representation is known."""
        return moebius_representation(game) is not None

    @classmethod
    def supported_indices(cls) -> tuple[IndexType, ...]:
        """The indices of :class:`~shapiq.MoebiusConverter`, and ``"Moebius"``."""
        return (*get_args(ValidMoebiusConverterIndices), "Moebius")

    def _compute(self, index: IndexType, order: int) -> InteractionValues:
        moebius = moebius_representation(self.game)
        if moebius is None:  # excluded by supports_game
            raise UnsupportedComputationError(type(self.game).__name__)
        if index == "Moebius":
            return moebius
        # supports() checked the index against the converter's declaration
        return MoebiusConverter(moebius)(cast("ValidMoebiusConverterIndices", index), order)


class PathDependentTreeComputer(Computer[PathDependentTreeGame]):
    """Path-dependent tree games via :class:`~shapiq.tree.TreeExplainer`."""

    name = "path_dependent_tree"

    @classmethod
    def supports_game(cls, game: Game) -> bool:
        """:class:`~shapiq_games.tree.PathDependentTreeGame` games."""
        return isinstance(game, PathDependentTreeGame)

    @classmethod
    def supported_indices(cls) -> tuple[IndexType, ...]:
        """The path-dependent indices (``QuadratureTreeSHAPIndices``)."""
        return get_args(QuadratureTreeSHAPIndices)

    def _compute(self, index: IndexType, order: int) -> InteractionValues:
        explainer = TreeExplainer(
            model=self.game.model,
            mode="pathdependent",
            index=cast("TreeExplainerIndices", index),  # checked by supports()
            max_order=order,
            min_order=1,
            class_index=self.game.class_index,
        )
        return explainer.explain(self.game.x)


class InterventionalTreeComputer(Computer[InterventionalTreeGame]):
    """Interventional tree games via :class:`~shapiq.tree.interventional.InterventionalTreeSHAPIQ`.

    A new core computer is built for every call, so any order is supported.
    """

    name = "interventional_tree"

    @classmethod
    def supports_game(cls, game: Game) -> bool:
        """:class:`~shapiq_games.tree.InterventionalTreeGame` games of tree models."""
        if not isinstance(game, InterventionalTreeGame):
            return False
        try:
            validate_tree_model(game.model, class_label=game.class_index)
        except TypeError:
            return False
        return True

    @classmethod
    def supported_indices(cls) -> tuple[IndexType, ...]:
        """The indices of ``InterventionalTreeSHAPIQ``.

        Excluded: ``"CUSTOM"`` (needs a user-defined weight function) and ``"CV"``, which core
        returns as relabelled ``CHII`` values and which brute force cannot check.
        """
        excluded = {"CUSTOM", "CV"}
        return tuple(i for i in get_args(InterventionalTreeSHAPIQIndices) if i not in excluded)

    def _compute(self, index: IndexType, order: int) -> InteractionValues:
        computer = InterventionalTreeSHAPIQ(
            model=self.game.model,
            data=self.game.reference_data,
            class_index=self.game.class_index,
            index=cast("InterventionalTreeSHAPIQIndices", index),  # checked by supports()
            max_order=order,
        )
        return computer.explain_function(x=self.game.x)


class KNNComputer(Computer[NNGameBase[Any]]):
    """Nearest-neighbor games via the KNN, weighted KNN, and threshold NN explainers.

    A :class:`~shapiq_games.nn.WeightedKNNGame` is supported when it uses the explainer's weight
    discretization (``n_bits`` is set) and ``k > 1``.
    """

    name = "knn"

    @classmethod
    def supports_game(cls, game: Game) -> bool:
        """KNN, weighted KNN (with ``n_bits`` and ``k > 1``), and threshold NN games."""
        if isinstance(game, WeightedKNNGame):  # the weighted KNN explainer needs k > 1
            return game.n_bits is not None and game.k > 1
        return isinstance(game, KNNGame | ThresholdNNGame)

    @classmethod
    def supported_indices(cls) -> tuple[IndexType, ...]:
        """The indices of the nearest-neighbor explainers (``ValidNNExplainerIndices``)."""
        return get_args(ValidNNExplainerIndices)

    def max_order(self) -> int:
        """The nearest-neighbor explainers compute order-1 values only."""
        return 1

    def _compute(self, index: IndexType, order: int) -> InteractionValues:  # noqa: ARG002
        game = self.game
        if isinstance(game, WeightedKNNGame) and game.n_bits is not None:
            explainer = WeightedKNNExplainer(
                game.model, class_index=game.class_index, n_bits=game.n_bits
            )
        elif isinstance(game, KNNGame):
            explainer = KNNExplainer(game.model, class_index=game.class_index)
        elif isinstance(game, ThresholdNNGame):
            explainer = ThresholdNNExplainer(game.model, class_index=game.class_index)
        else:  # excluded by supports_game
            raise UnsupportedComputationError(type(game).__name__)
        return explainer.explain(game.x)


class ProductKernelComputer(Computer[ProductKernelGame]):
    """Product kernel games via :class:`~shapiq.explainer.product_kernel.ProductKernelExplainer`."""

    name = "product_kernel"

    @classmethod
    def supports_game(cls, game: Game) -> bool:
        """:class:`~shapiq_games.kernel.ProductKernelGame` games."""
        return isinstance(game, ProductKernelGame)

    @classmethod
    def supported_indices(cls) -> tuple[IndexType, ...]:
        """The indices of the product kernel explainer (``ProductKernelSHAPIQIndices``)."""
        return get_args(ProductKernelSHAPIQIndices)

    def max_order(self) -> int:
        """The product kernel explainer computes order-1 values only."""
        return 1

    def _compute(self, index: IndexType, order: int) -> InteractionValues:  # noqa: ARG002
        # the explainer only accepts library models, so the core computer it uses is called with
        # the converted model directly (the same two calls as ProductKernelExplainer)

        model = self.game.model
        core = CoreProductKernelComputer(model, max_order=1, index="SV")
        kernel_vectors = core.compute_kernel_vectors(model.X_train, self.game.x)
        values = np.array([core.compute_shapley_value(kernel_vectors, j) for j in range(model.d)])
        return InteractionValues(
            values=values,
            index="SV",
            max_order=1,
            min_order=1,
            n_players=model.d,
            interaction_lookup={(j,): j for j in range(model.d)},
            baseline_value=float(self.game(self.game.empty_coalition)[0]),
        )


_STRUCTURED_COMPUTERS: tuple[type[Computer[Any]], ...] = (
    MoebiusComputer,
    PathDependentTreeComputer,
    InterventionalTreeComputer,
    KNNComputer,
    ProductKernelComputer,
)


def default_computer(game: Game, *, max_players: int = DEFAULT_MAX_PLAYERS) -> Computer[Any]:
    """Return the best available computer for a game.

    A structured computer for the game's type is preferred; otherwise brute force is used up to
    ``max_players`` players.

    Args:
        game: The game.
        max_players: The player cap of brute force. Defaults to ``25``.

    Returns:
        The computer bound to the game.

    Raises:
        UnsupportedComputationError: If no structured computer applies and the game is too large
            for brute force.
    """
    for computer_class in _STRUCTURED_COMPUTERS:
        if computer_class.supports_game(game):
            return computer_class(game)
    return BruteForceComputer(game, max_players=max_players)
