"""Closed-form tree explainers dispatching on the game type."""

from __future__ import annotations

from functools import singledispatch
from itertools import combinations
from math import comb

import jax.numpy as jnp
import numpy as np
from jax import Array

from shapiq.explainers._base import Explainer, reject_common_index_mistakes
from shapiq.explanations import SparseExplanationArray
from shapiq.games import Game
from shapiq.interactions import CardinalInteractionIndex
from shapiq.trees import InterventionalTreeGame


class TreeExplainer(Explainer[Array, Game[Array]]):
    """Closed-form interaction explainer for tree games.

    The game type carries the tree-explanation semantics and selects the
    algorithm: an ``InterventionalTreeGame`` decomposes into per-leaf
    present/absent constraints, whose Moebius masses are summed against the
    index's discrete-derivative weights — exact values in one pass over the
    leaves instead of ``2**n_players`` game evaluations. Any index with the
    cardinal capability works (SV, BV, SII, BII, the weighted Banzhaf
    family, CHII, STII, and the Moebius and Co-Moebius transforms).
    Explanations are sparse: only interactions whose players co-occur on a
    root-to-leaf path can carry mass, and all others default to zero.

    Example:
        >>> game = InterventionalTreeGame(to_tree_model(model), inputs=x, baseline=background)
        >>> explanation = TreeExplainer(game, SII(order=2)).explain()
        >>> pair_interaction = explanation((0, 1))
    """

    def __init__(self, game: InterventionalTreeGame, index: CardinalInteractionIndex) -> None:
        """Initialize without evaluating the game.

        Args:
            game: A registered tree game; ``InterventionalTreeGame`` ships,
                and a path-dependent sibling is the planned alternative.
            index: The interaction index to compute. Any index providing
                discrete-derivative weights works.

        Raises:
            TypeError: If the game is no registered tree game, or if the
                index provides no discrete-derivative weights.
        """
        reject_common_index_mistakes(index)
        registered = _registered_tree_games()
        if type(game) not in registered:
            supported = ", ".join(sorted(kind.__name__ for kind in registered))
            if isinstance(game, registered):
                msg = (
                    f"TreeExplainer dispatches on the exact game type: "
                    f"{type(game).__name__} subclasses a registered tree game; "
                    f"pass one of {supported} itself or register a family for it"
                )
                raise TypeError(msg)
            msg = (
                f"TreeExplainer computes registered tree games in closed form, "
                f"got {type(game).__name__}: registered games are {supported}; "
                "for arbitrary games use ExactExplainer or a sampling approximator"
            )
            raise TypeError(msg)
        if not isinstance(index, CardinalInteractionIndex):
            name = getattr(index, "name", type(index).__name__)
            msg = (
                f"TreeExplainer sums leaf Moebius masses against discrete-derivative "
                f"weights, but {name!r} provides none; any cardinal index works "
                "(for the faithful regression family use Regression on the tree game)"
            )
            raise TypeError(msg)
        super().__init__(game, index)

    def explain(self) -> SparseExplanationArray[Array]:
        """Compute the configured index exactly from the tree structure.

        Returns:
            A sparse explanation whose baseline is the game value of the
            empty coalition; interactions without tree support default to
            zero attributions.
        """
        return tree_explanation(self.game, self.index, self.order)


@singledispatch
def tree_explanation(
    game: object,
    index: CardinalInteractionIndex,
    order: int,
) -> SparseExplanationArray[Array]:
    """Compute an interaction index in closed form on a tree game.

    The algorithm dispatches on the exact game type; registering an
    implementation for a new tree game type extends ``TreeExplainer``.
    """
    del index, order
    supported = ", ".join(sorted(kind.__name__ for kind in _registered_tree_games()))
    msg = f"no closed-form tree explanation is registered for {type(game).__name__} ({supported})"
    raise TypeError(msg)


def _registered_tree_games() -> tuple[type, ...]:
    """Return the game types with a registered closed-form explanation."""
    return tuple(kind for kind in tree_explanation.registry if kind is not object)


@tree_explanation.register
def _interventional_explanation(
    game: InterventionalTreeGame,
    index: CardinalInteractionIndex,
    order: int,
) -> SparseExplanationArray[Array]:
    """Sum per-leaf Moebius masses against superset-summed derivative weights.

    A leaf with present set ``E``, absent set ``R``, and value ``v`` carries
    Moebius masses ``m(E | W) = (-1)**|W| v`` for ``W`` inside ``R``. For a
    cardinal index, ``I(T) = sum over Q containing T of m(Q) * omega_t(|Q - T|)``
    with ``omega_t(k)`` the superset-weighted sum of the discrete-derivative
    weights, so each leaf contributes to an interaction through a coefficient
    depending only on ``(|E|, |R|, |T & E|, |T & R|)``.
    """
    n_players = game.n_players
    min_size = index.min_interaction_size
    omegas = {
        size: _superset_weight_sums(
            np.asarray(index.derivative_weights(n_players, size), dtype=np.float64),
            n_players - size,
        )
        for size in range(min_size, order + 1)
    }
    coefficients: dict[tuple[int, int, int, int], float] = {}
    totals: dict[tuple[int, ...], np.ndarray | float] = {}
    for leaves in game.leaf_constraints:
        for row in range(leaves.values.shape[0]):
            present_members = np.flatnonzero(leaves.present[row])
            absent_members = np.flatnonzero(leaves.absent[row])
            value = leaves.values[row]
            n_present, n_absent = len(present_members), len(absent_members)
            for size in range(min_size, min(order, n_present + n_absent) + 1):
                for from_present in range(max(0, size - n_absent), min(n_present, size) + 1):
                    from_absent = size - from_present
                    key = (n_present, n_absent, from_present, from_absent)
                    coefficient = coefficients.get(key)
                    if coefficient is None:
                        coefficient = _leaf_coefficient(omegas[size], *key)
                        coefficients[key] = coefficient
                    if coefficient == 0.0:
                        continue
                    contribution = value * coefficient
                    for in_present in combinations(present_members, from_present):
                        for in_absent in combinations(absent_members, from_absent):
                            interaction = tuple(sorted((*in_present, *in_absent)))
                            totals[interaction] = totals.get(interaction, 0.0) + contribution
    baseline = 0.0
    for leaves in game.leaf_constraints:
        unconstrained = ~leaves.present.any(axis=-1)
        baseline = baseline + leaves.values[unconstrained].sum(axis=0)
    if min_size == 0:
        # attributions live on the centered game: centering removes exactly
        # the empty coalition's Moebius mass, which is the baseline
        totals[()] = totals.get((), 0.0) - baseline * float(omegas[0][0])
    value_shape = game.value_shape

    def _zero_attribution(interaction: object) -> Array:
        del interaction
        return jnp.zeros(value_shape, dtype=jnp.float32)

    return SparseExplanationArray(
        attributions={
            interaction: jnp.asarray(total, dtype=jnp.float32)
            for interaction, total in totals.items()
        },
        n_players=n_players,
        index=index,
        order=order,
        value_shape=value_shape,
        default_attribution=_zero_attribution,
        baseline=jnp.asarray(baseline, dtype=jnp.float32),
    )


def _superset_weight_sums(weights: np.ndarray, n_free: int) -> np.ndarray:
    """Return ``omega[k] = sum over supersets U of a k-set of weights[|U|]``."""
    omega = np.zeros(n_free + 1)
    for fixed in range(n_free + 1):
        omega[fixed] = sum(
            comb(n_free - fixed, outside - fixed) * float(weights[outside])
            for outside in range(fixed, n_free + 1)
        )
    return omega


def _leaf_coefficient(
    omega: np.ndarray,
    n_present: int,
    n_absent: int,
    from_present: int,
    from_absent: int,
) -> float:
    """Sum the leaf's signed Moebius masses hitting one interaction shape."""
    return sum(
        comb(n_absent - from_absent, extra - from_absent)
        * ((-1) ** extra)
        * float(omega[n_present - from_present + extra - from_absent])
        for extra in range(from_absent, n_absent + 1)
    )
