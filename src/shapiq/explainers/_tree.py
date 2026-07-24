"""Closed-form tree explainers dispatching on the game type."""

from __future__ import annotations

import importlib
from functools import singledispatch
from itertools import combinations
from math import comb, prod
from typing import TYPE_CHECKING, cast

import jax.numpy as jnp
import numpy as np
from jax import Array

from shapiq.errors import UnsupportedGameError
from shapiq.explainers._base import Explainer, reject_common_index_mistakes
from shapiq.explainers.approximators._estimate import Estimate
from shapiq.explanations import SparseExplanationArray
from shapiq.games import Game
from shapiq.interactions import CardinalInteractionIndex
from shapiq.sampling import EmptyState
from shapiq.trees import InterventionalTreeGame

if TYPE_CHECKING:
    from collections.abc import Callable


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
                Subclasses inherit the closest registered ancestor's
                algorithm.
            index: The interaction index to compute. Any index providing
                discrete-derivative weights works.

        Raises:
            UnsupportedGameError: If the game is no registered tree game.
            TypeError: If the index provides no discrete-derivative weights.
        """
        reject_common_index_mistakes(index)
        registered = _registered_tree_games()
        if not isinstance(game, registered):
            supported = ", ".join(sorted(kind.__name__ for kind in registered))
            msg = (
                f"TreeExplainer computes registered tree games in closed form, "
                f"got {type(game).__name__}: registered games are {supported}; "
                "for arbitrary games use ExactExplainer or a sampling approximator"
            )
            raise UnsupportedGameError(msg)
        if not isinstance(index, CardinalInteractionIndex):
            name = getattr(index, "name", type(index).__name__)
            msg = (
                f"TreeExplainer sums leaf Moebius masses against discrete-derivative "
                f"weights, but {name!r} provides none; any cardinal index works "
                "(for the faithful regression family use Regression on the tree game)"
            )
            raise TypeError(msg)
        super().__init__(game, index)

    def estimate(self) -> Estimate:
        """Compute the closed-form explanation as an estimate.

        The estimate's provenance is honest about the price: a tree
        explanation needs no game evaluations, so its evidence is empty
        and ``spent`` reads zero.
        """
        return Estimate(
            evidence=EmptyState(),
            bank=0,
            n_players=self.game.n_players,
            view=self._view(),
            target_shape=tuple(self.game.target_shape),
            value_shape=tuple(self.game.value_shape),
        )

    def _view(self) -> SparseExplanationArray[Array]:
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

    The algorithm dispatches on the game type; subclasses resolve to the
    closest registered ancestor's algorithm through the MRO, and registering
    an implementation for a new tree game type extends ``TreeExplainer``.
    """
    del index, order
    supported = ", ".join(sorted(kind.__name__ for kind in _registered_tree_games()))
    msg = f"no closed-form tree explanation is registered for {type(game).__name__} ({supported})"
    raise UnsupportedGameError(msg)


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
    accumulate = _accumulate_cext if _use_cext(game, order) else _accumulate_python
    totals = accumulate(game, omegas, min_size, order)
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
        return jnp.zeros(value_shape)

    # the closed forms run in exact float64 on the host; the explanation
    # re-enters the stack in the default JAX precision (float64 under x64)
    return SparseExplanationArray(
        attributions={
            interaction: jnp.asarray(total) for interaction, total in totals.items()
        },
        n_players=n_players,
        index=index,
        order=order,
        value_shape=value_shape,
        default_attribution=_zero_attribution,
        baseline=jnp.asarray(baseline),
    )


def _superset_weight_sums(weights: np.ndarray, n_free: int) -> np.ndarray:
    """Return ``omega[k] = sum over supersets U of a k-set of weights[|U|]``.

    Zero weights are skipped before the binomial factor is computed: the
    exact integer overflows float conversion near 1023 players, and indices
    with sparse weights (the Moebius family) stay serviceable at any size.
    Dense nonzero weights still overflow beyond that point; the honest fix
    is log-space accumulation, recorded in the tree-story plan.
    """
    omega = np.zeros(n_free + 1)
    for fixed in range(n_free + 1):
        total = 0.0
        for outside in range(fixed, n_free + 1):
            weight = float(weights[outside])
            if weight != 0.0:
                total += comb(n_free - fixed, outside - fixed) * weight
        omega[fixed] = total
    return omega


def _leaf_coefficient(
    omega: np.ndarray,
    n_present: int,
    n_absent: int,
    from_present: int,
    from_absent: int,
) -> float:
    """Sum the leaf's signed Moebius masses hitting one interaction shape."""
    total = 0.0
    for extra in range(from_absent, n_absent + 1):
        weight = float(omega[n_present - from_present + extra - from_absent])
        if weight != 0.0:
            total += comb(n_absent - from_absent, extra - from_absent) * ((-1) ** extra) * weight
    return total


def _load_cext() -> Callable[..., tuple[bytes, bytes]] | None:
    """Return the compiled kernel's entry point, if the extension was built."""
    try:
        module = importlib.import_module("shapiq.trees._interventional_cext")
    except ImportError:  # pragma: no cover - pure-python installs
        return None
    return cast("Callable[..., tuple[bytes, bytes]]", module.accumulate)


_cext_accumulate = _load_cext()
_CEXT_MAX_ORDER = 4  # the kernel packs interactions as four 16-bit players
_CEXT_MAX_PLAYERS = 0xFFFE


def _use_cext(game: InterventionalTreeGame, order: int) -> bool:
    """Return whether the compiled kernel serves this configuration.

    Vector-valued games (multiclass margins, class probabilities) are
    served too: the kernel accumulates rows of the flattened value shape.
    """
    return (
        _cext_accumulate is not None
        and order <= _CEXT_MAX_ORDER
        and game.n_players <= _CEXT_MAX_PLAYERS
    )


def _accumulate_python(
    game: InterventionalTreeGame,
    omegas: dict[int, np.ndarray],
    min_size: int,
    order: int,
) -> dict[tuple[int, ...], np.ndarray | float]:
    """Accumulate per-leaf interaction contributions in pure Python."""
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
    return totals


def _accumulate_cext(
    game: InterventionalTreeGame,
    omegas: dict[int, np.ndarray],
    min_size: int,
    order: int,
) -> dict[tuple[int, ...], np.ndarray | float]:
    """Accumulate per-leaf interaction contributions in the compiled kernel.

    The kernel receives the ensemble's flattened leaf constraints and one
    dense coefficient table computed from the same ``_leaf_coefficient`` as
    the Python path, and returns packed interaction keys with their sums.
    """
    kernel = _cext_accumulate
    if kernel is None:  # _use_cext gates this path; a direct call gets a real error
        msg = "the compiled tree kernel is not available in this install"
        raise RuntimeError(msg)
    present = np.concatenate([leaves.present for leaves in game.leaf_constraints])
    absent = np.concatenate([leaves.absent for leaves in game.leaf_constraints])
    value_width = prod(game.value_shape)
    values = np.ascontiguousarray(
        np.concatenate([leaves.values for leaves in game.leaf_constraints]).reshape(
            present.shape[0], value_width
        ),
        dtype=np.float64,
    )
    present_counts = present.sum(axis=1)
    absent_counts = absent.sum(axis=1)
    present_offsets = np.zeros(present.shape[0] + 1, dtype=np.int64)
    np.cumsum(present_counts, out=present_offsets[1:])
    absent_offsets = np.zeros(absent.shape[0] + 1, dtype=np.int64)
    np.cumsum(absent_counts, out=absent_offsets[1:])
    present_members = _padded_members(present)
    absent_members = _padded_members(absent)
    max_present = int(present_counts.max())
    max_absent = int(absent_counts.max())
    table = np.zeros((max_present + 1, max_absent + 1, order + 1, order + 1), dtype=np.float64)
    for n_present in range(max_present + 1):
        for n_absent in range(max_absent + 1):
            if n_present + n_absent > game.n_players:
                continue  # the maxima come from different leaves; no leaf has both
            for size in range(min_size, min(order, n_present + n_absent) + 1):
                for from_present in range(max(0, size - n_absent), min(n_present, size) + 1):
                    table[n_present, n_absent, from_present, size - from_present] = (
                        _leaf_coefficient(
                            omegas[size],
                            n_present,
                            n_absent,
                            from_present,
                            size - from_present,
                        )
                    )
    keys_bytes, sums_bytes = kernel(
        present_offsets,
        present_members,
        absent_offsets,
        absent_members,
        values,
        np.ascontiguousarray(table),
        max_present,
        max_absent,
        min_size,
        order,
        value_width,
    )
    keys = np.frombuffer(keys_bytes, dtype=np.uint64)
    sums = np.frombuffer(sums_bytes, dtype=np.float64).reshape(keys.shape[0], value_width)
    totals: dict[tuple[int, ...], np.ndarray | float] = {}
    for position, key in enumerate(keys.tolist()):
        interaction = []
        packed = key
        while packed:
            interaction.append((packed & 0xFFFF) - 1)
            packed >>= 16
        row = sums[position]
        total = float(row[0]) if game.value_shape == () else row.reshape(game.value_shape)
        totals[tuple(interaction)] = total
    return totals


def _padded_members(masks: np.ndarray) -> np.ndarray:
    """Return row-major member indices, padded so empty buffers stay valid."""
    members = np.nonzero(masks)[1].astype(np.int64)
    if members.size == 0:
        return np.zeros(1, dtype=np.int64)
    return np.ascontiguousarray(members)
