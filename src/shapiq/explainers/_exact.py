from __future__ import annotations

from itertools import combinations
from math import comb
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
from jax import Array

from shapiq._shape import validate_int
from shapiq.coalitions import DenseCoalitionArray
from shapiq.errors import UnsupportedGameError
from shapiq.explainers._base import Explainer
from shapiq.explanations import DenseExplanationArray
from shapiq.games import Game

if TYPE_CHECKING:
    from collections.abc import Callable

    from shapiq.interactions import InteractionIndexName

_CONSTRAINT_WEIGHT = 1e7
_SUPPORTED_INDICES = ("SV", "BV", "SII", "BII", "STII", "FSII")


class ExactExplainer(Explainer[Array, Game[Array]]):
    """Explainer computing interaction indices exactly from all coalitions.

    The exact explainer evaluates the game once on every one of the
    ``2**n_players`` coalitions and computes the requested index without
    sampling error, which is feasible for games with roughly up to fifteen
    players. Supported indices are SV, BV, SII, BII, STII, and FSII. The
    powerset evaluation happens on the first ``explain()`` call and is
    reused afterwards; construction never evaluates the game. Exactness is
    up to the game's own value precision (float32 by default); the index
    computations themselves run in float64.

    Example:
        >>> explainer = ExactExplainer(game, "SII", order=2)
        >>> explanation = explainer.explain()
        >>> pair_interaction = explanation((0, 1))
    """

    def __init__(
        self,
        game: Game[Array],
        interaction_index: InteractionIndexName,
        *,
        order: int | None = None,
    ) -> None:
        """Initialize without evaluating the game.

        Args:
            game: Game to explain. Must produce scalar values per coalition.
                The game is evaluated on all ``2**n_players`` coalitions.
            interaction_index: The index to compute. One of ``"SV"``,
                ``"BV"``, ``"SII"``, ``"BII"``, ``"STII"``, or ``"FSII"``.
            order: Maximum interaction order. Defaults to ``1`` for SV and
                BV and to ``2`` for the interaction indices.

        Raises:
            ValueError: If the index is not supported by the exact explainer
                or the order is out of range for the index.
        """
        if interaction_index not in _SUPPORTED_INDICES:
            msg = (
                f"ExactExplainer does not support {interaction_index!r}; "
                f"supported indices are {', '.join(_SUPPORTED_INDICES)}"
            )
            raise ValueError(msg)
        if order is None:
            order = 1 if interaction_index in ("SV", "BV") else 2
        validate_int("order", order, minimum=1)
        super().__init__(game, interaction_index, order)
        self._powerset_values: Array | None = None

    def explain(self) -> DenseExplanationArray[Array]:
        """Compute the configured index exactly from all game values.

        Returns:
            A dense explanation. SV, BV, STII, and FSII carry the
            empty-coalition value as the order-0 attribution; SII and BII
            represent orders one through ``order``.
        """
        n_players = self.game.n_players
        values = np.asarray(self._game_values(), dtype=np.float64)
        masks = _powerset_masks(n_players)
        index = self.interaction_index
        if index in ("SV", "SII"):
            attributions = _derivative_attributions(values, masks, self.order, _sii_weights)
        elif index in ("BV", "BII"):
            attributions = _derivative_attributions(values, masks, self.order, _bii_weights)
        elif index == "STII":
            attributions = _stii_attributions(values, masks, self.order)
        else:  # FSII
            attributions = _fsii_attributions(values, masks, self.order)
        if index in ("SV", "BV", "STII", "FSII"):
            attributions[0] = values[..., :1]
        return DenseExplanationArray(
            attributions_by_order={
                size: jnp.asarray(block) for size, block in attributions.items()
            },
            n_players=n_players,
            interaction_index=index,
            order=self.order,
            shape=self.game.target_shape,
        )

    def _game_values(self) -> Array:
        """Evaluate the game on the full powerset once and reuse the values."""
        if self._powerset_values is None:
            n_players = self.game.n_players
            coalitions = DenseCoalitionArray(jnp.asarray(_powerset_masks(n_players)))
            values = jnp.asarray(self.game(coalitions))
            if values.shape != (*self.game.target_shape, 2**n_players):
                msg = (
                    "exact explanation requires scalar game values per coalition: "
                    f"expected shape {(*self.game.target_shape, 2**n_players)}, "
                    f"got {values.shape}"
                )
                raise UnsupportedGameError(msg)
            self._powerset_values = values
        return self._powerset_values


def _powerset_masks(n_players: int) -> np.ndarray:
    """Return all coalitions as dense masks, ordered by size then lexicographically."""
    rows = []
    for size in range(n_players + 1):
        for members in combinations(range(n_players), size):
            row = np.zeros(n_players, dtype=bool)
            row[list(members)] = True
            rows.append(row)
    return np.asarray(rows)


def _sii_weights(n_players: int, size: int) -> np.ndarray:
    """Return SII discrete-derivative weights per outside-coalition size."""
    return np.asarray(
        [
            1.0 / ((n_players - size + 1) * comb(n_players - size, t))
            for t in range(n_players - size + 1)
        ],
    )


def _bii_weights(n_players: int, size: int) -> np.ndarray:
    """Return BII discrete-derivative weights per outside-coalition size."""
    return np.full(n_players - size + 1, 2.0 ** -(n_players - size))


def _moebius_weights(n_players: int, size: int) -> np.ndarray:
    """Return discrete-derivative-at-empty weights per outside-coalition size."""
    weights = np.zeros(n_players - size + 1)
    weights[0] = 1.0
    return weights


def _stii_top_weights(n_players: int, size: int) -> np.ndarray:
    """Return top-order STII discrete-derivative weights."""
    return np.asarray(
        [size / (n_players * comb(n_players - 1, t)) for t in range(n_players - size + 1)],
    )


def _derivative_attributions(
    values: np.ndarray,
    masks: np.ndarray,
    order: int,
    weight_fn: Callable[[int, int], np.ndarray],
) -> dict[int, np.ndarray]:
    """Accumulate weighted discrete derivatives for every size up to order."""
    return {
        size: _weighted_derivatives(values, masks, size, weight_fn)
        for size in range(1, order + 1)
    }


def _weighted_derivatives(
    values: np.ndarray,
    masks: np.ndarray,
    size: int,
    weight_fn: Callable[[int, int], np.ndarray],
) -> np.ndarray:
    """Sum signed, weighted game values into per-interaction attributions."""
    n_players = masks.shape[-1]
    weights = weight_fn(n_players, size)
    member_masks = _interaction_masks(n_players, size)
    intersections = masks.astype(np.int64) @ member_masks.T.astype(np.int64)
    outside_sizes = masks.sum(axis=-1)[:, None] - intersections
    kernel = ((-1.0) ** (size - intersections)) * weights[outside_sizes]
    return np.einsum("...c,ci->...i", values, kernel)


def _stii_attributions(
    values: np.ndarray,
    masks: np.ndarray,
    order: int,
) -> dict[int, np.ndarray]:
    """Combine exact lower-order derivatives at empty with top-order STII."""
    attributions = {
        size: _weighted_derivatives(values, masks, size, _moebius_weights)
        for size in range(1, order)
    }
    attributions[order] = _weighted_derivatives(values, masks, order, _stii_top_weights)
    return attributions


def _fsii_attributions(
    values: np.ndarray,
    masks: np.ndarray,
    order: int,
) -> dict[int, np.ndarray]:
    """Solve the faithful weighted least squares problem on the full powerset."""
    n_players = masks.shape[-1]
    sizes = masks.sum(axis=-1)
    mu = np.zeros(n_players + 1)
    mu[0] = mu[n_players] = _CONSTRAINT_WEIGHT
    for size in range(1, n_players):
        mu[size] = 1.0 / ((n_players - 1) * comb(n_players - 2, size - 1))
    sqrt_weights = np.sqrt(mu[sizes])
    design = _fsii_design_matrix(masks, order)
    response = values - values[..., :1]
    flat_response = response.reshape(-1, masks.shape[0]).T
    solution, *_ = np.linalg.lstsq(
        sqrt_weights[:, None] * design,
        sqrt_weights[:, None] * flat_response,
        rcond=None,
    )
    attributions: dict[int, np.ndarray] = {}
    offset = 1  # skip the empty-interaction column
    for size in range(1, order + 1):
        n_interactions = comb(n_players, size)
        block = solution[offset : offset + n_interactions].T
        attributions[size] = block.reshape(*values.shape[:-1], n_interactions)
        offset += n_interactions
    return attributions


def _fsii_design_matrix(masks: np.ndarray, order: int) -> np.ndarray:
    """Return subset-membership columns for all interactions up to order."""
    n_players = masks.shape[-1]
    columns = [np.ones((masks.shape[0], 1))]
    for size in range(1, order + 1):
        member_masks = _interaction_masks(n_players, size)
        intersections = masks.astype(np.int64) @ member_masks.T.astype(np.int64)
        columns.append((intersections == size).astype(np.float64))
    return np.concatenate(columns, axis=1)


def _interaction_masks(n_players: int, size: int) -> np.ndarray:
    """Return dense masks of all size-sized interactions in lexicographic order."""
    member_masks = np.zeros((comb(n_players, size), n_players), dtype=bool)
    for row, members in enumerate(combinations(range(n_players), size)):
        member_masks[row, list(members)] = True
    return member_masks
