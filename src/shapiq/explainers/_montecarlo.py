"""SHAP-IQ and SVARM-IQ: counts-times-law estimators over one size-law design.

Both methods estimate any cardinal interaction index from its standard
form (Fumagalli et al. 2023, Theorem 1): the index is a sum over all
coalitions ``W`` of ``v(W)`` times a weight depending only on ``|W|`` and
``|W ∩ T|``, with the discrete-derivative weights supplied by the index's
cardinal capability. The shared sampling design splits coalition sizes in
two: the border sizes (below the order, above ``n - order``) carry most
of the derivative weight and are enumerated exactly — the empty and grand
coalition in the seed block, the rest as the deterministic prelude — and
the interior sizes are sampled from the KernelSHAP size law
``1 / (s * (n - s))``, one coalition per unit.

The estimators differ only in how the sampled region enters the sum.
SHAP-IQ importance-weights every sampled row by the inverse of its
sampling probability. SVARM-IQ stratifies the sampled rows by the exact
intersection ``W ∩ T`` and the coalition size and enters each stratum's
empirical mean times the stratum's true size, which drops the sampling
law from the estimate entirely. Both are pure functions of the evidence —
repeated coalitions contribute through their multiplicity in the stored
stream — so budget splits, rollback, and replay are exact.

Unlike v1, the deterministic region never depends on the budget: v1 grew
its exactly-enumerated sizes with the spend, which ties the sampling law
to the total budget and is irreconcilable with anytime refinement.
"""

from __future__ import annotations

from itertools import chain, combinations
from math import comb
from typing import TYPE_CHECKING, NamedTuple

import jax.numpy as jnp
import numpy as np
from jax import Array

from shapiq._shape import ensure_bool
from shapiq.errors import InsufficientSamplesError
from shapiq.explainers._approximator import Approximator
from shapiq.explainers._binding import reject_common_index_mistakes
from shapiq.explainers._permutation import interaction_members
from shapiq.interactions import CardinalInteractionIndex
from shapiq.sampling import PairedSampler, SampledEvidence, SizeKernelSampler

if TYPE_CHECKING:
    from shapiq.games import Game
    from shapiq.sampling import Evidence, ShareSamples


class _SizeLawSampling(Approximator):
    """Shared design of the SHAP-IQ family: exact borders, sampled interior."""

    index: CardinalInteractionIndex

    def __init__(
        self,
        game: Game[Array],
        index: CardinalInteractionIndex,
        *,
        random_state: Array | int = 0,
        share_samples: ShareSamples = False,
        paired: bool | None = None,
        deduplicate: bool = False,
    ) -> None:
        """Initialize without evaluating the game.

        Args:
            game: Game to explain. Must have at least two players.
            index: The interaction index to estimate. Any index providing
                discrete-derivative weights works (``SV()``,
                ``SII(order=k)``, ``STII(order=k)``, the Banzhaf family,
                ``CHII(order=k)``).
            random_state: Integer seed or JAX PRNG key for drawing
                coalitions.
            share_samples: Policy for sharing sampled coalitions across
                explanation-target axes.
            paired: Whether every sampled coalition is accompanied by its
                complement (the pairing trick); the interior size law is
                complement-symmetric, so pairing is always sound here. The
                default ``None`` means unpaired: coalition draws pair only
                on request.
            deduplicate: Whether to evaluate each distinct coalition at most
                once; repeats reuse stored values and only novel evaluations
                count toward the budget. Requires shared samples.

        Raises:
            TypeError: If the index provides no discrete-derivative weights.
            ValueError: If every coalition size is deterministic
                (``2 * order > n_players``), leaving nothing to sample.
        """
        reject_common_index_mistakes(index)
        if not isinstance(index, CardinalInteractionIndex):
            name = getattr(index, "name", type(index).__name__)
            msg = (
                f"{type(self).__name__} estimates discrete-derivative representations, "
                f"but {name!r} provides none; any cardinal index works "
                "(for the faithful regression family use Regression)"
            )
            raise TypeError(msg)
        n_players = game.n_players
        order = n_players if index.order is None else index.order
        size_weights = np.zeros(n_players + 1)
        for size in range(order, n_players - order + 1):
            if 0 < size < n_players:
                size_weights[size] = 1.0 / (size * (n_players - size))
        if size_weights.sum() == 0.0:
            msg = (
                f"every coalition size of the {n_players}-player game is enumerated "
                f"exactly at order {order}: there is nothing left to sample; "
                "use ExactExplainer"
            )
            raise ValueError(msg)
        base_sampler = SizeKernelSampler(
            n_players,
            size_weights,
            game.target_shape,
            share_samples=share_samples,
            random_state=random_state,
        )
        if paired is None:
            paired = False  # the size law pairs only on request, like the walk families
        else:
            ensure_bool("paired", paired)
        sampler = PairedSampler(base_sampler) if paired else base_sampler
        super().__init__(
            game,
            sampler,
            index,
            render=_coalition_rows,
            unit_length=1,
            prelude_masks=_border_masks(n_players, order),
            deduplicate=deduplicate,
        )
        self._interior_probabilities = size_weights / size_weights.sum()

    def _decode(self, evidence: Evidence, bank: int) -> _SizeLawEvidence:
        """Split stored rows into the exact border block and the sampled interior."""
        if not isinstance(evidence, SampledEvidence):
            self._require_no_evidence_yet()
        n_deterministic = self.n_seed_samples
        if evidence.n_samples - n_deterministic < 1:
            msg = (
                "explaining requires at least one completed sampled unit: "
                f"estimate with at least {self.min_budget} evaluations in total "
                f"(currently {evidence.n_samples} stored, {bank} banked)"
            )
            raise InsufficientSamplesError(msg)
        masks = jnp.asarray(evidence.coalitions.to_dense())
        values = jnp.asarray(evidence.values)  # canonical: sample axis last
        value_empty = values[..., 0]
        return _SizeLawEvidence(
            masks=masks,
            centered=values - value_empty[..., None],
            sizes=jnp.sum(masks, axis=-1),
            n_deterministic=n_deterministic,
            value_empty=value_empty,
        )

    def _estimate_parts(
        self,
        evidence: Evidence,
        bank: int,
    ) -> tuple[dict[int, Array], Array | None]:
        """Estimate every order's block; the empty slot carries the baseline."""
        decoded = self._decode(evidence, bank)
        n_players = self.game.n_players
        attributions: dict[int, Array] = {}
        for size in range(max(self.index.min_interaction_size, 1), self.order + 1):
            members = interaction_members(n_players, size)
            table = _standard_form_table(self.index, n_players, size)
            attributions[size] = self._order_block(decoded, members, table, size)
        return attributions, decoded.value_empty

    def _order_block(
        self,
        decoded: _SizeLawEvidence,
        members: Array,
        table: Array,
        interaction_size: int,
    ) -> Array:
        """Estimate one order's interactions from the decoded evidence."""
        raise NotImplementedError


class SHAPIQ(_SizeLawSampling):
    """SHAP-IQ: importance-weighted any-order interaction estimation.

    Every sampled coalition enters the standard-form sum weighted by the
    inverse of its sampling probability under the interior size law, so
    the estimate is unbiased at any budget (Fumagalli et al. 2023); with
    ``SV()`` this is exactly Unbiased KernelSHAP (Covert and Lee 2021).
    Border sizes are enumerated in the seed block and prelude and enter
    the sum exactly.

    Example:
        >>> approximator = SHAPIQ(game, SII(order=2), random_state=0)
        >>> estimate = approximator.estimate(500)
        >>> pair_interaction = estimate[(0, 1)]
    """

    def _order_block(
        self,
        decoded: _SizeLawEvidence,
        members: Array,
        table: Array,
        interaction_size: int,
    ) -> Array:
        del interaction_size
        n_players = self.game.n_players
        boundary = decoded.n_deterministic
        row_weights = _row_term_weights(decoded, members, table)
        exact = jnp.sum(
            decoded.centered[..., :boundary, None] * row_weights[..., :boundary, :],
            axis=-2,
        )
        n_sampled = decoded.masks.shape[-2] - boundary
        # a sampled coalition's probability: size law times uniform-in-size
        inclusion = jnp.asarray(
            [
                self._interior_probabilities[size] / comb(n_players, size)
                for size in range(n_players + 1)
            ],
        )
        adjustment = 1.0 / (n_sampled * inclusion[decoded.sizes[..., boundary:]])
        sampled = jnp.sum(
            decoded.centered[..., boundary:, None]
            * row_weights[..., boundary:, :]
            * adjustment[..., None],
            axis=-2,
        )
        return exact + sampled


class SVARMIQ(_SizeLawSampling):
    """SVARM-IQ: stratified any-order interaction estimation.

    The sampled interior enters the standard-form sum stratified by the
    exact intersection ``W ∩ T`` and the coalition size: each stratum
    contributes its empirical mean times its true stratum size
    (Kolpaczki et al. 2024), so the sampling law cancels out of the
    estimate and strata without samples contribute zero. Border sizes are
    enumerated in the seed block and prelude and enter the sum exactly.

    Example:
        >>> approximator = SVARMIQ(game, SII(order=2), random_state=0)
        >>> estimate = approximator.estimate(500)
        >>> pair_interaction = estimate[(0, 1)]
    """

    def _order_block(
        self,
        decoded: _SizeLawEvidence,
        members: Array,
        table: Array,
        interaction_size: int,
    ) -> Array:
        n_players = self.game.n_players
        boundary = decoded.n_deterministic
        row_weights = _row_term_weights(decoded, members, table)
        exact = jnp.sum(
            decoded.centered[..., :boundary, None] * row_weights[..., :boundary, :],
            axis=-2,
        )
        # stratum of a sampled row, per interaction: the exact intersection
        # pattern (rank of W ∩ T inside T) paired with the coalition size
        bits = decoded.masks[..., boundary:, :][..., members]  # (..., rows, D, t)
        patterns = jnp.sum(bits * (2 ** jnp.arange(members.shape[-1])), axis=-1)
        strata = patterns * (n_players + 1) + decoded.sizes[..., boundary:, None]
        n_interactions = members.shape[0]
        n_bins = (2**interaction_size) * (n_players + 1)
        shape = jnp.broadcast_shapes(
            decoded.centered[..., boundary:, None].shape,
            strata.shape,
        )
        contributions = jnp.broadcast_to(decoded.centered[..., boundary:, None], shape)
        strata = jnp.broadcast_to(strata, shape)
        sums = _scatter_bins(contributions, strata, n_interactions, n_bins)
        counts = _scatter_bins(jnp.ones_like(contributions), strata, n_interactions, n_bins)
        means = sums / jnp.maximum(counts, 1.0)
        weights = _stratum_weight_table(self.index, n_players, interaction_size)
        sampled = jnp.sum(means * jnp.asarray(weights.reshape(-1)), axis=-1)
        return exact + sampled


class _SizeLawEvidence(NamedTuple):
    """Decoded evidence: border rows exact, interior rows sampled."""

    masks: Array
    centered: Array
    sizes: Array
    n_deterministic: int
    value_empty: Array


def _coalition_rows(draws: Array) -> Array:
    """Enter drawn coalitions directly as single-row units."""
    return draws[..., None, :]


def _border_masks(n_players: int, order: int) -> Array | None:
    """Enumerate the border sizes: below the order and above ``n - order``.

    These sizes carry most of the discrete-derivative weight, so they are
    evaluated deterministically in the prelude — a budget-independent
    stand-in for v1's budget-grown exact region. The empty and grand
    coalition live in the seed block, not here.
    """
    sizes = chain(range(1, order), range(n_players - order + 1, n_players))
    rows = [
        _mask_row(n_players, members)
        for size in sizes
        for members in combinations(range(n_players), size)
    ]
    if not rows:
        return None
    return jnp.asarray(np.array(rows, dtype=bool))


def _mask_row(n_players: int, members: tuple[int, ...]) -> np.ndarray:
    row = np.zeros(n_players, dtype=bool)
    row[list(members)] = True
    return row


def _standard_form_table(
    index: CardinalInteractionIndex,
    n_players: int,
    interaction_size: int,
) -> Array:
    """Build the standard-form weight table ``weight [|W|, |W ∩ T|]``.

    ``weight(w, i) = (-1) ** (t - i) * omega_t(w - i)`` with ``omega_t`` the
    index's discrete-derivative weights over the size of the coalition
    outside the interaction (Fumagalli et al. 2023, Theorem 1); entries outside the feasible intersection range
    are zero.
    """
    omega = np.asarray(
        index.derivative_weights(n_players, interaction_size),
        dtype=np.float64,
    )
    table = np.zeros((n_players + 1, interaction_size + 1))
    for coalition_size in range(n_players + 1):
        lower = max(0, interaction_size + coalition_size - n_players)
        upper = min(interaction_size, coalition_size)
        for intersection_size in range(lower, upper + 1):
            table[coalition_size, intersection_size] = (
                (-1.0) ** (interaction_size - intersection_size)
                * omega[coalition_size - intersection_size]
            )
    return jnp.asarray(table)


def _stratum_weight_table(
    index: CardinalInteractionIndex,
    n_players: int,
    interaction_size: int,
) -> np.ndarray:
    """Per-stratum weights: ``weight (w, i) * binom(n - t, w - i)`` on interior sizes.

    A stratum is an exact intersection pattern (rank ``r`` with
    ``i = popcount(r)``) and a coalition size ``w``; its true size is
    ``binom(n - t, w - i)``, which multiplies the stratum's empirical mean.
    Border sizes enter the exact block instead, so their strata weigh zero.
    """
    omega = np.asarray(
        index.derivative_weights(n_players, interaction_size),
        dtype=np.float64,
    )
    weights = np.zeros((2**interaction_size, n_players + 1))
    for pattern in range(2**interaction_size):
        intersection_size = pattern.bit_count()
        for coalition_size in range(interaction_size, n_players - interaction_size + 1):
            if not 0 < coalition_size < n_players:
                continue
            outside = coalition_size - intersection_size
            if outside < 0 or outside > n_players - interaction_size:
                continue
            weights[pattern, coalition_size] = (
                (-1.0) ** (interaction_size - intersection_size)
                * omega[outside]
                * comb(n_players - interaction_size, outside)
            )
    return weights


def _row_term_weights(decoded: _SizeLawEvidence, members: Array, table: Array) -> Array:
    """Gather ``weight (|W|, |W ∩ T|)`` for every stored row and interaction."""
    one_hot = jnp.zeros((members.shape[0], decoded.masks.shape[-1]), dtype=decoded.masks.dtype)
    one_hot = one_hot.at[jnp.arange(members.shape[0])[:, None], members].set(True)
    intersections = jnp.einsum(
        "...mn,dn->...md",
        decoded.masks.astype(jnp.int32),
        one_hot.astype(jnp.int32),
    )
    return table[decoded.sizes[..., None], intersections]


def _scatter_bins(values: Array, bins: Array, n_interactions: int, n_bins: int) -> Array:
    """Sum per-row values into per-interaction strata bins.

    The last two axes of ``values``/``bins`` are (rows, interactions); the
    result replaces them with (interactions, bins) so each interaction's
    strata can be reduced independently.
    """
    lead = values.shape[:-2]
    n_rows = values.shape[-2]
    flat_values = jnp.moveaxis(values, -1, -2).reshape(-1, n_rows)
    flat_bins = jnp.moveaxis(bins, -1, -2).reshape(-1, n_rows)
    rows = jnp.arange(flat_values.shape[0])[:, None]
    binned = (
        jnp.zeros((flat_values.shape[0], n_bins), dtype=values.dtype)
        .at[rows, flat_bins]
        .add(flat_values)
    )
    return binned.reshape(*lead, n_interactions, n_bins)
