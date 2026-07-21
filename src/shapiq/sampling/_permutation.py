from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import jax
import jax.numpy as jnp

from shapiq.sampling._schedule import UnitScheduleSampler

if TYPE_CHECKING:
    from jax import Array

    from shapiq._shape import ShapeLike
    from shapiq.sampling._base import ShareSamples


class WalkPlan(Protocol):
    """Declaration of the coalitions one permutation materializes into.

    The permutation sampler is a vehicle: it draws permutations, owns the
    emission schedule, and defines pairing as reversal. Which coalitions a
    permutation stands for is the estimator's business, declared as a walk
    plan: ``length`` fixes the sampling quantum, ``render`` turns player
    positions into the walk's coalition masks, and ``prelude`` may extend
    the deterministic seed block after the empty and grand coalition with
    evaluations the estimator needs exactly once (exact lower-order
    anchors). Plans are layout declarations, not samplers — they hold no
    randomness, and the estimator that declares a plan is the one that
    decodes its layout at explain time, so the layout has a single owner.
    """

    n_players: int

    @property
    def length(self) -> int:
        """Return the number of coalitions one walk contains."""
        ...

    def prelude(self) -> Array | None:
        """Return deterministic seed masks following the empty and grand coalition."""
        ...

    def render(self, positions: Array) -> Array:
        """Return the walk masks of one permutation, given player positions."""
        ...


@dataclass(frozen=True)
class ChainPlan:
    """The canonical walk: the proper prefix chain of one permutation.

    Prefixes of sizes one to ``n_players - 1``; together with the seed
    block's empty and grand coalition, one walk carries every marginal
    contribution along its permutation — the Shapley-value walk.
    """

    n_players: int

    @property
    def length(self) -> int:
        """Return the walk length: one proper prefix per size."""
        return self.n_players - 1

    def prelude(self) -> Array | None:
        """Return no prelude; the chain needs only the shared seed block."""
        return None

    def render(self, positions: Array) -> Array:
        """Return the proper prefix masks of one permutation."""
        return positions[..., None, :] < jnp.arange(1, self.n_players)[:, None]


class PermutationSampler(UnitScheduleSampler):
    """Sampler emitting coalition walks derived from random permutations.

    One walk is the coalition block one permutation materializes into. The
    sampler is the vehicle: each walk derives its permutation from
    ``fold_in(random_state, walk_index)``, so sampling does not depend on
    how a budget is split across calls, and pairing means walking the
    reversed permutation (the ``AntitheticDraws`` hook; wrap the sampler in
    ``PairedSampler`` to add the antithesis to every unit). The walk layout
    is the plan's declaration: the default ``ChainPlan`` renders the proper
    prefix chain, and estimators pass the richer layouts they decode,
    including any deterministic prelude extending the seed block.
    """

    plan: WalkPlan

    def __init__(
        self,
        n_players: int,
        target_shape: ShapeLike = (),
        *,
        plan: WalkPlan | None = None,
        share_samples: ShareSamples = False,
        random_state: Array | int = 0,
    ) -> None:
        """Initialize a permutation sampler.

        Args:
            n_players: Number of players in the explained game. Must be at
                least two.
            target_shape: Shape of the explanation targets, matching the
                game's target shape.
            plan: The walk layout to render, covering the same number of
                players. Defaults to the prefix chain.
            share_samples: Policy for sharing sampled coalitions across
                explanation-target axes. ``False`` samples independently per
                target; ``True`` shares across all target axes; an integer or
                tuple of integers shares across the selected axes.
            random_state: Integer seed or JAX PRNG key used to derive one
                permutation per walk.

        Raises:
            ValueError: If ``n_players`` is smaller than two, or if the plan
                lays out a different number of players.
            TypeError: If ``random_state`` is neither an integer nor a JAX
                PRNG key.
        """
        super().__init__(
            n_players,
            target_shape,
            share_samples=share_samples,
            random_state=random_state,
        )
        self.plan = ChainPlan(self.n_players) if plan is None else plan
        if self.plan.n_players != self.n_players:
            msg = (
                f"the walk plan lays out {self.plan.n_players} players but the "
                f"sampler draws permutations of {self.n_players}"
            )
            raise ValueError(msg)
        self._prelude = self.plan.prelude()

    @property
    def sampling_quantum(self) -> int:
        """Return the unit length: one walk."""
        return self.plan.length

    @property
    def n_seed_samples(self) -> int:
        """Return the seed block length, including the plan's prelude."""
        if self._prelude is None:
            return 2
        return 2 + int(self._prelude.shape[-2])

    def _seed_masks(self) -> Array:
        """Return the empty and grand coalition, then the plan's prelude."""
        base = super()._seed_masks()
        if self._prelude is None:
            return base
        return jnp.concatenate([base, self._prelude], axis=-2)

    def _sampled_unit_masks(self, unit_index: int) -> Array:
        """Return the walk masks of one sampled unit."""
        return self.render_draw(self.unit_draw(unit_index))

    def _sampled_unit_batch(self, unit_indices: Array) -> Array:
        """Render many walks in a few vectorized dispatches.

        Draws batch over the unit axis and ``render_draw`` broadcasts over
        leading axes, so the batch is bit-identical to per-unit rendering.
        """
        return self.render_draw(self.unit_draws(unit_indices))

    def unit_draw(self, unit_index: int) -> Array:
        """Return each player's position in the permutation of a walk."""
        players = jnp.broadcast_to(
            jnp.arange(self.n_players),
            (*self.shared_target_shape, self.n_players),
        )
        walk_key = jax.random.fold_in(self._key, unit_index)
        permutation = jax.random.permutation(walk_key, players, axis=-1, independent=True)
        return jnp.argsort(permutation, axis=-1)

    def unit_draws(self, unit_indices: Array) -> Array:
        """Return the draws of many units, stacked on a new leading axis."""
        return jax.vmap(self.unit_draw)(unit_indices)

    def antithetic_draw(self, draw: Array) -> Array:
        """Return the positions of the reversed permutation.

        This is the sampler's ``AntitheticDraws`` hook: pairing a walk means
        walking the reversed permutation, not complementing walk rows.
        """
        return self.n_players - 1 - draw

    def render_draw(self, draw: Array) -> Array:
        """Return the plan's walk masks for one draw of player positions."""
        return self.plan.render(draw)
