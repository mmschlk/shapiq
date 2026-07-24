"""Chunked evaluation as a game transformer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

from shapiq._shape import validate_int
from shapiq.coalitions import DenseCoalitionArray
from shapiq.games._base import Game

if TYPE_CHECKING:
    from jax import Array

    from shapiq.coalitions import CoalitionArray


class ChunkedGame(Game["Array"]):
    """A game evaluated in fixed-shape chunks along the sample axis.

    Chunking is an evaluation policy, and the policy is honestly part of
    the game's numeric identity: compiled array backends produce values
    that depend on the batch shape they were computed in, so the same
    coalition evaluated inside differently sized calls can differ in the
    last float32 bits. Wrapping a game fixes the policy — the wrapped
    game only ever sees blocks of exactly ``batch_size`` coalitions (the
    tail block padded with empty-coalition rows and trimmed from the
    result), so a coalition's value no longer depends on how a budget was
    split into calls. The fixed block also bounds peak evaluation size
    for expensive games; exact sweeps and sampling share the wrapper.

    Example:
        >>> game = ChunkedGame(expensive_game, batch_size=64)
        >>> estimate = Regression(game, FSII(order=2), deduplicate=True).estimate(500)
    """

    game: Game[Array]
    batch_size: int

    def __init__(self, game: Game[Array], batch_size: int) -> None:
        """Initialize the transformer without evaluating the game.

        Args:
            game: Game to evaluate in chunks.
            batch_size: Number of coalitions in every block the wrapped
                game sees.
        """
        validate_int("batch_size", batch_size, minimum=1)
        self.game = game
        self.batch_size = batch_size
        self.n_players = game.n_players
        self.target_shape = tuple(game.target_shape)
        self.value_shape = tuple(game.value_shape)

    def _call(self, coalitions: CoalitionArray) -> Array:
        """Evaluate whole blocks of ``batch_size`` coalitions and reassemble."""
        masks = jnp.asarray(coalitions.to_dense())
        n_samples = masks.shape[-2]
        if n_samples == 0:
            return self.game(coalitions)
        remainder = n_samples % self.batch_size
        if remainder:
            padding = jnp.zeros(
                (*masks.shape[:-2], self.batch_size - remainder, masks.shape[-1]),
                dtype=masks.dtype,
            )
            masks = jnp.concatenate([masks, padding], axis=-2)
        sample_axis = -1 - len(self.value_shape)
        blocks = [
            jnp.asarray(self.game(DenseCoalitionArray(masks[..., start : start + self.batch_size, :])))
            for start in range(0, masks.shape[-2], self.batch_size)
        ]
        values = blocks[0] if len(blocks) == 1 else jnp.concatenate(blocks, axis=sample_axis)
        trim = (Ellipsis, slice(0, n_samples), *(slice(None),) * len(self.value_shape))
        return values[trim]

    def __repr__(self) -> str:
        """Return a concise representation."""
        return f"{type(self).__name__}(game={self.game!r}, batch_size={self.batch_size!r})"
