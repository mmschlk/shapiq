"""Tests for chunked evaluation as a game transformer."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from shapiq import SV, CallableGame, ChunkedGame, Regression

N_PLAYERS = 5
WEIGHTS = jnp.asarray([0.7, -1.3, 0.1, 2.0, -0.4])
PAIRS = jnp.asarray(
    [
        [0.0, 0.5, -1.0, 0.0, 0.3],
        [0.5, 0.0, 0.2, -0.7, 0.0],
        [-1.0, 0.2, 0.0, 0.4, 0.9],
        [0.0, -0.7, 0.4, 0.0, -0.2],
        [0.3, 0.0, 0.9, -0.2, 0.0],
    ],
)


def quadratic_from_masks(masks):
    return masks @ WEIGHTS + 0.5 * jnp.einsum("...i,ij,...j->...", masks, PAIRS, masks)


def quadratic_game():
    return CallableGame(
        fn=lambda c: quadratic_from_masks(jnp.asarray(c.to_dense(), dtype=jnp.float32)),
        n_players=N_PLAYERS,
    )


def shape_recording_game(shapes):
    def fn(coalitions):
        masks = jnp.asarray(coalitions.to_dense(), dtype=jnp.float32)
        shapes.append(tuple(masks.shape))
        return quadratic_from_masks(masks)

    return CallableGame(fn=fn, n_players=N_PLAYERS)


def all_masks(n_samples, *, seed=0):
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.random((n_samples, N_PLAYERS)) < 0.5)


@pytest.mark.parametrize("n_samples", [3, 8, 13])
def test_chunked_values_match_the_unwrapped_game(n_samples):
    # below one block, exactly one block, and a padded tail block
    masks = all_masks(n_samples)
    plain = quadratic_game()(masks)
    chunked = ChunkedGame(quadratic_game(), batch_size=8)(masks)
    assert chunked.shape == plain.shape
    assert bool(jnp.all(chunked == plain))


def test_the_wrapped_game_only_sees_full_blocks():
    shapes = []
    game = ChunkedGame(shape_recording_game(shapes), batch_size=4)
    game(all_masks(10))
    # 10 samples become three canonical blocks; the tail is padded, never ragged
    assert shapes == [(4, N_PLAYERS)] * 3


def test_metadata_and_algebra_ride_through():
    game = ChunkedGame(quadratic_game(), batch_size=4)
    assert game.n_players == N_PLAYERS
    assert game.value_shape == ()
    residual = game - quadratic_game()
    masks = all_masks(6)
    assert bool(jnp.all(jnp.abs(residual(masks)) < 1e-6))


def test_deduplicated_split_replay_is_bit_identical_through_the_wrapper():
    def policy():
        return Regression(
            ChunkedGame(quadratic_game(), batch_size=4),
            SV(),
            random_state=3,
            deduplicate=True,
        )

    whole = policy().estimate(24)
    split = policy().refine(policy().estimate(11), 13)
    assert bool(jnp.all(whole.evidence.values == split.evidence.values))
    assert bool(
        jnp.all(whole.evidence.coalitions.to_dense() == split.evidence.coalitions.to_dense()),
    )
