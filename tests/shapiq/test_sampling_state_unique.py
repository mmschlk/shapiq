"""Tests for the state-owned unique-coalition view."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from shapiq import SV, CallableGame, Regression
from shapiq.coalitions import DenseCoalitionArray
from shapiq.sampling import SamplingState

N_PLAYERS = 6


def stream(seed: int, n_rows: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    rng = np.random.default_rng(seed)
    masks = rng.random((n_rows, N_PLAYERS)) < 0.5
    values = rng.normal(size=n_rows)
    return jnp.asarray(masks), jnp.asarray(values)


def test_unique_counts_match_the_numpy_oracle():
    masks, values = stream(0, 40)
    state = SamplingState(DenseCoalitionArray(masks), values)
    view = state.unique()
    host_masks = np.asarray(masks)
    expected = {row.tobytes() for row in np.packbits(host_masks, axis=-1)}
    assert len(view.counts) == len(expected)
    assert int(view.counts.sum()) == state.n_samples
    unique_rows = np.asarray(jnp.asarray(view.coalitions.to_dense()))
    # first-occurrence order, a faithful inverse, and aligned first positions
    assert np.all(np.diff(view.first_indices) > 0)
    assert np.array_equal(unique_rows[view.inverse], host_masks)
    assert np.array_equal(host_masks[view.first_indices], unique_rows)


def test_the_view_is_invariant_under_append_splits():
    masks, values = stream(1, 30)
    whole = SamplingState(DenseCoalitionArray(masks), values)
    split = SamplingState(DenseCoalitionArray(masks[:11]), values[:11]).append(
        DenseCoalitionArray(masks[11:]),
        values[11:],
    )
    left, right = whole.unique(), split.unique()
    assert np.array_equal(
        np.asarray(jnp.asarray(left.coalitions.to_dense())),
        np.asarray(jnp.asarray(right.coalitions.to_dense())),
    )
    assert np.array_equal(left.first_indices, right.first_indices)
    assert np.array_equal(left.counts, right.counts)
    assert np.array_equal(left.inverse, right.inverse)


def test_prefix_views_exclude_the_tail():
    masks, values = stream(2, 25)
    state = SamplingState(DenseCoalitionArray(masks), values)
    prefix = state.unique(10)
    assert int(prefix.counts.sum()) == 10
    assert int(prefix.first_indices.max()) < 10
    with pytest.raises(ValueError, match="stores 25"):
        state.unique(26)


def test_unshared_targets_get_the_teaching_error():
    rng = np.random.default_rng(3)
    masks = jnp.asarray(rng.random((2, 12, N_PLAYERS)) < 0.5)
    values = jnp.asarray(rng.normal(size=(2, 12)))
    state = SamplingState(DenseCoalitionArray(masks), values, target_shape=(2,))
    with pytest.raises(ValueError, match="share_samples=True"):
        state.unique()
    with pytest.raises(ValueError, match="share_samples=True"):
        state.key_index()


def test_key_index_agrees_with_the_unique_view():
    masks, values = stream(4, 40)
    state = SamplingState(DenseCoalitionArray(masks), values)
    index = state.key_index()
    view = state.unique()
    # same identity definition: one entry per distinct coalition, mapped to
    # its first stream position, in first-occurrence order
    assert list(index.values()) == list(view.first_indices)
    packed = state.packed_keys()
    for key, position in index.items():
        assert packed[position].tobytes() == key
    assert state.key_index() is index  # computed once per state and cached


def test_deduplicated_sampling_multiplicities_are_visible():
    def fn(coalitions):
        masks = jnp.asarray(coalitions.to_dense(), dtype=jnp.float32)
        return jnp.sum(masks, axis=-1)

    game = CallableGame(fn=fn, n_players=N_PLAYERS)
    budget = 20
    approximator = Regression(game, SV(), random_state=0, share_samples=True, deduplicate=True)
    estimate = approximator.estimate(budget)
    state = estimate.evidence
    assert isinstance(state, SamplingState)
    view = state.unique()
    # with deduplication every spent evaluation bought one distinct coalition,
    # and the stream still records how often each one was drawn
    assert len(view.counts) == budget
    assert int(view.counts.sum()) == state.n_samples
    assert state.n_samples >= budget
