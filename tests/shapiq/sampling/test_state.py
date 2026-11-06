"""Sampling states are prefixes of their steps and can be cut at any sample count."""

import jax
import jax.numpy as jnp
import pytest

from shapiq.errors import HistoryError
from shapiq.players import FinitePlayers
from shapiq.sampling import Sample, SamplingState

N_PLAYERS = 3
PLAYERS = FinitePlayers(N_PLAYERS)
KEYS = jax.random.split(jax.random.key(0), 6)
# Two games share their samples (batch 1) but get their own values (batch 2).
STEPS = [
    (
        Sample(
            coalitions=jax.random.bernoulli(KEYS[3 * i], shape=(1, m, N_PLAYERS)),
            log_weights=jax.random.normal(KEYS[3 * i + 1], (1, m)),
            players=PLAYERS,
            unit=2,
        ),
        jax.random.normal(KEYS[3 * i + 2], (2, m)),
    )
    for i, m in enumerate((100, 200))
]


def _state() -> SamplingState:
    state = SamplingState(players=PLAYERS)
    for sample, values in STEPS:
        state = state.append(sample, values)
    return state


def test_steps_concatenate_and_cut_across_a_step_boundary():
    state = _state()
    values = jnp.concatenate([values for _, values in STEPS], axis=-1)
    coalitions = jnp.concatenate([sample.coalitions for sample, _ in STEPS], axis=-2)
    log_weights = jnp.concatenate([sample.log_weights for sample, _ in STEPS], axis=-1)
    cut = state.head(150)

    assert state.history_n_samples == (100, 300)
    assert state.steps[0].shape == (2,)
    assert state.steps[0][1].coalitions.shape == (100, N_PLAYERS)
    assert jnp.array_equal(state.values, values)
    assert jnp.array_equal(state.coalitions, coalitions)
    assert jnp.array_equal(state.log_weights, log_weights)
    assert cut.history_n_samples == (100, 150)
    assert jnp.array_equal(cut.values, values[:, :150])
    assert jnp.array_equal(cut.coalitions, coalitions[:, :150])
    assert jnp.array_equal(cut.log_weights, log_weights[:, :150])


def test_rollback_and_history_are_the_step_prefixes():
    state = _state()

    assert state.rollback().history_n_samples == (100,)
    assert state.rollback(2).n_samples == 0
    assert [s.n_samples for s in state.history()] == [100, 300]
    assert [s.n_samples for s in state.history(reverse=True, include_self=False)] == [100]
    with pytest.raises(HistoryError):
        state.rollback(3)
    with pytest.raises(HistoryError):
        state.head(301)


def test_steps_must_cover_the_same_game_batch_and_keep_whole_units():
    sample, values = STEPS[0]
    state = SamplingState(players=PLAYERS).append(sample, values)

    with pytest.raises(ValueError, match="game batch"):
        state.append(sample, values[:1])
    with pytest.raises(ValueError, match="units of 2"):
        sample.head(99)
