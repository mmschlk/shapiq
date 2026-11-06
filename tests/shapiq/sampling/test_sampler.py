"""Size samplers batch over keys, advance between steps, and weight draws by 1 / p(S)."""

from math import comb

import jax
import jax.numpy as jnp

from shapiq.game.exact import ExactComputer, discrete_derivative_coefficients
from shapiq.game.table import TableGame
from shapiq.index import ShapleyValueIndex
from shapiq.players import FinitePlayers
from shapiq.sampling import SamplingState
from shapiq.sampling._base import SizeSampler

N_PLAYERS = 5
PLAYERS = FinitePlayers(N_PLAYERS)
GAME = TableGame(jax.random.normal(jax.random.key(0), (2**N_PLAYERS,)), PLAYERS)


def test_keys_are_independent_runs_with_one_size_distribution_and_steps_advance():
    weights = jnp.arange(1.0, N_PLAYERS + 2)  # q(s) grows with s
    sampler = SizeSampler(keys=jax.random.split(jax.random.key(1), 4), weights=weights)
    state = SamplingState(players=PLAYERS)
    sample, successor = sampler.sample(state, 30)
    again, _ = successor.sample(state, 30)
    sizes = jnp.sum(sample.coalitions, axis=-1)  # (4, 30)
    counts = jax.vmap(lambda row: jnp.bincount(row, length=N_PLAYERS + 1))(sizes)
    expected = 30 * weights / jnp.sum(weights)
    q = weights / jnp.sum(weights)
    one_over_p = jnp.array([comb(N_PLAYERS, s) for s in range(N_PLAYERS + 1)]) / q

    assert sample.shape == sampler.shape == (4,)
    assert sample.coalitions.shape == (4, 30, N_PLAYERS)
    assert not jnp.array_equal(sample.coalitions[0], sample.coalitions[1])
    assert not jnp.array_equal(sample.coalitions, again.coalitions)
    # Every size gets its expected count, rounded down or up.
    assert jnp.all((counts >= jnp.floor(expected)) & (counts <= jnp.ceil(expected)))
    assert jnp.allclose(jnp.exp(sample.log_weights), one_over_p[sizes], rtol=1e-5)


def test_weighted_mean_of_independent_runs_is_unbiased_for_the_shapley_value():
    # Every size has q(s) > 0, so the empty and grand coalitions are drawn too.
    n_runs, budget = 4000, 32
    sampler = SizeSampler.uniform_over_sizes(jax.random.split(jax.random.key(2), n_runs), N_PLAYERS)
    sample, sampler = sampler.sample(SamplingState(players=PLAYERS), budget)
    state = SamplingState(players=PLAYERS).append(sample, GAME(sample.coalitions))
    index = ShapleyValueIndex()
    interactions = index.interactions(N_PLAYERS)
    coefficients = jax.vmap(lambda c: discrete_derivative_coefficients(index, c, interactions))(
        state.coalitions
    )  # (runs, n_interactions, budget)
    terms = state.values * jnp.exp(sampler.log_weights(state))  # (runs, budget)
    estimates = jnp.mean(coefficients * terms[:, None, :], axis=-1)  # (runs, n_interactions)
    exact = ExactComputer.from_game(GAME)(index).values
    standard_error = jnp.std(estimates, axis=0) / jnp.sqrt(n_runs)

    assert jnp.all(jnp.abs(jnp.mean(estimates, axis=0) - exact) < 4 * standard_error)
