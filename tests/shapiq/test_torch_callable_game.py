"""Tests for the torch call policy on coalition callables."""

from __future__ import annotations

import inspect

import jax
import jax.numpy as jnp
import pytest

torch = pytest.importorskip("torch")

from shapiq import SV, CallableGame, DenseCoalitionArray, Regression  # noqa: E402
from shapiq.games.torch import TorchCallableGame  # noqa: E402

N_PLAYERS = 4
WEIGHTS = [1.0, 2.0, 3.0, 4.0]


def all_coalitions():
    rows = [[bool(mask >> player & 1) for player in range(N_PLAYERS)] for mask in range(16)]
    return DenseCoalitionArray(jnp.asarray(rows))


def test_the_wrapped_callable_runs_without_autograd():
    grad_states = []
    weight = torch.tensor(WEIGHTS, requires_grad=True)

    def scorer(coalitions):
        grad_states.append(torch.is_grad_enabled())
        return (coalitions.to(torch.float32) * weight).sum(dim=-1)

    game = TorchCallableGame(fn=scorer, n_players=N_PLAYERS)
    values = game(DenseCoalitionArray(jnp.asarray([[True, False, True, False]])))
    assert grad_states == [False]
    assert isinstance(values, jax.Array)
    assert jnp.allclose(values, 4.0)


def test_coalitions_arrive_as_boolean_torch_tensors():
    received = {}

    def scorer(coalitions):
        received["type"] = type(coalitions)
        received["dtype"] = coalitions.dtype
        return coalitions.sum(dim=-1, dtype=torch.float32)

    TorchCallableGame(fn=scorer, n_players=N_PLAYERS)(all_coalitions())
    assert received["type"] is torch.Tensor
    assert received["dtype"] == torch.bool


def test_opting_out_of_no_grad_still_detaches_for_conversion():
    grad_states = []
    weight = torch.tensor(WEIGHTS, requires_grad=True)

    def scorer(coalitions):
        grad_states.append(torch.is_grad_enabled())
        return (coalitions.to(torch.float32) * weight).sum(dim=-1)

    game = TorchCallableGame(fn=scorer, n_players=N_PLAYERS, no_grad=False)
    values = game(DenseCoalitionArray(jnp.asarray([[True, True, False, False]])))
    assert grad_states == [True]  # the graph is built, then detached at the boundary
    assert jnp.allclose(values, 3.0)


@pytest.mark.filterwarnings("ignore::shapiq.errors.SamplingStallWarning")
def test_sampling_explainers_consume_the_torch_game():
    weight = torch.tensor(WEIGHTS)

    def scorer(coalitions):
        return (coalitions.to(torch.float32) * weight).sum(dim=-1)

    game = TorchCallableGame(fn=scorer, n_players=N_PLAYERS)
    approximator = Regression(game, SV(), random_state=0, deduplicate=True)
    explanation = approximator.estimate(SEEDS_AND_ROOM := 16)
    assert approximator.min_budget <= SEEDS_AND_ROOM
    for player, expected in enumerate(WEIGHTS):
        assert jnp.allclose(explanation[(player,)], expected, atol=1e-4)


def test_signatures_align_with_the_base_game():
    base = inspect.signature(CallableGame)
    extended = inspect.signature(TorchCallableGame)
    for name, parameter in base.parameters.items():
        assert extended.parameters[name].kind == parameter.kind
