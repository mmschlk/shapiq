"""Tests pinning the Explainer protocol: vocabulary as a structural type."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from shapiq import (
    SHAPIQ,
    SII,
    SV,
    SVARMIQ,
    CallableGame,
    ExactExplainer,
    Explainer,
    PermutationSampling,
    Regression,
)

N_PLAYERS = 4


def additive_game():
    return CallableGame(
        fn=lambda c: jnp.sum(jnp.asarray(c.to_dense(), dtype=jnp.float32), axis=-1),
        n_players=N_PLAYERS,
    )


@pytest.mark.parametrize(
    "build",
    [
        lambda game: ExactExplainer(game, SV()),
        lambda game: PermutationSampling(game, SII(order=2)),
        lambda game: Regression(game, SV()),
        lambda game: SHAPIQ(game, SII(order=2)),
        lambda game: SVARMIQ(game, SII(order=2)),
    ],
)
def test_every_entry_point_speaks_the_explainer_vocabulary(build):
    explainer = build(additive_game())
    assert isinstance(explainer, Explainer)


def test_third_party_policies_conform_structurally():
    # nothing inherits the protocol: having the members IS being an explainer
    class HomeGrown:
        def __init__(self, game, index) -> None:
            self.game = game
            self.index = index
            self.order = 1

        def estimate(self, budget: int):
            del budget
            return ExactExplainer(self.game, self.index).estimate()

    assert isinstance(HomeGrown(additive_game(), SV()), Explainer)


def test_non_explainers_do_not_pass_as_the_vocabulary():
    assert not isinstance(additive_game(), Explainer)
    assert not isinstance(SV(), Explainer)
