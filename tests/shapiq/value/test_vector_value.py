"""Vector values through SHAPIQ: every output is estimated as its own scalar game would be."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ndlike.tuple.jax import JaxBroadcastTuple

from shapiq.approximator.base import SHAPIQ
from shapiq.game import DenseCoalitionGame, MappedGame
from shapiq.index import ShapleyInteractionIndex, ShapleyValueIndex
from shapiq.players import FinitePlayers
from shapiq.sampling._base import PairedSizeWithoutReplacement, SizeWithReplacement
from shapiq.value import VectorValue

N_PLAYERS = 5


@pytest.mark.parametrize(
    "sampler_type", [PairedSizeWithoutReplacement, SizeWithReplacement], ids=["draws", "estimates"]
)
@pytest.mark.parametrize(
    "index", [ShapleyInteractionIndex(max_order=2), ShapleyValueIndex()], ids=["sii", "sv"]
)
def test_outputs_match_scalar_games(sampler_type, index):
    """Does every output of a vector-valued game get the estimate of its scalar game, step by step?"""
    tables = [
        DenseCoalitionGame(
            jax.random.normal(jax.random.key(k), (2, 2**N_PLAYERS)), FinitePlayers(N_PLAYERS)
        )
        for k in range(3)
    ]
    # The tables' values are plain arrays: stack them into one vector per coalition.
    vector = MappedGame(
        JaxBroadcastTuple(tuple(tables)),
        lambda vs: VectorValue(jnp.stack([jnp.asarray(v) for v in vs], -1)),
    )

    def approximator(game):
        sampler = sampler_type(keys=jax.random.key(7), weights=jnp.ones(N_PLAYERS + 1))
        return SHAPIQ(sampler=sampler, game=game, index=index)

    joint, singles = approximator(vector), [approximator(table) for table in tables]
    coalition = jnp.array([True, False, True, True, False])
    for _ in range(3):
        estimate, joint = joint(6)
        assert isinstance(estimate.values, VectorValue)
        for k, single in enumerate(singles):
            expected, singles[k] = single(6)
            np.testing.assert_allclose(estimate.values.values[..., k], expected.values, atol=1e-5)
            np.testing.assert_allclose(
                estimate(coalition).values[..., k], expected(coalition), atol=1e-5
            )
