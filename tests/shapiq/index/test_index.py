"""Index construction keeps its orders and weights consistent through replace and jit."""

import dataclasses

import jax
import jax.numpy as jnp
import pytest

from shapiq.game.weights import AlphaBetaWeight
from shapiq.index import FaithfulShapleyTopOrderIndex, ShapleyInteractionIndex


def test_top_order_fsii_derives_its_orders_and_beta_weights_through_replace_and_jit():
    index = FaithfulShapleyTopOrderIndex(max_order=2)
    rebuilt = dataclasses.replace(index, weights=index.weights)
    sizes, order = jnp.arange(4), jnp.asarray(2)
    beta = AlphaBetaWeight(alpha=2.0, beta=2.0).discrete_derivative_weight(sizes, order, 5)
    jitted = jax.jit(lambda i: i.discrete_derivative_weight(sizes, order, 5))(index)

    assert (rebuilt.min_order, rebuilt.max_order) == (2, 2)
    assert jnp.allclose(jitted, beta)
    with pytest.raises(TypeError, match="min_order"):
        FaithfulShapleyTopOrderIndex(max_order=2, min_order=1)
    with pytest.raises(ValueError, match="min_order"):
        ShapleyInteractionIndex(min_order=3, max_order=2)
