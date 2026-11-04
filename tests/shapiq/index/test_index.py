"""Index construction keeps its orders and weights consistent through replace and jit."""

import dataclasses

import jax
import pytest

from shapiq.game.weights import DDAlphaBetaWeight
from shapiq.index import FaithfulShapleyTopOrderIndex, ShapleyInteractionIndex


def test_top_order_fsii_survives_replace_and_jit_and_rejects_inconsistent_construction():
    index = FaithfulShapleyTopOrderIndex(max_order=2)
    rebuilt = dataclasses.replace(index, weights=index.weights)

    def make(alpha):
        return FaithfulShapleyTopOrderIndex(
            max_order=2, weights=DDAlphaBetaWeight(alpha=alpha, beta=alpha)
        )

    assert (rebuilt.min_order, rebuilt.max_order, rebuilt.weights.alpha) == (2, 2, 2)
    assert jax.jit(lambda i: i.weights.alpha)(index) == 2
    assert jax.jit(lambda a: make(a).weights.beta)(2.0) == 2
    with pytest.raises(ValueError, match="Beta"):
        make(1.0)
    with pytest.raises(ValueError, match="min_order"):
        FaithfulShapleyTopOrderIndex(max_order=2, min_order=1)
    with pytest.raises(ValueError, match="min_order"):
        ShapleyInteractionIndex(min_order=3, max_order=2)
