"""Coalition aliases preserve independent batches and representation widths."""

import jax.numpy as jnp
import pytest

from shapiq.coalition.coalition import BoolCoalition, PackedBoolCoalition
from shapiq.typing import typechecked


@typechecked
def _independent_coalitions(left: BoolCoalition, right: BoolCoalition) -> BoolCoalition:
    return right


@typechecked
def _independent_packed_coalitions(
    left: PackedBoolCoalition, right: PackedBoolCoalition
) -> PackedBoolCoalition:
    return right


@pytest.mark.parametrize(
    ("accept", "dtype"),
    [(_independent_coalitions, bool), (_independent_packed_coalitions, jnp.uint8)],
)
def test_coalition_aliases_have_independent_batches_and_widths(accept, dtype):
    left = jnp.zeros((2, 3), dtype=dtype)
    right = jnp.zeros((4, 5, 9), dtype=dtype)
    assert accept(left, right) is right


@pytest.mark.parametrize("alias", [BoolCoalition, PackedBoolCoalition])
def test_coalition_aliases_require_a_representation_axis(alias):
    dtype = bool if alias is BoolCoalition else jnp.uint8
    assert isinstance(jnp.zeros((0,), dtype=dtype), alias)
    assert not isinstance(jnp.zeros((), dtype=dtype), alias)
    assert not isinstance(jnp.zeros((3,), dtype=jnp.float32), alias)
