"""Representation aliases do not silently relate independent dimensions."""

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Shaped, TypeCheckError

from shapiq.coalition.coalition import BoolCoalition, PackedBoolCoalition
from shapiq.typing import typechecked
from shapiq.value import FloatVectorValue, GenericValue, NumericValue, Value


@typechecked
def _independent_coalitions(left: BoolCoalition, right: BoolCoalition) -> BoolCoalition:
    return right


@typechecked
def _independent_packed_coalitions(
    left: PackedBoolCoalition, right: PackedBoolCoalition
) -> PackedBoolCoalition:
    return right


@typechecked
def _independent_values(left: NumericValue, right: NumericValue) -> NumericValue:
    return right


@typechecked
def _same_batch[V: Value](
    left: Shaped[V, "*batch"], right: Shaped[V, "*batch"]
) -> Shaped[V, "*batch"]:
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


def test_numeric_alias_has_no_implicit_batch_relationship():
    right = jnp.ones((4, 5))
    assert _independent_values(jnp.zeros((2,)), right) is right


@dataclass
class _DuckValue:
    shape: tuple[int, ...]
    dtype: str = "float32"


@pytest.mark.parametrize("value", [jnp.zeros((2, 3)), np.zeros((2, 3)), _DuckValue((2, 3))])
def test_values_support_shaped_backends_and_structured_representations(value):
    assert isinstance(value, GenericValue)
    assert isinstance(value, Value)
    assert isinstance(value, NumericValue)
    assert isinstance(value, FloatVectorValue)
    assert _same_batch(value, value) is value


@pytest.mark.parametrize("value", [2, [2], {"count": jnp.array(2)}])
def test_values_require_shape_and_dtype(value):
    assert not isinstance(value, Value)


def test_local_shape_annotations_still_require_equal_batches():
    with pytest.raises(TypeCheckError):
        _same_batch(jnp.zeros((2,)), jnp.zeros((3,)))


def test_generic_value_shape_checks_use_the_bound_without_nested_variadics():
    annotation = Shaped[Value, "2 3"]
    assert isinstance(jnp.zeros((2, 3)), annotation)
    assert not isinstance(jnp.zeros((3, 2)), annotation)
