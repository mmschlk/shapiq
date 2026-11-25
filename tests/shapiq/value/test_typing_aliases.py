"""Value aliases preserve independent shapes and enforce local relationships."""

from dataclasses import dataclass
from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Shaped, TypeCheckError
from ndlike.protected_axis.jax import JaxAxisProtected

from shapiq.typing import typechecked
from shapiq.value import FloatVectorValue, JaxValue, NumericValue, Value


@typechecked
def _independent_values(left: NumericValue, right: NumericValue) -> NumericValue:
    return right


@typechecked
def _same_batch[V: Value](
    left: Shaped[V, "*batch"], right: Shaped[V, "*batch"]
) -> Shaped[V, "*batch"]:
    return right


def test_numeric_alias_has_no_implicit_batch_relationship():
    right = jnp.ones((4, 5))
    assert _independent_values(jnp.zeros((2,)), right) is right


@dataclass
class _DuckValue:
    shape: tuple[int, ...]
    dtype: str = "float32"


@dataclass(frozen=True)
class _StructuredJaxValue(JaxAxisProtected["_StructuredJaxValue"]):
    values: Shaped[jax.Array, "*batch 2"]
    protected_axes: ClassVar[dict[str, int]] = {"values": 1}


@pytest.mark.parametrize(
    "value",
    [jnp.zeros((2, 3)), _StructuredJaxValue(jnp.zeros((2, 3, 2)))],
)
def test_values_support_jax_arrays_and_structured_jax_like_representations(value):
    assert isinstance(value, JaxValue)
    assert isinstance(value, Value)
    assert isinstance(value, NumericValue)
    assert isinstance(value, FloatVectorValue)
    assert _same_batch(value, value) is value


@pytest.mark.parametrize(
    "value",
    [2, [2], {"count": jnp.array(2)}, np.zeros((2, 3)), _DuckValue((2, 3))],
)
def test_values_require_jax_array_or_jax_like(value):
    assert not isinstance(value, Value)
    assert not isinstance(value, JaxValue)


def test_local_shape_annotations_still_require_equal_batches():
    with pytest.raises(TypeCheckError):
        _same_batch(jnp.zeros((2,)), jnp.zeros((3,)))


def test_value_shape_checks_use_the_bound_without_nested_variadics():
    annotation = Shaped[Value, "2 3"]
    assert isinstance(jnp.zeros((2, 3)), annotation)
    assert isinstance(_StructuredJaxValue(jnp.zeros((2, 3, 2))), annotation)
    assert not isinstance(jnp.zeros((3, 2)), annotation)
    assert not isinstance(np.zeros((2, 3)), annotation)
