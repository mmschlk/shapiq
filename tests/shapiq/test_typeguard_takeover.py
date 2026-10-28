"""Typeguard must exhaustively check both automatic and explicit test boundaries."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import pytest
from jaxtyping import Float, TypeCheckError

from shapiq.typing import typechecked

pytestmark = pytest.mark.typeguard


def _ordinary_integers(values: list[int]) -> list[int]:
    return values


@typechecked
def _boundary_integers(values: list[int]) -> list[int]:
    return values


@typechecked
@dataclass
class _IntegerBatch:
    values: list[int]


@typechecked
@dataclass
class _ArrayBatch:
    values: list[Float[jax.Array, " length"]]


@pytest.mark.parametrize("boundary", [_ordinary_integers, _boundary_integers, _IntegerBatch])
@pytest.mark.parametrize("bad_index", range(8))
def test_typeguard_checks_every_container_element(boundary, bad_index):
    values = list(range(8))
    assert boundary(values) is not None
    values[bad_index] = "not an integer"
    with pytest.raises(TypeCheckError):
        boundary(values)


@pytest.mark.parametrize("bad_index", range(8))
def test_typeguard_checks_every_dataclass_array_shape(bad_index):
    values = [jnp.zeros((3,)) for _ in range(8)]
    assert len(_ArrayBatch(values).values) == 8
    values[bad_index] = jnp.zeros((4,))
    with pytest.raises(TypeCheckError):
        _ArrayBatch(values)


def _wrong_container_return() -> list[int]:
    return [0, 1, 2, "not an integer"]


def test_typeguard_checks_container_return_elements():
    with pytest.raises(TypeCheckError, match="return value"):
        _wrong_container_return()
