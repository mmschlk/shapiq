"""Regression checks that --typeguard enables automatic runtime shape checking."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import pytest
from jaxtyping import Float, TypeCheckError

pytestmark = pytest.mark.typeguard


def _same_shape(
    left: Float[jax.Array, " length"], right: Float[jax.Array, " length"]
) -> Float[jax.Array, " length"]:
    # The body deliberately does not enforce the relationship itself.
    del right
    return left


def _wrong_return(value: Float[jax.Array, " length"]) -> Float[jax.Array, " length"]:
    return jnp.zeros((value.shape[0] + 1,))


def test_pytest_checks_shared_shape_variables():
    left = jnp.zeros((3,))
    assert _same_shape(left, jnp.ones((3,))) is left
    with pytest.raises(TypeCheckError):
        _same_shape(left, jnp.ones((4,)))


def test_pytest_checks_dtypes():
    with pytest.raises(TypeCheckError):
        _same_shape(jnp.arange(3), jnp.arange(3))


def test_pytest_checks_return_shapes():
    with pytest.raises(TypeCheckError):
        _wrong_return(jnp.zeros((3,)))


def test_pytest_checks_dataclass_field_annotations():
    @dataclass
    class Batch:
        values: Float[jax.Array, "batch features"]

    assert Batch(jnp.zeros((2, 3))).values.shape == (2, 3)
    with pytest.raises(TypeCheckError):
        Batch(jnp.zeros((3,)))
