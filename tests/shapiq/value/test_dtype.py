"""JAX promotion over lazy value specs."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from shapiq.value import VectorValue
from shapiq.value._dtype import (
    is_numeric_dtype,
    match_promoted_dtype,
    op_spec,
    promote_spec,
    resolve_weight_dtype,
    unit_scalar,
    validate_numeric_dtype,
    validate_numeric_value,
    value_spec,
)


def test_value_spec_reads_metadata_without_evaluation():
    class Lazy:
        def _value_spec(self):
            return jax.ShapeDtypeStruct((), jnp.float16, weak_type=False)

        def __call__(self):
            pytest.fail("dtype inference must not evaluate the game")

    spec = value_spec(Lazy())
    assert spec == jax.ShapeDtypeStruct((), jnp.float16)
    assert value_spec(1.0) == value_spec(jnp.full((2,), 1.0))
    assert value_spec(np.float16(1)) == spec
    assert value_spec(object()) is None
    assert promote_spec((spec, object()), jnp.float32).dtype == jnp.float32

    for op in (jnp.add, jnp.true_divide, jnp.power):
        assert op_spec(op, Lazy(), 2.0) == value_spec(op(jnp.float16(1), 2.0))


def test_numeric_validation_and_matching_follow_jax():
    assert is_numeric_dtype(jnp.bfloat16)
    assert validate_numeric_dtype("float16", "value") == jnp.float16
    for invalid in (None, bool, object, "not-a-dtype"):
        assert not is_numeric_dtype(invalid)
    with pytest.raises(TypeError, match="must have a numeric dtype"):
        validate_numeric_value(jnp.ones((), dtype=bool), "value")
    with pytest.raises(TypeError, match="must be a numeric dtype"):
        validate_numeric_dtype(bool, "value")

    assert unit_scalar(-1, jnp.float16).dtype == jnp.float16
    assert match_promoted_dtype(jnp.asarray(1.0), jnp.float16).dtype == jnp.float16


@pytest.mark.parametrize("dtype", [np.timedelta64, np.longdouble])
def test_numeric_dtype_validation_rejects_dtypes_unsupported_by_jax(dtype):
    if dtype is np.longdouble and np.dtype(dtype) == np.dtype(np.float64):
        pytest.skip("NumPy longdouble aliases float64 on this platform")
    with pytest.raises(TypeError, match="numeric dtype supported by JAX"):
        validate_numeric_dtype(dtype, "value")


def test_match_promoted_dtype_strengthens_vector_values():
    value = VectorValue(jnp.full((2,), 1.0))
    assert value.weak_type
    strengthened = match_promoted_dtype(value, jnp.zeros((), dtype=value.dtype))
    assert strengthened.dtype == value.dtype
    assert not strengthened.weak_type
    assert jnp.array_equal(strengthened.values, value.values)


@pytest.mark.parametrize("x64", [False, True])
def test_weight_dtype_promotion_uses_a_weak_float(x64):
    with jax.enable_x64(new_val=x64):
        values = jnp.ones(2, dtype=jnp.float16)
        assert resolve_weight_dtype(values, 1, 2.0, jnp.asarray(1.0)) == jnp.float16
        assert resolve_weight_dtype(values, np.array(1, dtype=np.float32)) == jnp.float32
        assert resolve_weight_dtype(jnp.int8(1)) == jnp.asarray(0.0).dtype
