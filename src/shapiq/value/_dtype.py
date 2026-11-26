"""Lazy dtype metadata and JAX promotion for values and games."""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Shaped

from .value import JaxValue

if TYPE_CHECKING:
    from jax.sharding import Sharding
    from jax.typing import DTypeLike
    from numpy.typing import DTypeLike as NpDTypeLike


def weak_zero() -> jax.Array:
    """Return a weakly typed zero-dimensional scalar array for default affine bias."""
    return jnp.asarray(0)


def weak_zero_for_dtype(dtype: jnp.dtype) -> jax.Array:
    """Return a weakly typed zero scalar matching ``dtype``'s numeric kind."""
    return jnp.asarray(_weak_zero_literal(dtype))


def _weak_zero_literal(dtype: jnp.dtype) -> int | float | complex:
    """Choose a Python scalar that participates in the same JAX promotion."""
    if jnp.issubdtype(dtype, jnp.complexfloating):
        return 0j
    if jnp.issubdtype(dtype, jnp.floating):
        return 0.0
    return 0


def is_numeric_dtype(dtype: object) -> bool:
    """Return whether ``dtype`` represents a numeric JAX/NumPy dtype."""
    if dtype is None:
        return False
    try:
        return bool(jnp.issubdtype(cast("Any", dtype), jnp.number))
    except TypeError:
        return False


def validate_numeric_value(value: object, name: str) -> None:
    """Validate that an arithmetic parameter carries a numeric dtype."""
    spec = value_spec(value)
    if spec is None or not is_numeric_dtype(spec.dtype):
        msg = f"{name} must have a numeric dtype."
        raise TypeError(msg)


def value_spec(value: object) -> jax.ShapeDtypeStruct | None:
    """Return scalar dtype and weak typing from metadata alone."""
    hook = getattr(value, "_value_spec", None)
    if hook is not None:
        return hook()
    if isinstance(value, (bool, int, float, complex)) and not isinstance(value, np.generic):
        return jax.ShapeDtypeStruct(
            (), jnp.result_type(value), weak_type=not isinstance(value, bool)
        )
    dtype = getattr(value, "dtype", None)
    if isinstance(value, (np.dtype, type)):
        dtype = value
    if dtype is None:
        return None
    try:
        return jax.ShapeDtypeStruct(
            (), jnp.dtype(dtype), weak_type=bool(getattr(value, "weak_type", False))
        )
    except TypeError:
        return None


def require_value_spec(value: object) -> jax.ShapeDtypeStruct:
    """Return scalar dtype metadata or raise if the value has no dtype."""
    spec = value_spec(value)
    if spec is None:
        msg = f"Cannot determine dtype for {type(value).__name__}."
        raise TypeError(msg)
    return spec


def op_spec(fn: Callable[..., object], *operands: object) -> jax.ShapeDtypeStruct:
    """Run a JAX operation on abstract operands and return its scalar output spec.

    Explicit abstract operands keep their shapes, so this also supports
    operations that need an axis. Tuples can pass multiple operands as one argument.
    """

    def abstract(operand: object) -> object:
        if isinstance(operand, jax.ShapeDtypeStruct):
            return operand
        if isinstance(operand, tuple):
            return tuple(abstract(item) for item in operand)
        return require_value_spec(operand)

    return require_value_spec(jax.eval_shape(fn, *(abstract(operand) for operand in operands)))


def validate_numeric_dtype(dtype: object, name: str) -> jnp.dtype:
    """Resolve a numeric dtype supported by JAX."""
    return _resolve_dtype(dtype, name, allow_bool=False)


def _resolve_dtype(dtype: object, name: str, *, allow_bool: bool) -> jnp.dtype:
    """Canonicalize and validate a JAX numeric or boolean cast target."""
    description = "numeric or boolean" if allow_bool else "numeric"
    try:
        if dtype is not None:
            resolved = jax.dtypes.canonicalize_dtype(jnp.dtype(cast("Any", dtype)))
            if is_numeric_dtype(resolved) or (allow_bool and resolved == jnp.bool_):
                # Canonicalization alone accepts unsupported NumPy dtypes like longdouble.
                jax.dtypes.result_type(resolved)
                return resolved
    except (TypeError, ValueError) as error:
        msg = f"{name} must be a {description} dtype supported by JAX."
        raise TypeError(msg) from error
    msg = f"{name} must be a {description} dtype supported by JAX."
    raise TypeError(msg)


def default_float_dtype() -> jnp.dtype:
    """Return the JAX default float under the current x64 setting."""
    return jax.dtypes.canonicalize_dtype(jnp.dtype(float))


def resolve_weight_dtype(values: object, *params: object) -> jnp.dtype:
    """Resolve the dtype for multiplying floating weights by game values."""
    specs = (require_value_spec(value) for value in (values, *params))
    operands = (
        _weak_zero_literal(spec.dtype)
        if spec.weak_type
        else jax.dtypes.canonicalize_dtype(spec.dtype)
        for spec in specs
    )
    return jnp.result_type(*operands, 0.0)


def cast_value[T: JaxValue](
    value: T,
    dtype: "DTypeLike | NpDTypeLike | None",
    *,
    copy: bool,
    device: "jax.Device | Sharding | None",
) -> T:
    """Place a value before casting it, preserving its dtype for ``None``."""
    if device is not None:
        value = cast("T", value.to_device(device))
    if dtype is not None:
        return cast("T", value.astype(cast("DTypeLike", dtype), copy=copy))
    return cast("T", value.copy()) if copy else value


def _fallback_spec(fallback: object) -> jax.ShapeDtypeStruct:
    """Build a conservative fallback ``ShapeDtypeStruct``."""
    spec = value_spec(fallback)
    if spec is not None:
        return spec
    return jax.ShapeDtypeStruct((), cast("jnp.dtype", fallback), weak_type=False)


def _promote_add(*xs: jax.Array) -> jax.Array:
    """Abstract addition chain used with ``jax.eval_shape`` to follow JAX weak-type promotion."""
    acc = xs[0]
    for x in xs[1:]:
        acc = jnp.add(acc, x)
    return acc


def promote_spec(values: tuple[object, ...], fallback: object) -> jax.ShapeDtypeStruct:
    """Compute promoted JAX dtype and ``weak_type`` across operands without evaluating games."""
    specs: list[jax.ShapeDtypeStruct] = []
    for item in values:
        spec = value_spec(item)
        if spec is None:
            return _fallback_spec(fallback)
        specs.append(spec)
    if not specs:
        return _fallback_spec(fallback)
    if len(specs) == 1:
        return specs[0]
    try:
        return op_spec(_promote_add, *specs)
    except TypeError:
        return _fallback_spec(fallback)


def promote_dtype(values: tuple[object, ...], fallback: object) -> jnp.dtype:
    """Compute promoted JAX dtype across operands while preserving weak array/game dtypes."""
    return promote_spec(values, fallback).dtype


def match_promoted_dtype[T: JaxValue](
    value: Shaped[T, "..."], reference: object
) -> Shaped[T, "..."]:
    """Promote ``value`` to match the promoted dtype (and ``weak_type``) with ``reference``."""
    spec = promote_spec((value, reference), getattr(value, "dtype", None))
    if isinstance(value, jax.Array):
        if value.dtype == spec.dtype and value.weak_type == spec.weak_type:
            return value
        if spec.weak_type:
            promoted = value + weak_zero_for_dtype(spec.dtype)
            if promoted.dtype == spec.dtype and promoted.weak_type:
                return cast("Shaped[T, '...']", promoted)
        return cast("Shaped[T, '...']", value.astype(spec.dtype))
    val_spec = require_value_spec(value)
    if (val_spec.dtype, val_spec.weak_type) != (spec.dtype, spec.weak_type):
        return cast("Shaped[T, '...']", value.astype(spec.dtype))
    return value


def unit_scalar(value: int, reference: object) -> jax.Array:
    """Return a dynamic scalar ``0``, ``1``, or ``-1`` preserving ``reference``'s dtype/weakness."""
    spec = value_spec(reference)
    if spec is None or not is_numeric_dtype(spec.dtype):
        return jnp.asarray(value)
    if spec.weak_type:
        if value == -1:
            if jnp.issubdtype(spec.dtype, jnp.complexfloating):
                return jnp.asarray(-1.0 + 0j)
            if jnp.issubdtype(spec.dtype, jnp.floating):
                return jnp.asarray(-1.0)
        return jnp.asarray(value)
    dt = spec.dtype
    # Use ``-ones`` for -1 so unsigned dtypes wrap cleanly without NumPy/JAX conversion errors.
    return -jnp.ones((), dtype=dt) if value == -1 else jnp.full((), value, dtype=dt)
