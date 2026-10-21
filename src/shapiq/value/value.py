"""Datatypes for representing values in a game."""

from __future__ import annotations

from typing import Any

import jax
from jaxtyping import Float, Num

type GenericValue = Any
type NumericValue = Num[jax.Array, "*batch"]
type FloatVectorValue = Float[jax.Array, "*batch val_dim"]
type Value = GenericValue | NumericValue | FloatVectorValue
