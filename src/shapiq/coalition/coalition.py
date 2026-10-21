"""Data structure for representing a coalition in a game."""

from __future__ import annotations

from typing import Any

from jax import Array
from jaxtyping import Bool

type FiniteCoalition = Bool[Array, "*batch n_players"]
# Each Players[C] defines the admissible coalition representation C.
type Coalition = Any
