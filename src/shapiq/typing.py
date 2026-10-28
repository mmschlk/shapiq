"""Selective runtime type and shape checking for public boundaries."""

from beartype import beartype
from jaxtyping import jaxtyped

# pytest --typeguard replaces this before collection for exhaustive checks.
typechecked = jaxtyped(typechecker=beartype)
