"""Selective runtime type and shape checking for public boundaries, and shared protocols."""

from typing import Protocol, runtime_checkable

from beartype import beartype
from jaxtyping import jaxtyped

# pytest --typeguard replaces this before collection for exhaustive checks.
typechecked = jaxtyped(typechecker=beartype, include_typechecker_message=True)


@runtime_checkable
class HasShape(Protocol):
    """Data with a ``shape``, e.g. an array, a tensor or a DataFrame."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the data."""
