"""Context-independent representation contracts for game values.

Shapes describe logical batches. Concrete representations hide and validate
internal value axes rather than including them in these annotations.
"""

from typing import Protocol, runtime_checkable

from jaxtyping import Float, Num


@runtime_checkable
class GenericValue(Protocol):
    """A shaped value whose internal structure is hidden from its logical batch."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the logical value batch shape, excluding internal value axes."""
        ...

    @property
    def dtype(self) -> object:
        """Return dtype metadata understood by jaxtyping."""
        ...


# Assignment aliases are also understood by the test suite's typeguard 2.x.
Value = GenericValue
NumericValue = Num[GenericValue, "..."]
# The concrete vector representation owns its protected axis and vector width.
FloatVectorValue = Float[GenericValue, "..."]
