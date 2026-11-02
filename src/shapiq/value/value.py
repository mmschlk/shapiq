"""Context-independent representation contracts for game values.

Shapes describe logical batches. Concrete representations hide and validate
internal value axes rather than including them in these annotations.
"""

from typing import Any, Protocol, runtime_checkable

import jax
from jaxtyping import Float, Num
from ndlike.jax_like import JaxLike


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


@runtime_checkable
class GenericModel[ModelInputT, V: Value](Protocol):
    """A shaped model whose internal structure is hidden from its logical batch."""

    def __call__(self, *args: object, **kwargs: object) -> V:
        """Return a generic value whose logical batch shape is broadcasted from the inputs."""
        ...


# Assignment aliases are also understood by the test suite's typeguard 2.x.
Value = GenericValue
Model = GenericModel
NumericValue = Num[GenericValue, "..."]
# The concrete vector representation owns its protected axis and vector width.
FloatVectorValue = Float[GenericValue, "..."]
JaxValue = jax.Array | JaxLike[Any]
