"""Context-independent representation contracts for game values.

Shapes describe logical batches. Concrete representations hide and validate
internal value axes rather than including them in these annotations.
"""

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import jax
from jaxtyping import Float, Num
from ndlike.jax_like import JaxLike

# Assignment aliases are also understood by the test suite's typeguard 2.x.
# Use subscripted JaxLike[Any] for static typing (so OpValueT defaults to Any)
# and unsubscripted JaxLike at runtime so isinstance(x, Value) works directly.
if TYPE_CHECKING:
    JaxValue = jax.Array | JaxLike[Any]
else:
    JaxValue = jax.Array | JaxLike
Value = JaxValue
NumericValue = Num[JaxValue, "..."]
# The concrete vector representation owns its protected axis and vector width.
FloatVectorValue = Float[JaxValue, "..."]


@runtime_checkable
class GenericModel[ModelInputT, V: Value](Protocol):
    """A shaped model whose internal structure is hidden from its logical batch."""

    def __call__(self, *args: object, **kwargs: object) -> V:
        """Return a game value whose logical batch shape is broadcasted from the inputs."""
        ...


Model = GenericModel
