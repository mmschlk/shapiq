"""Vector values: one entry per output, hidden from the logical batch."""

from dataclasses import dataclass
from typing import Any, ClassVar

import jax
from jaxtyping import Float
from ndlike import jax_functions as jf
from ndlike.protected_axis.jax import JaxAxisProtected

from shapiq.typing import typechecked


@typechecked
@dataclass(frozen=True, slots=True, weakref_slot=True)
class VectorValue(JaxAxisProtected["VectorValue"]):
    """A batch of vectors, e.g. the class probabilities of a classifier.

    ``values`` has shape ``(*batch, n_outputs)``. The output axis is protected, so the logical
    shape is the batch: arithmetic, reductions and ``where`` act on the batch and treat every
    output alike.
    """

    values: Float[jax.Array, "*batch n_outputs"]
    protected_axes: ClassVar[dict[str, int]] = {"values": 1}
    permitted_functions: ClassVar[set[Any]] = {
        jf.jax_add,
        jf.jax_subtract,
        jf.jax_multiply,
        jf.jax_true_divide,
        jf.jax_negative,
        jf.jax_sum,
        jf.jax_mean,
        jf.jax_where,
        jf.jax_zeros_like,
    }

    @property
    def n_outputs(self) -> int:
        """Return the number of outputs."""
        return self.values.shape[-1]
