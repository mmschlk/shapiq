"""Recontextualized games: remove absent features from an in-context learner's data."""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, override

import jax
import jax.numpy as jnp
import numpy as np
from flextype import flexdispatch
from jaxtyping import Bool, Float, Int
from ndlike import jax_functions as jf

from shapiq.masker import select
from shapiq.masker.base import require_eager
from shapiq.typing import HasShape, typechecked
from shapiq.value import VectorValue
from shapiq.value.value import JaxValue

from .base import LocalXAIGame, batch_index, check_input_shape, predict


@flexdispatch
def contextualize(model: Any, x: Any, y: Any) -> Any:  # noqa: ANN401 - any model, native data
    """Return ``model`` contextualized on ``x`` and ``y``; by default ``model.fit(x, y)``.

    In-context learners such as TabPFN and TabICL store the data as their context.
    """
    return model.fit(x, y)


@flexdispatch
def prior(model: Any, y: Any) -> Float[JaxValue, ""]:  # noqa: ANN401 - any model and targets
    """Return the model's prediction without features: the mean target, or the class frequencies.

    Models with ``predict_proba`` give class probabilities, so their prior is the frequency of
    every class, in sorted order as ``classes_``; other models give the mean target.
    """
    targets = np.asarray(y)
    if hasattr(model, "predict_proba"):
        classes = np.unique(targets)
        return VectorValue(jnp.asarray((targets[:, None] == classes).mean(axis=0)))
    return jnp.asarray(targets.mean(), dtype=float)


@typechecked
@dataclass(frozen=True, slots=True, weakref_slot=True)
class RecontextualizedGame[In: HasShape, P](LocalXAIGame[In, P, "RecontextualizedGame[In, P]"]):
    """Remove the features outside a coalition from the data and recontextualize the model.

    Remove-and-recontextualize (Rundel et al., 2024) explains in-context learners such as TabPFN:
    the model is contextualized on the ``background`` and its targets ``y_background``, both
    restricted to a coalition's features, and predicts the explained inputs restricted alike.
    The data is tabular, ``input_shape`` is ``(n_features,)``, and the whole ``background`` is
    the context: it adds no games to the batch. One context serves all inputs of a coalition.
    The empty coalition calls no model: its value is the ``prior`` of ``y_background``, the mean
    target or the class frequencies. The model is copied once per evaluation and never changed.
    """

    y_background: Any = field(kw_only=True)

    def __post_init__(self) -> None:
        """Check that the data is tabular.

        Raises:
            ValueError: The input shape is not ``(n_features,)``, or the data does not end with it.
        """
        if len(self.input_shape) != 1:
            msg = f"Recontextualization selects columns: the input shape {self.input_shape} must be (n_features,)."
            raise ValueError(msg)
        check_input_shape("background", self.background, self.input_shape)
        super(RecontextualizedGame, self).__post_init__()

    @override
    def _eval_bool(
        self, coalition: Bool[jax.Array, "#*batch {self.n_players}"]
    ) -> Float[JaxValue, "*batch"]:
        """Recontextualize once per distinct coalition and predict its inputs."""
        require_eager("Recontextualization", coalition)  # a fit per distinct coalition
        coalitions, (rows,), shape = self._flatten(coalition, self.explain_index)
        # Group the evaluations by coalition: a context costs a fit, predicting its inputs little.
        distinct, inverse = np.unique(np.asarray(coalitions), axis=0, return_inverse=True)
        inverse = inverse.reshape(-1)
        model = deepcopy(self.model)
        parts, order = [], []
        for k, keep in enumerate(distinct):
            positions = np.flatnonzero(inverse == k)
            parts.append(self._recontextualized(model, jnp.asarray(keep), rows[positions]))
            order.append(positions)
        values = jf.jax_concatenate(parts)[np.argsort(np.concatenate(order))]
        return jf.jax_reshape(values, shape)

    def _recontextualized(
        self, model: object, keep: Bool[jax.Array, " n_features"], rows: Int[jax.Array, " n"]
    ) -> Float[JaxValue, " n"]:
        """Return the values of ``rows`` with the model contextualized on the ``keep`` features."""
        if not keep.any():  # nothing to contextualize on: the prior of every row
            return jf.jax_broadcast_to(prior(model, self.y_background), (rows.size,))
        context_rows = batch_index(self.background, 1).reshape(-1)
        background = select(self.background, context_rows, keep)
        context = contextualize(model, background, self.y_background)
        return predict(context, select(self.x_explain, rows, keep))
