"""The structure of games explaining machine learning models.

- ``XAIGame``: a model and players; how coalitions are evaluated is left open.
- ``LocalXAIGame``: explains rows of ``x_explain`` against a ``background``, both kept in the
  model's own format, e.g. a DataFrame; how coalitions are evaluated is still left open.

``predict`` calls a model and returns values through ``as_value``: a model with one output per
input gives scalar values, a model with a vector of outputs, e.g. class probabilities, gives
``VectorValue`` values. Games over JAX data and JAX models trace, e.g. under ``jit``; other data
and models, e.g. NumPy, pandas, scikit-learn or torch, are called eagerly and raise under tracing.
How to call a model efficiently is up to its ``predict`` registration.
"""

from abc import ABC
from dataclasses import dataclass, field
from math import prod
from typing import Any, ClassVar, cast

import jax
import jax.numpy as jnp
import numpy as np
from flextype import flexdispatch
from jax.core import Tracer
from jaxtyping import Bool, Float, Int, Shaped
from ndlike.array_like import NumpyLike
from ndlike.jax_like import JaxLike
from ndlike.protected_axis.jax import JaxAxisProtected

from shapiq.game.finite import BoolFinitePlayerGame, _FiniteGame
from shapiq.masker.base import require_eager
from shapiq.players import FinitePlayers
from shapiq.typing import HasShape, typechecked
from shapiq.value import VectorValue
from shapiq.value.value import JaxValue


@flexdispatch
def as_value(
    outputs: Any,  # noqa: ANN401 - dispatches on any output type
) -> Float[JaxValue, " n_samples"]:
    """Return a model's outputs on ``n_samples`` inputs as one value per input.

    Array-likes give a scalar per input, or a ``VectorValue`` of each input's outputs; values such
    as a ``VectorValue`` pass unchanged. Register an output type to convert it.

    Raises:
        TypeError: The outputs have no registered conversion.
    """
    msg = (
        f"Model outputs of type {type(outputs).__qualname__} have no conversion to values; "
        "register one with `as_value.register`."
    )
    raise TypeError(msg)


def _require_floats(dtype: np.dtype) -> None:
    """Raise unless the model's outputs are floats.

    Raises:
        TypeError: The outputs are no floats, e.g. class labels.
    """
    if not jnp.issubdtype(dtype, jnp.floating):
        msg = f"The model returns {dtype} outputs, e.g. class labels, not float values."
        raise TypeError(msg)


@as_value.register(JaxLike)
def _(outputs: JaxLike[Any]) -> Float[JaxValue, " n_samples"]:
    """Return values, e.g. a ``VectorValue``, unchanged."""
    return outputs


# A JAX array is also a JaxLike, but no value yet; a traced array is no ``jax.Array`` subclass.
@as_value.register(jax.Array | Tracer)
def _(outputs: jax.Array) -> Float[JaxValue, " n_samples"]:
    """Return a scalar per entry of a 1-D array, a ``VectorValue`` per row of a 2-D one.

    Raises:
        ValueError: An input has more than one vector of outputs.
    """
    _require_floats(outputs.dtype)
    if outputs.ndim == 1:
        return outputs
    if outputs.ndim == 2:
        return VectorValue(outputs)
    msg = f"The model returns outputs of shape {outputs.shape[1:]} per input, not one vector."
    raise ValueError(msg)


@as_value.register(NumpyLike)
def _(outputs: NumpyLike[Any]) -> Float[JaxValue, " n_samples"]:
    """Return the values of a NumPy array, checked first: ``jnp.asarray`` fails unclearly on strings."""
    _require_floats(outputs.dtype)
    return as_value(jnp.asarray(outputs))


@flexdispatch
def predict(
    model: Any,  # noqa: ANN401 - dispatches on any model type
    x: Any,  # noqa: ANN401 - inputs in the model's own format
) -> Float[JaxValue, " n_samples"]:
    """Return the model's value on every row of ``x``.

    The default calls the model on ``x``. Register a model type to call it differently;
    ``as_value`` converts the outputs.
    """
    return as_value(model(x))


@predict.register("sklearn.base.BaseEstimator")
def _(model: Any, x: Any) -> Float[JaxValue, " n_samples"]:  # noqa: ANN401
    """Return the class probabilities, else the decision scores, else the predictions.

    Asks the model, not its type: a pipeline or search over a classifier is no ``ClassifierMixin``,
    and not every classifier has ``predict_proba``, e.g. ``LinearSVC``.
    """
    require_eager("scikit-learn models", x)
    if hasattr(model, "predict_proba"):
        return as_value(model.predict_proba(x))
    if hasattr(model, "decision_function"):
        return as_value(model.decision_function(x))
    return as_value(model.predict(x))


# Optional backends register in their own modules, imported once their objects first arrive.
@as_value.delayed_register("torch.Tensor")
@predict.delayed_register("torch.nn.Module")
def _(cls: type) -> None:  # noqa: ARG001
    import shapiq.xai.torch  # noqa: F401, PLC0415


@as_value.delayed_register("pandas.Series | pandas.DataFrame")
def _(cls: type) -> None:  # noqa: ARG001
    import shapiq.xai.pandas  # noqa: F401, PLC0415


@typechecked
@dataclass(frozen=True, slots=True)
class XAIGame[P, OwnerT: JaxLike[Any] = "XAIGame[Any, Any]"](
    BoolFinitePlayerGame[P, Float[JaxValue, "..."]],
    JaxAxisProtected[OwnerT, Any, _FiniteGame[P]],
    ABC,
):
    """A game explaining a model; ``_eval_bool`` is left to the explanation.

    ``model`` is static metadata, shared by the batch.
    """

    model: object
    # A required slot replaces the inherited abstract players property.
    players: Shaped[FinitePlayers[P], "#*batch"] = field()
    protected_axes: ClassVar[dict[str, int]] = {"players": 0}
    broadcasted_batch: ClassVar[set[str]] = {"players"}


def batch_index(data: HasShape, input_ndim: int) -> Int[jax.Array, "*batch"]:
    """Return the position of every batch entry of ``data``: its axes before the last ``input_ndim``."""
    batch = tuple(data.shape[: len(data.shape) - input_ndim])
    return jnp.arange(prod(batch)).reshape(batch)


def check_input_shape(name: str, data: HasShape, input_shape: tuple[int, ...]) -> None:
    """Raise unless the last axes of ``data`` are ``input_shape``.

    Raises:
        ValueError: The last axes of ``data`` are not ``input_shape``.
    """
    shape = tuple(data.shape)
    if len(shape) < len(input_shape) or shape[len(shape) - len(input_shape) :] != input_shape:
        msg = f"The shape {shape} of {name} does not end with the input shape {input_shape}."
        raise ValueError(msg)


@typechecked
@dataclass(frozen=True, slots=True)
class LocalXAIGame[In: HasShape, P, OwnerT: JaxLike[Any] = "LocalXAIGame[Any, Any, Any]"](
    XAIGame[P, OwnerT], ABC
):
    """Explain the inputs ``x_explain`` against a ``background``; ``_eval_bool`` is still open.

    Both datasets stay in the model's own format ``In``, e.g. a tensor or a DataFrame, as static
    metadata. ``x_explain`` has the shape ``(*batch, *input_shape)``: the axes before the
    ``input_shape`` are games of the batch, e.g. images ``(n, 3, height, width)`` with the input
    shape ``(3, height, width)``. The ``background`` is what absent inputs are compared against:
    inputs to fill them from, or the context of an in-context learner.

    ``explain_index`` is the position of every game's input in the batch of ``x_explain``, by
    default every input once. Indexing and broadcasting games act on it, never on the data.
    """

    x_explain: In
    background: In
    input_shape: tuple[int, ...]
    explain_index: Int[jax.Array, "#*batch"] | None = field(default=None, kw_only=True)
    protected_axes: ClassVar[dict[str, int]] = {"explain_index": 0}
    broadcasted_batch: ClassVar[set[str]] = {"explain_index"}

    def __post_init__(self) -> None:
        """Check the input shape and index every explained input by default.

        Raises:
            ValueError: The last axes of ``x_explain`` are not ``input_shape``.
        """
        self._index_explained()

    def _index_explained(self) -> None:
        """Check the input shape of ``x_explain`` and index its batch, unless given.

        Raises:
            ValueError: The last axes of ``x_explain`` are not ``input_shape``.
        """
        check_input_shape("x_explain", self.x_explain, self.input_shape)
        if self.explain_index is None:
            index = batch_index(self.x_explain, len(self.input_shape))
            object.__setattr__(self, "explain_index", index)

    def _flatten(
        self, coalition: Bool[jax.Array, "#*batch n_players"], *indices: jax.Array | None
    ) -> tuple[Bool[jax.Array, "n n_players"], list[Int[jax.Array, " n"]], tuple[int, ...]]:
        """Return every evaluation's coalition and positions in ``indices``, and the batch shape.

        The game and coalition batches broadcast to one shape, merged into ``n`` evaluations:
        models take a flat batch of inputs.
        """
        shape = jnp.broadcast_shapes(self.shape, coalition.shape[:-1])
        coalitions = jnp.broadcast_to(coalition, (*shape, self.n_players)).reshape(
            -1, self.n_players
        )
        flat = [jnp.broadcast_to(cast("jax.Array", index), shape).reshape(-1) for index in indices]
        return coalitions, flat, shape
