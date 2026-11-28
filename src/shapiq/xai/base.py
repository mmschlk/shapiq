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
from typing import TYPE_CHECKING, Any, ClassVar, cast, override

import jax
import jax.numpy as jnp
import numpy as np
from flextype import flexdispatch
from jax.core import Tracer
from jaxtyping import Bool, Float, Int, Shaped
from ndlike import jax_functions as jf
from ndlike.array_like import NumpyLike
from ndlike.jax_like import JaxLike
from ndlike.protected_axis.jax import JaxAxisProtected

from shapiq.coalition import FiniteCoalition
from shapiq.game.cast import CastGame
from shapiq.game.finite import CoalitionDefinedGame, _FiniteGame
from shapiq.lazy_types import PANDAS_OUTPUT, SKLEARN_MODULE, TORCH_MODULE, TORCH_TENSOR
from shapiq.players import FinitePlayers
from shapiq.typing import HasShape, typechecked
from shapiq.value import VectorValue
from shapiq.value._dtype import default_float_dtype, validate_numeric_dtype, value_spec
from shapiq.value.value import JaxValue

if TYPE_CHECKING:
    from jax.sharding import Sharding
    from jax.typing import DTypeLike
    from numpy.typing import DTypeLike as NpDTypeLike


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


def _opaque_model_spec() -> jax.ShapeDtypeStruct:
    """Return the weak default used for opaque model outputs."""
    return jax.ShapeDtypeStruct((), default_float_dtype(), weak_type=True)


@flexdispatch
def infer_model_spec(
    model: Any,  # noqa: ANN401 - dispatches on any model type
    x_sample: Any = None,  # noqa: ANN401 - sample in the model's own format
) -> jax.ShapeDtypeStruct:
    """Infer the scalar value spec using one abstract row of batched JAX input.

    Backend registrations provide a lightweight metadata estimate without making a prediction.
    Opaque models default to a weak float. Use a game's ``output_dtype=`` or ``.astype(...)`` to
    enforce an explicit dtype.
    """
    if not isinstance(x_sample, (jax.Array, Tracer, JaxLike, jax.ShapeDtypeStruct)):
        return _opaque_model_spec()
    try:
        # Structured inputs keep their JAX-like representation and internal axes.
        sample = jax.tree.map(
            lambda x: jax.ShapeDtypeStruct(
                (1, *x.shape[1:]) if x.ndim > 0 else (),
                x.dtype,
                weak_type=getattr(x, "weak_type", False),
            ),
            x_sample,
        )
        result = jax.eval_shape(lambda x: predict(model, x), sample)
        spec = value_spec(result)
    except Exception:  # noqa: BLE001 - arbitrary models may reject abstract inputs
        return _opaque_model_spec()
    return _opaque_model_spec() if spec is None else spec


# Optional backends register in their own modules, imported once their objects first arrive.
@as_value.delayed_register(TORCH_TENSOR)
@predict.delayed_register(TORCH_MODULE)
@infer_model_spec.delayed_register(TORCH_MODULE)
def _(_: type[object]) -> None:
    from . import torch  # noqa: F401, PLC0415


@as_value.delayed_register(PANDAS_OUTPUT)
def _(_: type[object]) -> None:
    from . import pandas  # noqa: F401, PLC0415


@predict.delayed_register(SKLEARN_MODULE)
@infer_model_spec.delayed_register(SKLEARN_MODULE)
def _(_: type[object]) -> None:
    from . import sklearn  # noqa: F401, PLC0415


@typechecked
@dataclass(frozen=True, slots=True)
class XAIGame[P, OwnerT: JaxLike[Any] = "XAIGame[Any, Any]"](
    CoalitionDefinedGame[P, Float[JaxValue, "..."]],
    JaxAxisProtected[OwnerT, Any, _FiniteGame[P]],
    ABC,
):
    """A game explaining a model; ``_coalition_bool`` is left to the explanation.

    ``model`` is static metadata, shared by the batch. An explicit ``output_dtype`` sets
    a strong value dtype and casts model values on evaluation.
    """

    model: object
    # A required slot replaces the inherited abstract players property.
    players: Shaped[FinitePlayers[P], "#*batch"] = field()
    output_dtype: Any = field(default=None, kw_only=True)
    protected_axes: ClassVar[dict[str, int]] = {"players": 0}
    broadcasted_batch: ClassVar[set[str]] = {"players"}

    def __post_init__(self) -> None:
        """Validate an explicit dtype without touching the model."""
        if self.output_dtype is not None:
            resolved = validate_numeric_dtype(self.output_dtype, "XAIGame output_dtype")
            object.__setattr__(self, "output_dtype", resolved)

    def _infer_value_spec(self) -> jax.ShapeDtypeStruct:
        """Infer model output metadata without an input sample."""
        return infer_model_spec(self.model)

    @override
    def _value_spec(self) -> jax.ShapeDtypeStruct:
        """Return the inferred or explicitly declared value spec."""
        if self.output_dtype is not None:
            return jax.ShapeDtypeStruct((), self.output_dtype)
        return self._infer_value_spec()

    @property
    @override
    def dtype(self) -> jnp.dtype:
        """Return the value dtype without a concrete model call."""
        return self._value_spec().dtype

    @override
    def astype(
        self,
        dtype: "DTypeLike | NpDTypeLike | None",
        copy: bool = False,
        device: "jax.Device | Sharding | None" = None,
    ) -> Any:
        """Apply device placement after evaluation to keep inputs and indices together."""
        if dtype is None and device is not None:
            game = cast("XAIGame[P, OwnerT]", self.copy()) if copy else self
            return CastGame(game, None, device=device)
        return super(XAIGame, self).astype(dtype, copy=copy, device=device)

    @override
    def eval(self, coalition: FiniteCoalition) -> JaxValue:
        """Evaluate a coalition and apply an explicit output dtype if present."""
        value = super(XAIGame, self).eval(coalition)
        if self.output_dtype is not None:
            return value.astype(self.output_dtype, copy=False)
        return value


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
    """Explain the inputs ``x_explain`` against a ``background``; ``_coalition_bool`` is still open.

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
        super(LocalXAIGame, self).__post_init__()

    @override
    def _infer_value_spec(self) -> jax.ShapeDtypeStruct:
        """Infer from one abstract input row without reading the explained data."""
        sample: object = self.x_explain
        if isinstance(sample, (jax.Array, Tracer)):
            sample = jax.ShapeDtypeStruct(
                (1, *self.input_shape), sample.dtype, weak_type=sample.weak_type
            )
        elif isinstance(sample, JaxLike):
            sample = jf.jax_reshape(sample, (-1, *self.input_shape))
        return infer_model_spec(self.model, sample)

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
