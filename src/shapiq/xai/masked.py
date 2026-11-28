"""Masked games: fill the features outside a coalition and call the model."""

from dataclasses import dataclass, field
from math import prod
from typing import ClassVar, override

import jax
import jax.numpy as jnp
from jax.core import Tracer
from jaxtyping import Bool, Float, Int
from ndlike import jax_functions as jf
from ndlike.jax_like import JaxLike

from shapiq.masker import MarginalMasker, Masker
from shapiq.typing import HasShape, typechecked
from shapiq.value._dtype import value_spec
from shapiq.value.value import JaxValue

from .base import (
    LocalXAIGame,
    _opaque_model_spec,
    batch_index,
    check_input_shape,
    infer_model_spec,
    predict,
)


@typechecked
@dataclass(frozen=True, slots=True, weakref_slot=True)
class MaskedGame[In: HasShape, P](LocalXAIGame[In, P, "MaskedGame[In, P]"]):
    """Evaluate the model with the inputs outside every coalition filled by the masker.

    The ``background`` has the shape ``(*batch, *samples, *input_shape)``, with the masker's
    sample axes: none for a ``BaselineMasker``, so several baselines are separate games; one for
    the default ``MarginalMasker``, whose background rows are averaged. ``background_index`` is the
    position of every game's background entry, by default every entry once.

    ``batch_size`` bounds the model inputs of one model call, e.g. for a model on a GPU; every
    explained input's samples stay in one call, so a call has at least ``n_samples`` inputs. By
    default the model is called once on all inputs.
    """

    masker: Masker = field(default_factory=MarginalMasker)
    background_index: Int[jax.Array, "#*batch"] | None = field(default=None, kw_only=True)
    batch_size: int | None = field(default=None, kw_only=True)
    protected_axes: ClassVar[dict[str, int]] = {"background_index": 0}
    broadcasted_batch: ClassVar[set[str]] = {"background_index"}

    def __post_init__(self) -> None:
        """Check both input shapes and index every background entry by default.

        Raises:
            ValueError: The last axes of a dataset are not ``input_shape``, or ``batch_size`` is
                not positive.
        """
        if self.batch_size is not None and self.batch_size < 1:
            msg = f"The batch size must be positive, not {self.batch_size}."
            raise ValueError(msg)
        check_input_shape("background", self.background, self.input_shape)
        if self.background_index is None:
            input_ndim = len(self.input_shape) + self.masker.sample_ndim
            object.__setattr__(self, "background_index", batch_index(self.background, input_ndim))
        super(MaskedGame, self).__post_init__()

    @override
    def _infer_value_spec(self) -> jax.ShapeDtypeStruct:
        """Trace masking, prediction and aggregation for one abstract coalition row."""
        jax_backed = (jax.Array, Tracer, JaxLike)
        if isinstance(self.x_explain, jax_backed) and isinstance(self.background, jax_backed):
            try:
                coalitions = jax.ShapeDtypeStruct((1, self.n_players), jnp.bool_)
                rows = jax.ShapeDtypeStruct((1,), jnp.int32)
                values = jax.eval_shape(
                    self._values, self.x_explain, self.background, coalitions, rows, rows
                )
                spec = value_spec(values)
            except Exception:  # noqa: BLE001 - maskers and models may reject abstract inputs
                spec = None
            if spec is not None:
                return spec
            if infer_model_spec.dispatch(type(self.model)) is infer_model_spec.dispatch(object):
                return _opaque_model_spec()
        return super(MaskedGame, self)._infer_value_spec()

    @override
    def _coalition_bool(
        self, coalition: Bool[jax.Array, "#*batch {self.n_players}"]
    ) -> Float[JaxValue, "*batch"]:
        """Mask every explained input, call the model in batches and aggregate."""
        coalitions, rows, shape = self._flatten(
            coalition, self.explain_index, self.background_index
        )
        n = len(coalitions)
        step = n if self.batch_size is None else max(1, self.batch_size // self.n_samples)
        parts = [
            self._values(
                self.x_explain,
                self.background,
                coalitions[start : start + step],
                *(r[start : start + step] for r in rows),
            )
            for start in range(0, max(n, 1), max(step, 1))
        ]
        values = parts[0] if len(parts) == 1 else jf.jax_concatenate(parts)
        return jf.jax_reshape(values, shape)

    @property
    def n_samples(self) -> int:
        """Return the number of background samples of every explained input."""
        end = len(self.background.shape) - len(self.input_shape)
        return prod(self.background.shape[end - self.masker.sample_ndim : end])

    def _values(
        self,
        x_explain: In,
        background: In,
        coalitions: Bool[jax.Array, "n n_players"],
        explain_rows: Int[jax.Array, " n"],
        background_rows: Int[jax.Array, " n"],
    ) -> Float[JaxValue, " n"]:
        """Return the aggregated values of ``n`` evaluations from one model call."""
        inputs = self.masker.mask(
            x_explain,
            background,
            explain_rows,
            background_rows,
            coalitions,
            input_shape=self.input_shape,
        )
        values = predict(self.model, inputs)  # consecutive samples per input and coalition
        return self.masker.aggregate(jf.jax_reshape(values, (len(coalitions), -1)))
