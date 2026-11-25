"""Maskers: turn coalitions into model inputs in the model's own data format.

A masker is a static strategy shared by a game batch. It receives the explained inputs and the
background in whatever format the user passed them, e.g. NumPy, JAX, torch or pandas, with the
shape ``(*batch, *input_shape)``, and returns model inputs in that same format with the shape
``(n, *input_shape)``: the batch and sample axes merged, as models take them. The input axes are
never touched, so images keep their channels, height and width.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import prod
from typing import Any, ClassVar

import jax
import jax.numpy as jnp
import numpy as np
from array_api_compat import array_namespace, device, is_jax_namespace
from flextype import flexdispatch
from jax.core import Tracer
from jaxtyping import Bool, Float, Int
from ndlike import jax_functions as jf
from ndlike.array_like import WeakArrayLike

from shapiq.value.value import JaxValue


def require_eager(what: str, *arrays: object) -> None:
    """Raise if any array is traced: ``what`` needs concrete values, e.g. a non-JAX backend.

    Raises:
        TypeError: An array is traced, e.g. under ``jit``.
    """
    if any(isinstance(array, Tracer) for array in arrays):
        msg = f"{what} cannot be traced; evaluate the game outside jit, or use JAX data and models."
        raise TypeError(msg)


@flexdispatch
def impute[T: Any](
    x_explain: T,
    background: T,  # noqa: ARG001 - used by registrations
    explain_rows: Int[jax.Array, " n"],  # noqa: ARG001
    background_rows: Int[jax.Array, " n"],  # noqa: ARG001
    keep: Bool[jax.Array, "n ..."],  # noqa: ARG001
    *,
    sample_ndim: int,  # noqa: ARG001
) -> T:
    """Return the explained inputs with the inputs outside ``keep`` from their background.

    ``explain_rows`` and ``background_rows`` index the batch axes of ``x_explain`` and
    ``background``, merged into one. Each background entry has ``sample_ndim`` sample axes, e.g.
    one for several background rows, zero for one baseline. The result has the shape
    ``(n * n_samples, *input_shape)``: the samples of each explained input are consecutive.

    Raises:
        TypeError: The data type has no registered imputation.
    """
    msg = (
        f"Model inputs of type {type(x_explain).__qualname__} have no imputation; "
        "register one with `impute.register`."
    )
    raise TypeError(msg)


def _to_backend(xp: Any, dev: Any, indices: jax.Array) -> Any:  # noqa: ANN401 - any backend
    """Return a JAX array on the data's backend and device; JAX data keeps it as it is."""
    if is_jax_namespace(xp):
        return xp.asarray(indices, device=dev)
    # Other backends take JAX arrays through NumPy; copy, as those views are read-only.
    return xp.asarray(np.array(indices), device=dev)


def _take_batch(xp: Any, dev: Any, data: Any, rows: jax.Array, ndim: int) -> Any:  # noqa: ANN401
    """Return the entries ``rows`` of the merged batch axes of ``data``, keeping its last ``ndim``."""
    entries = xp.reshape(data, (-1, *data.shape[data.ndim - ndim :]))
    return xp.take(entries, _to_backend(xp, dev, rows), axis=0)


@impute.register(np.ndarray | jax.Array)
def impute_array[T: WeakArrayLike[Any]](
    x_explain: T,
    background: T,
    explain_rows: Int[jax.Array, " n"],
    background_rows: Int[jax.Array, " n"],
    keep: Bool[jax.Array, "n ..."],
    *,
    sample_ndim: int,
) -> T:
    """Impute arrays through the array API, on their own backend and device; JAX data traces."""
    xp, dev = array_namespace(x_explain, background), device(x_explain)
    if not is_jax_namespace(xp):
        require_eager(f"{type(x_explain).__qualname__} data", explain_rows, background_rows, keep)
    input_shape = keep.shape[1:]
    n, ndim = len(keep), len(input_shape)
    x = _take_batch(xp, dev, x_explain, explain_rows, ndim)  # (n, *input_shape)
    reference = _take_batch(xp, dev, background, background_rows, ndim + sample_ndim)
    expand = (n, *(1,) * sample_ndim, *input_shape)  # align with the sample axes
    mask = _to_backend(xp, dev, keep.reshape(expand))
    filled = xp.where(mask, xp.reshape(x, expand), reference)  # (n, *samples, *input_shape)
    return xp.reshape(filled, (-1, *input_shape))


@flexdispatch
def select[T: Any](
    data: T,
    rows: Int[jax.Array, " n"],  # noqa: ARG001 - used by registrations
    features: Bool[jax.Array, " n_features"],  # noqa: ARG001
) -> T:
    """Return the entries ``rows`` of the merged batch axes of tabular ``data``, only ``features``.

    Raises:
        TypeError: The data type has no registered selection.
    """
    msg = (
        f"Model inputs of type {type(data).__qualname__} have no selection; "
        "register one with `select.register`."
    )
    raise TypeError(msg)


@select.register(np.ndarray | jax.Array)
def select_array[T: WeakArrayLike[Any]](
    data: T, rows: Int[jax.Array, " n"], features: Bool[jax.Array, " n_features"]
) -> T:
    """Select through the array API, on the array's own backend and device."""
    require_eager("Selecting features", rows, features)  # the number of columns is data-dependent
    xp, dev = array_namespace(data), device(data)
    picked = _take_batch(xp, dev, data, rows, 1)
    columns = jnp.flatnonzero(features)
    return xp.take(picked, _to_backend(xp, dev, columns), axis=1)


# Optional backends register in their own modules, imported once their data first arrives.
@impute.delayed_register("torch.Tensor")
@select.delayed_register("torch.Tensor")
def _(cls: type) -> None:  # noqa: ARG001
    import shapiq.masker.torch  # noqa: F401, PLC0415


@impute.delayed_register("pandas.DataFrame")
@select.delayed_register("pandas.DataFrame")
def _(cls: type) -> None:  # noqa: ARG001
    import shapiq.masker.pandas  # noqa: F401, PLC0415


class Masker(ABC):
    """How a coalition's absent inputs are filled from the background.

    Maskers are hashable: games keep them as static metadata. The background has the shape
    ``(*batch, *samples, *input_shape)`` with ``sample_ndim`` sample axes. The hooks:

    - ``input_mask``: the inputs every coalition keeps; one player per input by default.
    - ``mask``: the model inputs of every explained input and coalition, in the data's format.
    - ``aggregate``: combine the values of every explained input's samples; the mean by default.
    """

    __slots__ = ()

    @property
    @abstractmethod
    def sample_ndim(self) -> int:
        """Return the number of sample axes of every background entry."""

    def input_mask(
        self, coalitions: Bool[jax.Array, "n n_players"], input_shape: tuple[int, ...]
    ) -> Bool[jax.Array, "n ..."]:
        """Return the inputs every coalition keeps: one player per input.

        Raises:
            ValueError: The number of players is not the number of inputs.
        """
        if coalitions.shape[-1] != prod(input_shape):
            msg = (
                f"{coalitions.shape[-1]} players do not match the inputs {input_shape}; "
                "a masker mapping players to groups of inputs is needed."
            )
            raise ValueError(msg)
        return coalitions.reshape(len(coalitions), *input_shape)

    def mask(
        self,
        x_explain: Any,  # noqa: ANN401 - native data
        background: Any,  # noqa: ANN401
        explain_rows: Int[jax.Array, " n"],
        background_rows: Int[jax.Array, " n"],
        coalitions: Bool[jax.Array, "n n_players"],
        *,
        input_shape: tuple[int, ...],
    ) -> Any:  # noqa: ANN401
        """Return the model inputs ``(n * n_samples, *input_shape)``: consecutive samples."""
        keep = self.input_mask(coalitions, input_shape)
        return impute(
            x_explain, background, explain_rows, background_rows, keep, sample_ndim=self.sample_ndim
        )

    def aggregate(self, values: Float[JaxValue, "n n_samples"]) -> Float[JaxValue, " n"]:
        """Return the mean value of every explained input's samples."""
        return jf.jax_mean(values, axis=-1)


@dataclass(frozen=True, slots=True)
class BaselineMasker(Masker):
    """Fill the inputs outside a coalition from one baseline: no sample axis.

    Several baselines are separate games of the batch, never averaged.
    """

    sample_ndim: ClassVar[int] = 0


@dataclass(frozen=True, slots=True)
class MarginalMasker(Masker):
    """Fill the inputs outside a coalition from every background row: one sample axis, averaged."""

    sample_ndim: ClassVar[int] = 1
