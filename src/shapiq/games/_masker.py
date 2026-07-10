from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
from array_api_compat import array_namespace, device, to_device

if TYPE_CHECKING:
    from shapiq._shape import Shape
    from shapiq.coalitions import CoalitionArray


@runtime_checkable
class BackendArray(Protocol):
    """Minimal structural surface of an array from any Array API backend."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the array shape."""
        ...

    @property
    def ndim(self) -> int:
        """Return the number of axes."""
        ...

    @property
    def dtype(self) -> object:
        """Return the backend-native dtype."""
        ...


class Masker[ModelInputT](ABC):
    """Base abstraction for converting coalitions to model-native inputs."""

    n_players: int
    target_shape: Shape

    def __call__(self, coalitions: CoalitionArray) -> ModelInputT:
        """Create masked model-native inputs for coalitions."""
        self._validate_coalitions(coalitions)
        return self._mask(coalitions)

    def _validate_coalitions(self, coalitions: CoalitionArray) -> None:
        """Validate coalition compatibility at the masker boundary."""
        if coalitions.n_players != self.n_players:
            msg = "coalitions use a different number of players"
            raise ValueError(msg)

    @abstractmethod
    def _mask(self, coalitions: CoalitionArray) -> ModelInputT:
        """Create masked inputs after base validation."""


def coalition_masks_like(coalitions: CoalitionArray, like: object) -> object:
    """Return dense boolean coalition masks in the backend and device of ``like``.

    Maskers compute in the backend their arrays come from; this is the one
    backend-specific seam, converting the JAX-side coalition masks through
    DLPack (with a host-memory fallback) and moving them to the reference
    array's device.
    """
    dense = coalitions.to_dense()
    xp = array_namespace(like)
    if xp is array_namespace(dense):
        masks = dense
    else:
        try:
            masks = xp.from_dlpack(dense)
        except (BufferError, RuntimeError, TypeError, ValueError):
            masks = xp.asarray(np.asarray(dense).copy())
        masks = xp.astype(masks, xp.bool)
    if device(masks) != device(like):
        masks = to_device(masks, device(like))
    return masks


def require_shared_backend(reference: object, /, **named: object) -> None:
    """Raise when the named arrays come from a different backend or device.

    A masker's arrays must live in one backend on one device so masking
    stays a single ``where`` there; mixing backends would otherwise fail
    deep inside backend dispatch with an untranslated error.
    """
    xp = array_namespace(reference)
    for name, array in named.items():
        other = array_namespace(array)
        if other is not xp:
            msg = (
                f"{name} comes from {other.__name__.removeprefix('array_api_compat.')} "
                f"but the inputs from {xp.__name__.removeprefix('array_api_compat.')}; "
                "build a masker from arrays of one backend"
            )
            raise ValueError(msg)
        if device(array) != device(reference):
            msg = (
                f"{name} lives on {device(array)} but the inputs live on "
                f"{device(reference)}; keep the masker's arrays on one device"
            )
            raise ValueError(msg)
