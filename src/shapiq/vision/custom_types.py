"""This module contains all custom types used in the shapiq vision subpackage."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Protocol, overload, runtime_checkable

if TYPE_CHECKING:
    import torch


class CoalitionDomain(Enum):
    """Enumeration of coalition domains used by players and masking strategies."""

    PIXEL = "pixel"
    TOKEN = "token"  # noqa: S105 token is not a security issue here, it's naming for vit tokenization


@runtime_checkable
class VisionModel(Protocol):
    """Protocol for vision classification models."""

    @overload
    def __call__(self, x: torch.Tensor, /) -> torch.Tensor | ClassificationOutput: ...

    @overload
    def __call__(
        self,
        *,
        pixel_values: torch.Tensor,
        bool_masked_pos: torch.Tensor | None = None,
    ) -> torch.Tensor | ClassificationOutput: ...

    def __call__(self, *args, **kwargs) -> torch.Tensor | ClassificationOutput:
        """Return classification logits from a direct tensor or `pixel_values` call."""
        ...


@runtime_checkable
class ClassificationOutput(Protocol):
    """Protocol for model outputs exposing classification logits."""

    logits: torch.Tensor
