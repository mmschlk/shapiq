"""Masker base — abstract contract and configuration for feature occlusion strategies.

Defines:
    - Per-strategy parameter dataclasses: CrossModalMeanParams,
      CrossModalBlurParams, VisionMeanParams, VisionBlurParams, TextAttentionParams
    - MaskerConfig — caller-provided configuration
    - Masker(ABC) — abstract base for all maskers
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from shapiq.imputer.vision.base import EmptyParams

if TYPE_CHECKING:
    from shapiq.imputer.vision.base import PhysicalMask, ProcessorOutput

# Per-strategy parameter dataclasses


CrossModalMeanParams = EmptyParams
VisionMeanParams = EmptyParams
TextAttentionParams = EmptyParams


@dataclass
class CrossModalBlurParams:
    """Cross-modal occlusion (vision-blur + text-attention).

    Attributes:
        sigma: Gaussian blur sigma forwarded to VisionBlurMasker.
    """

    sigma: float = 3.0


@dataclass
class VisionBlurParams:
    """Pure image occlusion via Gaussian blur."""

    sigma: float = 3.0



# Masker Configuration


@dataclass
class MaskerConfig:
    """Complete configuration for a Masker.

    Caller-provided: strategy + per-strategy params.
    Default strategy is ``"crossmodal_mean"``.
    """

    strategy: str = "crossmodal_mean"
    params: CrossModalMeanParams | CrossModalBlurParams | VisionMeanParams | VisionBlurParams | TextAttentionParams = field(default_factory=CrossModalMeanParams)


# Abstract Masker


class Masker(ABC):
    """Abstract base class for feature occlusion strategies.

    A Masker applies a PhysicalMask to model inputs (ProcessorOutput)
    and returns modified inputs ready for model.forward().
    """

    def __init__(self, config: MaskerConfig | None = None) -> None:
        """Initialize the masker with an optional configuration."""
        self.config = config or MaskerConfig()

    @abstractmethod
    def apply(
        self,
        processor_output: ProcessorOutput,
        physical_mask: PhysicalMask,
    ) -> ProcessorOutput:
        """Apply the physical mask to processor outputs.

        Args:
            processor_output: Preprocessed model inputs.
            physical_mask: Concrete pixel/token masks to apply.

        Returns:
            Modified ProcessorOutput with occlusion applied.

        Note:
            Implementations MUST clone inputs before mutation.
        """
        ...
