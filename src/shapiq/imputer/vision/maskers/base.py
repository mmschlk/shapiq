"""
Masker base — abstract contract and configuration for feature occlusion strategies.

Defines:
    - Per-strategy parameter dataclasses: CrossModalMeanParams,
      CrossModalBlurParams, VisionMeanParams, VisionBlurParams, TextAttentionParams
    - MaskerConfig — caller-provided configuration
    - Masker(ABC) — abstract base for all maskers
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from ..base import PhysicalMask, ProcessorOutput


# ═══════════════════════════════════════════════════════════════════════
# Per-strategy parameter dataclasses
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class CrossModalMeanParams:
    """Cross-modal occlusion (vision-mean + text-attention)."""


@dataclass
class CrossModalBlurParams:
    """Cross-modal occlusion (vision-blur + text-attention)."""


@dataclass
class VisionMeanParams:
    """Pure image occlusion via multiplicative binary mask."""


@dataclass
class VisionBlurParams:
    """Pure image occlusion via Gaussian blur."""

    sigma: float = 3.0


@dataclass
class TextAttentionParams:
    """Pure text occlusion via attention_mask replacement."""


# ═══════════════════════════════════════════════════════════════════════
# Masker Configuration
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class MaskerConfig:
    """Complete configuration for a Masker.

    Caller-provided: strategy + per-strategy params.
    Default strategy is ``"crossmodal_mean"``.
    """

    strategy: str = "crossmodal_mean"
    crossmodal_mean: CrossModalMeanParams = field(default_factory=CrossModalMeanParams)
    crossmodal_blur: CrossModalBlurParams = field(default_factory=CrossModalBlurParams)
    vision_mean: VisionMeanParams = field(default_factory=VisionMeanParams)
    vision_blur: VisionBlurParams = field(default_factory=VisionBlurParams)
    text_attn: TextAttentionParams = field(default_factory=TextAttentionParams)

    @property
    def active_params(self):
        return getattr(self, self.strategy, None)


# ═══════════════════════════════════════════════════════════════════════
# Abstract Masker
# ═══════════════════════════════════════════════════════════════════════


class Masker(ABC):
    """Abstract base class for feature occlusion strategies.

    A Masker applies a PhysicalMask to model inputs (ProcessorOutput)
    and returns modified inputs ready for model.forward().
    """

    def __init__(self, config: MaskerConfig | None = None):
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
