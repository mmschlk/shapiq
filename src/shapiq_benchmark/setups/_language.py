"""Setup of the sentiment analysis game."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from shapiq_games import SentimentAnalysis

from ._base import Setup, runtime_field

__all__ = ["SentimentAnalysisSetup"]


@dataclass(frozen=True, kw_only=True)
class SentimentAnalysisSetup(Setup, name="sentiment_analysis"):
    """A :class:`~shapiq_games.SentimentAnalysis` game of a text with the default sentiment model.

    Attributes:
        input_text: The text to explain.
        mask_strategy: ``"mask"`` (default) or ``"remove"``.
        revision: The Hugging Face revision of the model (``None`` for the default branch).
        normalize: Whether to center the game. Defaults to ``True``.
        device: The device of the pipeline. A runtime field: it does not change the cache key.

    Examples:
        >>> setup = SentimentAnalysisSetup(input_text="A great cast, but a thin plot.")
        >>> game = setup.build()  # doctest: +SKIP
    """

    input_text: str
    mask_strategy: Literal["mask", "remove"] = "mask"
    revision: str | None = None
    normalize: bool = True
    device: int | str | None = runtime_field(None)

    def build(self) -> SentimentAnalysis:
        """Load the model and build the game."""
        return SentimentAnalysis(
            self.input_text,
            mask_strategy=self.mask_strategy,
            revision=self.revision,
            device=self.device,
            normalize=self.normalize,
        )
