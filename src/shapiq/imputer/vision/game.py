"""VisionLanguageGame — Thin shapiq.Game adapter for HuggingFace VLMs.

All masking, batching, and model-forward logic is delegated to a
VisionImputer. The Game only handles Shapley scheduling concerns.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from shapiq.game import Game

if TYPE_CHECKING:
    from transformers import ProcessorMixin

    from shapiq.imputer.vision.imputer import VisionImputer


class VisionLanguageGame(Game):
    """A shapiq.Game adapter for HuggingFace CLIP / SigLIP.

    Delegates all masking, batching, and model forward passes to
    a VisionImputer. Handles only normalisation values and the
    shapiq value_function interface.
    """

    def __init__(
        self,
        imputer: VisionImputer,
        batch_size: int = 64,
        *,
        verbose: bool = False,
    ) -> None:
        """Initialize the vision-language game.

        Args:
            imputer: Configured VisionImputer instance.
            batch_size: Batch size for model forward passes.
            verbose: Whether to log normalisation values.
        """
        self._imputer = imputer
        self._batch_size = batch_size

        self.n_players_image = imputer.n_players_image
        self.n_players_text = imputer.n_players_text

        # Compute normalisation values
        coalitions = np.zeros((2, self.n_players_image + self.n_players_text), dtype=bool)
        coalitions[1, :] = True
        game_output = self.value_function(coalitions=coalitions)
        self.empty_value = float(game_output[0])
        self.full_value = float(game_output[1])

        if verbose:
            logging.getLogger(__name__).info(
                "Similarity: %s (empty=%s)", self.full_value, self.empty_value
            )

        super().__init__(
            n_players=self.n_players_image + self.n_players_text,
            normalize=True,
            normalization_value=self.empty_value,
        )

    @property
    def inputs(self) -> dict:
        """Raw HuggingFace processor output (BatchEncoding)."""
        return self._imputer.inputs_raw

    @property
    def processor(self) -> ProcessorMixin:
        """The HuggingFace processor used by the imputer."""
        return self._imputer.processor

    def value_function(
        self,
        coalitions: np.ndarray,
        batch_size: int | None = None,
    ) -> np.ndarray:
        """Evaluate the game for a set of 1D coalitions."""
        if batch_size is None:
            batch_size = self._batch_size
        return self._imputer.forward_1d(coalitions, batch_size=batch_size)

    def value_function_crossmodal(
        self,
        coalitions_image: np.ndarray,
        coalitions_text: np.ndarray,
        batch_size: int | None = None,
    ) -> np.ndarray:
        """Evaluate the game for cross-modal (2D) coalitions."""
        if batch_size is None:
            batch_size = self._batch_size
        return self._imputer.forward_crossmodal(
            coalitions_image,
            coalitions_text,
            batch_size=batch_size,
        )
