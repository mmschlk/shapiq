"""Text imputer for word-level masking explanations."""

from __future__ import annotations  # noqa: I001

import numpy as np
import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase


class TextImputer:
    """Simple version text imputer for word-level masking explanations.

    Supports:
    - HuggingFace encoder classifiers
    - word-level players
    - [MASK] replacement
    - batched coalition evaluation
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        text: str,
        class_idx: int = 0,
        batch_size: int = 16,
        device: str | None = None,
        output_type: str = "logit",
    ) -> None:
        """Initializes the Text Imputer.

        Args:
            model:HuggingFace sequence classification model.

            tokenizer:HuggingFace tokenizer. Use only whitespace separated words as players.

            text:Input sentence.

            class_idx:Target class index.

            batch_size:Batch size for model inference.

            device:"cuda", "cpu", or None (auto detect).

            output_type:"logit" or "probability"
        """
        self.model = model
        self.tokenizer = tokenizer
        self.text = text
        self.class_idx = class_idx
        self.batch_size = batch_size
        self.output_type = output_type

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"


        self.device = device
        self.model = self.model.to(self.device) # noqa: PGH003
        self.model.eval()

        self.words = self.text.split()
        self.n_players = len(self.words)

        self.mask_token = tokenizer.mask_token

        if self.mask_token is None:
            msg = (
                "Tokenizer does not define a mask token. "
                "This implementation requires a masked language model tokenizer."
            )
            raise ValueError(msg)

        if output_type not in {"logit", "probability"}:
            msg_0 = "output_type must be 'logit' or 'probability'"
            raise ValueError(msg_0)

    def mask_words(
        self,
        coalition: np.ndarray,
    ) -> str:
        """Replace missing words with [MASK].

        coalition:
            Binary vector of shape (n_players,)
            1 = keep word
            0 = mask word
        """
        if len(coalition) != self.n_players:
            msg = f"Coalition length {len(coalition)} does not match n_players={self.n_players}"
            raise ValueError(msg)

        masked_words = []

        for keep, word in zip(coalition, self.words, strict=False):
            if keep:
                masked_words.append(word)
            else:
                masked_words.append(self.mask_token)

        return " ".join(masked_words)

    def _coalitions_to_texts(
        self,
        coalitions: np.ndarray,
    ) -> list[str]:
        """Convert coalition matrix into masked texts."""
        return [self.mask_words(coalition) for coalition in coalitions]

    def _predict_batch(
        self,
        texts: list[str],
    ) -> np.ndarray:
        """Run batched inference."""
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        encoded = {key: value.to(self.device) for key, value in encoded.items()}

        with torch.no_grad():
            outputs = self.model(**encoded)
            logits = outputs.logits

        if self.output_type == "logit":
            scores = logits[:, self.class_idx]

        else:
            probs = torch.softmax(logits, dim=-1)
            scores = probs[:, self.class_idx]

        return scores.detach().cpu().numpy()

    def _batched_predict(
        self,
        texts: list[str],
    ) -> np.ndarray:
        """Predict in batches."""
        all_scores = []

        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            batch_scores = self._predict_batch(batch)
            all_scores.append(batch_scores)

        return np.concatenate(all_scores)

    def value_function(
        self,
        coalitions: np.ndarray,
    ) -> np.ndarray:
        """Evaluate coalition values.

        Args:
            coalitions:
                shape (n_coalitions, n_players)

        Returns:
            shape (n_coalitions,)
        """
        coalitions = np.asarray(coalitions)

        if coalitions.ndim == 1:
            coalitions = coalitions.reshape(1, -1)

        if coalitions.shape[1] != self.n_players:
            msg = f"Expected coalition width {self.n_players}, got {coalitions.shape[1]}"
            raise ValueError(msg)

        texts = self._coalitions_to_texts(coalitions)
        return self._batched_predict(texts)

    def __call__(
        self,
        coalitions: np.ndarray,
    ) -> np.ndarray:
        """Make TextImputer compatible with shapiq approximators."""
        return self.value_function(coalitions)

    def full_prediction(self) -> float:
        """Score of the original unmasked text."""
        score = self._predict_batch([self.text])[0]
        return float(score)
