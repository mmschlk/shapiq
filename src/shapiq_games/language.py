"""Language games: a sentiment classifier's score with tokens of the input removed."""

from __future__ import annotations

from typing import Any, Literal, Self

import numpy as np

from shapiq.game import Game
from shapiq_games._base import ConfigMixin, as_bool_coalitions
from shapiq_games._optional import require

__all__ = ["SENTIMENT_MODEL_ID", "SentimentAnalysis"]

SENTIMENT_MODEL_ID = "lvwerra/distilbert-imdb"
"""The default Hugging Face model: DistilBERT fine-tuned on IMDb movie reviews."""


class SentimentAnalysis(ConfigMixin, Game):
    """The sentiment analysis game: the signed sentiment score of a text with tokens removed.

    The players are the tokens of the input text (without the special tokens). Absent tokens are
    replaced by the mask token (``mask_strategy="mask"``) or deleted (``"remove"``). The value of a
    coalition is the classifier's score, positive for the ``positive_label`` and negative
    otherwise, so it lies in ``[-1, 1]``.

    Attributes:
        input_text: The decoded input text.
        tokens: The token ids of the players.
        original_model_output: The signed score of the full text.
        model_commit: The Hugging Face commit of the loaded model, or ``None`` if unknown.
    """

    def __init__(
        self,
        input_text: str,
        *,
        classifier: Any = None,  # noqa: ANN401
        mask_strategy: Literal["mask", "remove"] = "mask",
        positive_label: str = "POSITIVE",
        device: int | str | None = None,
        revision: str | None = None,
        normalize: bool = True,
        verbose: bool = False,
    ) -> None:
        """Initialize the sentiment analysis game.

        Args:
            input_text: The text to explain.
            classifier: A Hugging Face text-classification pipeline (anything with a ``tokenizer``
                attribute that maps a list of texts to ``[{"label": ..., "score": ...}]``). If
                ``None``, the pipeline of :data:`SENTIMENT_MODEL_ID` is loaded.
            mask_strategy: ``"mask"`` or ``"remove"``. Defaults to ``"mask"``.
            positive_label: The label whose score counts as positive. Defaults to ``"POSITIVE"``,
                the label of the default model.
            device: The device of the default pipeline.
            revision: The Hugging Face revision (branch, tag, or commit) of the default model.
                ``None`` loads the default branch; :attr:`model_commit` records what was loaded.
            normalize: Whether to center the game such that the value of the empty coalition is
                zero. Defaults to ``True``.
            verbose: Whether to show a progress bar when evaluating the game.

        Raises:
            ValueError: If ``mask_strategy`` is unknown, or is ``"mask"`` for a tokenizer without
                a mask token.
        """
        if mask_strategy not in ("mask", "remove"):
            msg = f"mask_strategy must be 'mask' or 'remove', got {mask_strategy!r}."
            raise ValueError(msg)
        if classifier is None:
            transformers = require("transformers", purpose="the sentiment analysis game")
            classifier = transformers.pipeline(
                task="sentiment-analysis",
                model=SENTIMENT_MODEL_ID,
                revision=revision,
                device=device,
            )
        if mask_strategy == "mask" and classifier.tokenizer.mask_token_id is None:
            msg = "The tokenizer has no mask token; use mask_strategy='remove'."
            raise ValueError(msg)
        self.mask_strategy = mask_strategy
        self.positive_label = positive_label
        self._classifier = classifier
        self._tokenizer = classifier.tokenizer
        self.model_commit = _commit_hash(classifier)
        self.tokens = np.asarray(
            self._tokenizer(input_text, add_special_tokens=False)["input_ids"], dtype=int
        )
        self.input_text = str(self._tokenizer.decode(self.tokens))
        self.original_model_output = float(self._scores([input_text])[0])
        n_players = self.tokens.shape[0]
        empty_value = float(self._evaluate(np.zeros((1, n_players), dtype=bool))[0])
        super().__init__(
            n_players,
            normalize=normalize,
            normalization_value=empty_value,
            verbose=verbose,
        )

    def _scores(self, texts: list[str]) -> np.ndarray:
        outputs = self._classifier(texts, truncation=True)
        return np.array(
            [
                out["score"] if out["label"] == self.positive_label else -out["score"]
                for out in outputs
            ],
            dtype=float,
        )

    def _evaluate(self, coalitions: np.ndarray) -> np.ndarray:
        texts = []
        for coalition in coalitions:
            if self.mask_strategy == "remove":
                tokens = self.tokens[coalition]
            else:
                tokens = self.tokens.copy()
                tokens[~coalition] = self._tokenizer.mask_token_id
            texts.append(str(self._tokenizer.decode(tokens)))
        return self._scores(texts)

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        """Return the signed sentiment score of the text restricted to each coalition."""
        return self._evaluate(as_bool_coalitions(coalitions))

    @classmethod
    def from_config(
        cls,
        *,
        input_text: str,
        mask_strategy: Literal["mask", "remove"] = "mask",
        revision: str | None = None,
        device: int | str | None = None,
        normalize: bool = True,
    ) -> Self:
        """Build the game with the default sentiment model.

        The configuration records the Hugging Face commit of the loaded model, so a new version
        of the model gets a new fingerprint and never reuses cached ground truth.

        Args:
            input_text: The text to explain.
            mask_strategy: ``"mask"`` or ``"remove"``.
            revision: The Hugging Face revision of the model (``None`` for the default branch).
            device: The device of the pipeline (not part of the configuration).
            normalize: Whether to center the game.

        Returns:
            The configured game.
        """
        game = cls(
            input_text,
            mask_strategy=mask_strategy,
            device=device,
            revision=revision,
            normalize=normalize,
        )
        return game._set_config(
            input_text=input_text,
            model=SENTIMENT_MODEL_ID,
            revision=revision,
            model_commit=game.model_commit,
            mask_strategy=mask_strategy,
            normalize=normalize,
        )


def _commit_hash(classifier: Any) -> str | None:  # noqa: ANN401
    """Return the Hugging Face commit of a pipeline's model, or ``None`` if it is unknown."""
    config = getattr(getattr(classifier, "model", None), "config", None)
    commit = getattr(config, "_commit_hash", None)
    return str(commit) if commit else None
