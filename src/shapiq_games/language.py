"""Language games: a sentiment classifier's score with tokens of the input removed."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from shapiq.game import Game
from shapiq_games._base import as_bool_coalitions
from shapiq_games._optional import require

if TYPE_CHECKING:
    from shapiq.typing import CoalitionMatrix, FloatVector, GameValues
    from shapiq_games.typing import MaskStrategy

__all__ = ["SENTIMENT_MODEL_ID", "SentimentAnalysis"]

SENTIMENT_MODEL_ID = "lvwerra/distilbert-imdb"
"""The default Hugging Face model: DistilBERT fine-tuned on IMDb movie reviews."""


class SentimentAnalysis(Game):
    """The sentiment analysis game: the signed sentiment score of a text with tokens removed.

    The players are the tokens of the input text (without the special tokens). Absent tokens are
    replaced by the mask token (``mask_strategy="mask"``) or deleted (``"remove"``). The value of a
    coalition is the signed score ``2 P(positive) - 1`` of the classifier, i.e.
    ``P(positive) - P(negative)`` for a binary classifier. It lies in ``[-1, 1]`` and is ``0`` where
    the classifier is undecided.

    Attributes:
        input_text: The decoded input text.
        tokens: The token ids of the players.
        original_model_output: The signed score of the full text.

    Examples:
        >>> game = SentimentAnalysis("A great cast, but a thin plot.")  # doctest: +SKIP
        >>> game.original_model_output  # the signed score of the full text  # doctest: +SKIP
    """

    def __init__(
        self,
        input_text: str,
        *,
        classifier: Any = None,  # noqa: ANN401
        mask_strategy: MaskStrategy = "mask",
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
                attribute that maps a list of texts and ``top_k=None`` to the scores of all labels,
                ``[[{"label": ..., "score": ...}, ...], ...]``). If ``None``, the pipeline of
                :data:`SENTIMENT_MODEL_ID` is loaded.
            mask_strategy: ``"mask"`` or ``"remove"``. Defaults to ``"mask"``.
            positive_label: The label whose probability is explained. Defaults to
                ``"POSITIVE"``, the label of the default model.
            device: The device of the default pipeline.
            revision: The Hugging Face revision (branch, tag, or commit) of the default model.
                ``None`` loads the default branch.
            normalize: Whether to center the game such that the value of the empty coalition is
                zero. Defaults to ``True``.
            verbose: Whether to show a progress bar when evaluating the game.

        Raises:
            ValueError: If ``mask_strategy`` is unknown, is ``"mask"`` for a tokenizer without a
                mask token, or the classifier has no ``positive_label``.
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

    def _scores(self, texts: list[str]) -> FloatVector:
        """Return the signed score ``2 P(positive) - 1`` of every text."""
        outputs = self._classifier(texts, truncation=True, top_k=None)
        scores = np.zeros(len(texts))
        for i, labels in enumerate(outputs):
            probabilities = {out["label"]: out["score"] for out in labels}
            if self.positive_label not in probabilities:
                msg = (
                    f"The classifier has no label {self.positive_label!r}; its labels are "
                    f"{sorted(probabilities)}. Pass positive_label."
                )
                raise ValueError(msg)
            scores[i] = 2.0 * float(probabilities[self.positive_label]) - 1.0
        return scores

    def _evaluate(self, coalitions: CoalitionMatrix) -> GameValues:
        texts = []
        for coalition in coalitions:
            if self.mask_strategy == "remove":
                tokens = self.tokens[coalition]
            else:
                tokens = self.tokens.copy()
                tokens[~coalition] = self._tokenizer.mask_token_id
            texts.append(str(self._tokenizer.decode(tokens)))
        return self._scores(texts)

    def value_function(self, coalitions: CoalitionMatrix) -> GameValues:
        """Return the signed sentiment score of the text restricted to each coalition."""
        return self._evaluate(as_bool_coalitions(coalitions))
