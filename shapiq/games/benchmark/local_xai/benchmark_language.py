"""This module contains the Sentiment Classification Game class, which is a subclass of the Game"""

import numpy as np

from shapiq.games.base import Game


class SentimentAnalysis(Game):
    """Sentiment Classification Game.

    The Sentiment Classification Game uses a sentiment classification model from huggingface to
    classify the sentiment of a given text. The game is defined by the number of players, which is
    equal to the number of tokens in the input text. The worth of a coalition is the sentiment of
    the coalition's text. The sentiment is encoded as a number between -1 (strong negative
    sentiment) and 1 (strong positive sentiment).

    Note:
        This benchmark game requires the `transformers` package to be installed. You can install it
        via pip:
        ```bash
        pip install transformers
        ```

    Args:
        input_text: The input text to be classified.
        normalize: Whether to normalize the game. Defaults to True.
        mask_strategy: The strategy to handle the tokens not in the coalition. Either 'remove' or
            'mask'. Defaults to 'mask'. With 'remove', the tokens not in the coalition are removed
            from the text. With 'mask', the tokens not in the coalition are replaced by the
            mask_token_id.

    Attributes:
        n_players: The number of players in the game.
        original_input_text: The original input text (as given in the constructor).
        input_text: The input text after tokenization took place (may differ from the original).
        original_model_output: The sentiment of the original input text in the range [-1, 1].
        normalization_value: The score used for normalization.

    Properties:
        normalize: Whether the game is normalized.

    Examples:
        >>> game = SentimentAnalysis("This is a six word sentence")
        >>> game.n_players
        6
        >>> game.original_input_text
        'This is a six word sentence'
        >>> game.input_text
        'this is a six word sentence'
        >>> game.original_model_output
        0.6615
        >>> game(np.asarray([1, 1, 1, 1, 1, 1], dtype=bool))
        0.6615
    """

    def __init__(
        self, input_text: str, mask_strategy: str = "mask", verbose: bool = False, *args, **kwargs
    ):
        # import the required modules locally (to avoid having to install them for all)
        from transformers import pipeline

        if mask_strategy not in ["remove", "mask"]:
            raise ValueError(
                f"'mask_strategy' must be either 'remove' or 'mask' and not {mask_strategy}"
            )
        self.mask_strategy = mask_strategy

        # get the model
        self._classifier = pipeline(model="lvwerra/distilbert-imdb", task="sentiment-analysis")
        self._tokenizer = self._classifier.tokenizer
        self._mask_toke_id = self._tokenizer.mask_token_id
        # for this model: {0: [PAD], 100: [UNK], 101: [CLS], 102: [SEP], 103: [MASK]}

        # get the text
        self.original_input_text: str = input_text
        self._tokenized_input = np.asarray(
            self._tokenizer(self.original_input_text)["input_ids"][1:-1]
        )
        self.input_text: str = str(self._tokenizer.decode(self._tokenized_input))

        # setup players
        n_players = len(self._tokenized_input)

        # get original sentiment
        self.original_model_output = self._classifier(self.original_input_text)[0]["score"]
        self._full_output = self.value_function(np.ones((1, n_players), dtype=bool))[0]
        self._empty_output = self.value_function(np.zeros((1, n_players), dtype=bool))[0]

        # setup game object
        super().__init__(
            n_players, normalization_value=self._empty_output, verbose=verbose, *args, **kwargs
        )

    def value_function(self, coalitions: np.ndarray[bool]) -> np.ndarray[float]:
        """Returns the sentiment of the coalition's text.

        Args:
            coalitions: The coalition as a binary matrix of shape `(n_coalitions, n_players)`.

        Returns:
            The sentiment of the coalition's text as a vector of length `n_coalitions`.
        """
        # get the texts of the coalitions
        texts = []
        for coalition in coalitions:
            if self.mask_strategy == "remove":
                tokenized_coalition = self._tokenized_input[coalition]
            else:  # mask_strategy == "mask"
                tokenized_coalition = self._tokenized_input.copy()
                # all tokens not in the coalition are set to mask_token_id
                tokenized_coalition[~coalition] = self._mask_toke_id
            coalition_text = self._tokenizer.decode(tokenized_coalition)
            texts.append(coalition_text)

        # get the sentiment of the texts
        sentiments = self._model_call(texts)

        return sentiments

    def _model_call(self, input_texts: list[str]) -> np.ndarray[float]:
        """Calls the sentiment classification model with a list of texts.

        Args:
            input_texts: A list of input texts.

        Returns:
            The sentiment of the input texts as a vector of length `n_coalitions`.
        """
        # get the sentiment of the input texts
        outputs = self._classifier(input_texts)
        outputs = [
            output["score"] * 1 if output["label"] == "POSITIVE" else output["score"] * -1
            for output in outputs
        ]
        sentiments = np.array(outputs, dtype=float)

        return sentiments
