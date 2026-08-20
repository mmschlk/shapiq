"""This module contains the Sentiment Classification Game class, which is a subclass of the Game."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from transformers import BatchEncoding, PreTrainedModel, PreTrainedTokenizerBase

import numpy as np
import torch
import torch.nn.functional as f

from shapiq.game import Game


class SentimentAnalysis(Game):
    """Sentiment Classification Game.

    The Sentiment Classification Game uses a sentiment classification model from huggingface to
    classify the sentiment of a given text. The game is defined by the number of players, which is
    equal to the number of tokens in the input text. The worth of a coalition is the sentiment of
    the coalition's text. The sentiment is encoded as a number between -1 (strong negative
    sentiment) and 1 (strong positive sentiment).

    Note:
        This benchmark game requires the ``transformers`` package to be installed.

    Attributes:
        original_input_text: The original input text (as given in the constructor).
        input_text: The input text after tokenization took place (may differ from the original).
        original_model_output: The sentiment of the original input text in the range [-1, 1].
        normalization_value: The score used for normalization.
        mask_strategy: The strategy to use for the tokens not in the coalition.

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
        self,
        input_text: str,
        *,
        mask_strategy: str = "mask",
        verbose: bool = False,
        device: int | str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the Sentiment Classification Game.

        Args:
            input_text: The input text to analyze as a string.

            mask_strategy: The strategy to use for the tokens not in the coalition. Can be either
                ``"remove"`` or ``"mask"``. Defaults to ``"mask"``.

            verbose: Whether to print additional information. Defaults to ``False``.

            device: The device to use for the model. Can be an integer (GPU index) or a string
                (e.g., "cuda:0" for the first GPU, "cpu" for CPU). Defaults to ``None``, which uses
                huggingface's default device setting (usually CPU or GPU if available).

            **kwargs: Additional keyword arguments (not used).
        """
        # import the required modules locally (to avoid having to install them for all)
        from transformers import pipeline

        if mask_strategy not in ["remove", "mask"]:
            msg = f"'mask_strategy' must be either 'remove' or 'mask' and not {mask_strategy}"
            raise ValueError(msg)
        self.mask_strategy = mask_strategy

        # get the model
        self._classifier = pipeline(
            model="lvwerra/distilbert-imdb", task="sentiment-analysis", device=device
        )
        self._tokenizer = self._classifier.tokenizer
        self._mask_toke_id = self._tokenizer.mask_token_id
        # for this model: {0: [PAD], 100: [UNK], 101: [CLS], 102: [SEP], 103: [MASK]}

        # get the text
        self.original_input_text: str = input_text
        self._tokenized_input = np.asarray(
            self._tokenizer(self.original_input_text)["input_ids"][1:-1],
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
            n_players,
            normalization_value=self._empty_output,
            verbose=verbose,
            **kwargs,
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
        return self._model_call(texts)

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
        return np.array(outputs, dtype=float)


class LMGeneration(Game):
    """LMGeneration explanation game for language model black box sequence generations.

    The game evaluates a generating language model on masked versions of a given
    input prompts. Each non-special token is treated as one player. For a coalition,
    active tokens (players in the coalition) keep their original token id, while
    inactive tokens (players outside the coalition) are replaced by a specified
    token id.

    The resulting model output is embedded using a second language model, and compared
    to the embedding of the output for the original sequence. The semantical similarity
    (cosine similarity) is treated as the value of that coalition and can be used
    to compute Shapley interaction values.

    Attributes:
        sequence: Input sequence (prompt) for which the generated output is to be explained.
        device: The torch device used for inference.
        model: Generating language model to be explained. The output of ``model`` must be a
            sequence of text.
        tokenizer: The tokenizer of the ``model``.
        sampling_params: Dictionary of parameters that can override the default sampling
            process of the generator model.
        batch_size: Batch sized used for sampling and embedding.
        embed_model: Embedding language model used in the value function.
            The output of ``embed_model`` must be an embedding vector.
        embed_tokenizer: The tokenizer of the ``embed_model``.
        special_token_ids: Set of special tokens of the ``tokenizer``
        baseline: Token id used for imputation.
        original_embed: Embedding of the output for the original ``sequence`` generated by the
            ``model``. This is computed once at initialization to save compute in the value function.
        n_players: Number of non special tokens in the ``sequence``.
        normalize: Whether to normalize the game by subtracting the empty coalition value.
        normalization_value: The value of the empty coalition value.
        verbose: Whether to print progress information
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        embed_model: PreTrainedModel,
        embed_tokenizer: PreTrainedTokenizerBase,
        sequence: str,
        *,
        baseline_id: int | None = None,
        sampling_params: dict | None = None,
        batch_size: int | None = None,
        normalize: bool = False,
        verbose: bool = False,
    ) -> None:
        """Initialize the LMGeneration explanation game.

        Args:
            model: Generating language model to be explained. The output of ``model``
                must be a sequence of text.
            tokenizer: The tokenizer of the ``model``.
            embed_model: Embedding language model used in the value function.
                The output of ``embed_model`` must be an embedding vector.
            embed_tokenizer: The tokenizer of the ``embed_model``.
            sequence: Input sequence (prompt) for which the generated output is to be explained.
            baseline_id: Token id used for token imputation.
                Defaults to the token id for a blank space for the given ``tokenizer``.
            sampling_params: Overwrite for the default sampling parameters
                for the ``model`` inference. Defaults to:
                    ``max_new_tokens``=256,
                    ``do_sample``=False
            batch_size: batch size used for inference.
                Defaults to 1.
            normalize: Whether to normalize the game by subtracting the empty
                coalition value.
            verbose: Whether to print progress information inherited from
                :class:`shapiq.game.Game`.

        Raises:
            ValueError: If ``sequence`` is empty.
        """
        self.sequence = sequence
        if len(self.sequence) == 0:
            msg = "Sequence must not be empty."
            raise ValueError(msg)

        self.device = next(model.parameters()).device

        self.model = model
        self.model.eval().to(self.device)
        self.sampling_params = sampling_params
        self.batch_size = batch_size if batch_size is not None else 1
        self.tokenizer = tokenizer
        self.special_token_ids = set(self.tokenizer.all_special_ids)

        self.embed_model = embed_model
        self.embed_model.eval().to(self.device)
        self.embed_tokenizer = embed_tokenizer
        if self.embed_tokenizer.pad_token is None:
            self.embed_tokenizer.pad_token = self.embed_tokenizer.eos_token

        self.n_players = len(self.tokenizer.encode(self.sequence, add_special_tokens=False))

        if baseline_id is not None:
            self.baseline = baseline_id
        else:
            self.baseline = tokenizer.encode(" ", add_special_tokens=False)[0]

        original_sequence = self.mask_input(torch.ones(self.n_players))
        original_output = self.model_generate(original_sequence)[0]
        self.original_embed = self.embed(original_output)

        normalization_value = None
        if normalize:
            empty_coalition_value = self.value_function(np.zeros(self.n_players))
            normalization_value = float(empty_coalition_value[0])

        super().__init__(
            n_players=self.n_players,
            normalize=normalize,
            normalization_value=normalization_value,
            verbose=verbose,
        )

    def mask_input(self, coalition: np.ndarray) -> BatchEncoding:
        """Create an imputed sequence for a coalition.

        Args:
            coalition: Boolean or binary array of shape ``(n_players,)`` indicating
                which tokens are active.

        Raises:
            ValueError: If coalition is the wrong size.

        Returns:
            sequence_masked: Tokenized sequence with inactive tokens replaced
                by ``baseline``.
        """
        if len(coalition) != self.n_players:
            msg = "Coalition must be the same size as number of non-special tokens."
            raise ValueError(msg)
        sequence_masked = self.tokenizer(self.sequence, return_tensors="pt").to(self.device)
        c = 0
        for s, token_id in enumerate(sequence_masked.input_ids[0].tolist()):
            if token_id in self.special_token_ids:
                pass
            else:
                if not coalition[c]:
                    sequence_masked.input_ids[0, s] = self.baseline
                c += 1
        return sequence_masked

    def mask_input_batch(self, coalitions: np.ndarray) -> BatchEncoding:
        """Create a batched masked input for multiple coalitions.

        All masked sequences stem from the same input sequence and therefore
        have identical length, so they can be stacked without padding.

        Args:
            coalitions: Coalition matrix of shape ``(n_coalitions, n_players)``.

        Returns:
            A dict with batched ``input_ids`` and ``attention_mask`` tensors of
            shape ``(n_coalitions, sequence_length)``.
        """
        masked = [self.mask_input(coalition) for coalition in coalitions]
        return {
            "input_ids": torch.cat([m.input_ids for m in masked], dim=0),
            "attention_mask": torch.cat([m.attention_mask for m in masked], dim=0),
        }

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        """Evaluate coalition values by generating text from masked input sequences.

        For each coalition, the input sequence is masked (inactive tokens are replaced
        by ``baseline``), the ``model`` generates an output for the masked sequence, and
        that output is embedded with ``embed_model``. The coalition value is the cosine
        similarity between this embedding and ``original_embed``, the embedding of the
        output generated for the unmasked sequence. Coalitions are processed in chunks
        of ``batch_size``.

        Args:
            coalitions: Boolean or binary coalition matrix of shape
                ``(n_coalitions, n_players)``. A one-dimensional array of shape
                ``(n_players,)`` is interpreted as a single coalition.

        Returns:
            One-dimensional NumPy array of shape ``(n_coalitions,)`` containing the
            cosine similarity between the generated output for each coalition and the
            original output, in the same order as the input coalitions.
        """
        if coalitions.ndim == 1:
            coalitions = coalitions.reshape(1, -1)

        coalition_values = []

        for start in range(0, len(coalitions), self.batch_size):
            coalition_batch = coalitions[start : start + self.batch_size]
            batched_input = self.mask_input_batch(coalition_batch)
            model_outputs = self.model_generate(batched_input)
            output_embeds = self.embed(model_outputs)
            batch_values = self.similarity(self.original_embed, output_embeds)
            coalition_values.extend(coalition_value.item() for coalition_value in batch_values)
        return np.array(coalition_values)

    def embed(
        self, texts: str | list[str], prompt: str = "task: sentence similarity | query: "
    ) -> torch.Tensor:
        """Embed one or more texts into vectors using ``embed_model``.

        Args:
            texts: A single string or a list of strings to embed.
            prompt: Instruction prefix prepended to every text before encoding. Should
                match the prefix format expected by ``embed_model``.
                Defaults to a sentence-similarity query prefix used by EmbeddingGemma.

        Returns:
            Tensor of shape ``(n_texts, embedding_dim)`` with L2-normalized embeddings
        """
        if isinstance(texts, str):
            texts = [texts]
        inputs = self.embed_tokenizer(
            [prompt + text for text in texts],
            return_tensors="pt",
            padding=True,
        ).to(self.device)
        with torch.no_grad():
            hidden = self.embed_model(**inputs).last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1)
        emb = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
        emb = emb.float()
        return f.normalize(emb, dim=-1)

    def similarity(self, e1: torch.Tensor, e2: torch.Tensor) -> torch.Tensor:
        """Compute cosine similarity between two batches of embeddings.

        Args:
            e1: Embedding tensor of shape ``(n, embedding_dim)`` or ``(1, embedding_dim)``.
            e2: Embedding tensor broadcastable against ``e1``.

        Returns:
            Tensor of shape ``(n,)`` with one similarity per pair, in ``[-1, 1]``.
        """
        return f.cosine_similarity(e1, e2, dim=-1)

    def model_generate(self, inputs: BatchEncoding) -> list[str]:
        """Generate text continuations for a batch of input sequences.

        Runs ``model.generate`` using ``sampling_params`` if provided, otherwise
        greedy decoding with ``max_new_tokens=256``. The prompt is stripped from the result,
        so only the newly generated tokens are decoded.

        Args:
            inputs: :class:`~transformers.BatchEncoding` with ``input_ids`` and
                ``attention_mask`` tensors of shape ``(n_sequences, sequence_length)``

        Returns:
            List of ``n_sequences`` decoded strings with special tokens removed.
        """
        with torch.no_grad():
            if self.sampling_params is None:
                model_output = self.model.generate(**inputs, max_new_tokens=256, do_sample=False)
            else:
                model_output = self.model.generate(**inputs, **self.sampling_params)
        input_length = inputs["input_ids"].shape[1]
        generated_only = model_output[:, input_length:]
        return [
            self.tokenizer.decode(sequence, skip_special_tokens=True) for sequence in generated_only
        ]
