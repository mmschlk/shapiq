"""Text imputer for coalition-based text explanations."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

try:
    import torch
except ImportError as err:
    from ._error import _text_import_error
    raise _text_import_error from err

from ..base import Imputer
from .callables import (
    CausalLMCallable,
    EncoderClassifierCallable,
    Seq2SeqCallable,
)
from .perturbations import (
    AttentionMaskPerturbation,
    BasePerturbationStrategy,
    MLMInfillingPerturbation,
    create_perturbation_strategy,
)
from .players import (
    BasePlayerStrategy,
    create_player_strategy,
)

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

# =============================================================================
# TEXT IMPUTER
# =============================================================================


class TextImputer(Imputer):
    """Coalition-based text imputer for model-agnostic Shapley explanations.

    ``TextImputer`` combines three independent components:
    - a player strategy, which chooses the text features to explain;
    - a perturbation strategy, which represents missing features;
      - a target callable, which maps perturbed text to a scalar model score.

    The resulting object is callable with a coalition matrix and can therefore
    be passed directly to shapiq approximators. A coalition entry of ``1``
    keeps a player; ``0`` marks it as missing.

    For ordinary perturbation strategies, each coalition is evaluated once.
    For ``MLMInfillingPerturbation``, the imputer evaluates multiple sampled
    infillings and returns their average score, approximating ``E[f(X) | X_S]``.

    Parameters:
    model:
        Hugging Face model whose output is explained.
    tokenizer:
        Tokenizer associated with ``model``.
    text:
        Original text instance to explain.

    player_level: Player granularity
        ``"subword"``, ``"word"``, ``"named_entity"``, ``"chunk"``, or ``"sentence"``.

    perturbation_type: Missing-player strategy
        ``"mask"``, ``"pad"``, ``"removal"``, ``"neutral"``, ``"wordnet_neutral"``, or ``"mlm_infilling"``.

    model_type: Target-model interface
        ``"encoder_classifier"``, ``"causal_lm"``, or ``"seq2seq"``. Seq2seq is currently a placeholder.

    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        text: str,
        *,
        batch_size: int = 16,
        device: str | None = None,
        # ---------------------------------------------------------------------
        # encoder classifier settings
        # ---------------------------------------------------------------------
        class_idx: int = 1,
        output_type: str = "logit",
        # ---------------------------------------------------------------------
        # causal LM settings
        # ---------------------------------------------------------------------
        target_label: str = "good",
        prompt_template: str = ("Review: {text}\n\nSentiment:"),
        # ---------------------------------------------------------------------
        # Seq2Seq settings
        # ---------------------------------------------------------------------
        normalize_target_logprob: bool = True,
        # ---------------------------------------------------------------------
        # architecture selection
        # ---------------------------------------------------------------------
        player_level: str = "word",
        perturbation_type: str = "mask",
        player_strategy: BasePlayerStrategy | None = None,
        perturbation_strategy: BasePerturbationStrategy | None = None,
        # ---------------------------------------------------------------------
        # MLM infilling settings
        # ---------------------------------------------------------------------
        mlm_model_name: str = "bert-base-uncased",
        mlm_num_samples: int = 100,
        # ---------------------------------------------------------------------
        # Generalize target callable support.
        # ---------------------------------------------------------------------
        model_type: str = "encoder_classifier",
    ) -> None:
        """Initialize the Text Imputer."""
        self.model = model
        self.tokenizer = tokenizer
        self.text = text
        self.batch_size = batch_size

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"

            elif torch.backends.mps.is_available():
                device = "mps"

            else:
                device = "cpu"
        self.device = device

        if not hasattr(self.model, "hf_device_map"):
            self.model = self.model.to(self.device)

        self.model.eval()

        # =============================================================================
        # PLAYER STRATEGY
        # =============================================================================

        if player_strategy is None:
            player_strategy = create_player_strategy(
                level=player_level, text=text, tokenizer=tokenizer
            )
        super().__init__(
            model=model,
            data=np.empty((1, player_strategy.n_players)),
        )

        self.player_level = player_level
        self.player_strategy = player_strategy
        self.model_type = model_type

        # =============================================================================
        # PERTURBATION STRATEGY
        # =============================================================================

        if perturbation_strategy is None:
            perturbation_strategy = create_perturbation_strategy(
                strategy=perturbation_type,
                tokenizer=tokenizer,
                mlm_model_name=mlm_model_name,
                mlm_num_samples=mlm_num_samples,
                device=self.device,
            )

        self.perturbation_type = perturbation_type
        self.perturbation_strategy = perturbation_strategy

        # MLM infilling currently supports only word, named-entity, and chunk players.

        if isinstance(
            self.perturbation_strategy,
            MLMInfillingPerturbation,
        ) and self.player_level in {
            "sentence",
            "subword",
        }:
            msg = "MLMInfillingPerturbation currently supports only word, named-entity, and chunk players."
            raise ValueError(msg)

        # =============================================================================
        # TARGET CALLABLE
        # =============================================================================

        if model_type == "encoder_classifier":
            self.target_callable = EncoderClassifierCallable(
                model=model,
                tokenizer=tokenizer,
                device=device,
                class_idx=class_idx,
                output_type=output_type,
            )

        elif model_type == "seq2seq":
            self.target_callable = Seq2SeqCallable(
                model=model,
                tokenizer=tokenizer,
                device=self.device,
                target_label=target_label,
                prompt_template=prompt_template,
                normalize=normalize_target_logprob,
            )

        elif model_type == "causal_lm":
            self.target_callable = CausalLMCallable(
                model=model,
                tokenizer=tokenizer,
                device=self.device,
                target_label=target_label,
                prompt_template=prompt_template,
            )

        else:
            msg = "model_type must be one of:\n- 'encoder_classifier'\n- 'causal_lm'\n- 'seq2seq'"
            raise ValueError(msg)

    def coalition_to_text(
        self,
        coalition: np.ndarray,
    ) -> str:
        """Convert coalition into perturbed text."""
        return self.player_strategy.coalition_to_text(coalition, self.perturbation_strategy)

    def _coalitions_to_texts(
        self,
        coalitions: np.ndarray,
    ) -> list[str]:
        """Convert coalition matrix into perturbed texts."""
        return [self.coalition_to_text(coalition) for coalition in coalitions]

    def _predict_batch(
        self,
        texts: list[str],
    ) -> np.ndarray:
        """Run model-family-specific inference."""
        return self.target_callable.predict(texts)

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

    def _batched_predict_from_inputs(
        self,
        inputs: list[dict[str, torch.Tensor]],
    ) -> np.ndarray:
        """Predict from pre-tokenized model inputs in batches."""
        all_scores = []

        for start in range(0, len(inputs), self.batch_size):
            batch = inputs[start : start + self.batch_size]
            batch_scores = self.target_callable.predict_from_inputs(batch)
            all_scores.append(batch_scores)

        return np.concatenate(all_scores)

    def value_function(
        self,
        coalitions: np.ndarray,
    ) -> np.ndarray:
        """Evaluate one or more coalitions.

        For standard perturbations, each coalition is converted to one perturbed text and scored once.
        For MLM infilling, the process is repeated ``mlm_num_samples`` times with fresh sampled replacements;
        the returned value
        is the mean score across samples.
        """
        coalitions = np.asarray(coalitions)

        if coalitions.ndim == 1:
            coalitions = coalitions.reshape(1, -1)

        if coalitions.shape[1] != self.n_features:
            msg = f"Expected coalition width {self.n_features}, got {coalitions.shape[1]}"
            raise ValueError(msg)

        if isinstance(
            self.perturbation_strategy,
            MLMInfillingPerturbation,
        ):
            num_samples = self.perturbation_strategy.num_samples
            all_scores = []

            # Stored for debugging and demonstrations of sampled MLM infillings.
            self._last_generated_texts = []

            for _ in range(num_samples):
                self.perturbation_strategy.clear_cache()
                texts = self._coalitions_to_texts(coalitions)
                self._last_generated_texts.extend(texts)
                scores = self._batched_predict(texts)
                all_scores.append(scores)

            all_scores = np.stack(all_scores, axis=0)
            return np.mean(all_scores, axis=0)

        if isinstance(self.perturbation_strategy, AttentionMaskPerturbation):
            players = self.player_strategy.get_players()

            masked_inputs = self.perturbation_strategy.evaluate(
                players=players,
                coalitions=coalitions,
                model_type=self.model_type,
                prompt_template=cast(
                    "str | None",
                    getattr(self.target_callable, "prompt_template", None),
                ),
                player_separator="" if self.player_level == "subword" else " ",
            )

            return self._batched_predict_from_inputs(masked_inputs)

        texts = self._coalitions_to_texts(coalitions)
        return self._batched_predict(texts)

    def full_prediction(self) -> float:
        """Score of original unperturbed text."""
        full_coalition = np.ones((1, self.n_features), dtype=bool)
        score = self.value_function(full_coalition)[0]
        return float(score)
