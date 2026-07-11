"""Text imputer for coalition-based text explanations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

import numpy as np

try:
    import torch
except ImportError as err:
    from ._error import _text_import_error

    raise _text_import_error from err

from shapiq.imputer.base import Imputer

from .callables import (
    CausalLMCallable,
    EncoderClassifierCallable,
    Seq2SeqCallable,
)
from .perturbations import (
    BasePerturbationStrategy,
    MLMInfillingPerturbation,
    create_perturbation_strategy,
)
from .players import (
    BasePlayerStrategy,
    create_player_strategy,
)
from .tensor_perturbation import (
    TENSOR_PERTURBATION_STRATEGIES,
    BaseTensorPerturbationStrategy,
    create_tensor_perturbation_strategy,
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
    - either a text perturbation strategy, which creates perturbed strings,
      or a tensor perturbation strategy, which creates model-ready inputs;
    - a target callable, which maps the perturbed representation to a scalar
      model score.

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
        Text perturbations include ``"mask"``, ``"pad"``, ``"removal"``,
        ``"neutral"``, ``"wordnet_neutral"``, and ``"mlm_infilling"``.
        Tensor perturbations include ``"attention_mask"``. Tensor perturbations
        do not create perturbed strings; they build model-ready inputs directly.

    model_type: Target-model interface
        ``"encoder_classifier"``, ``"causal_lm"``, and ``"seq2seq"``.

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
        tensor_perturbation_strategy: BaseTensorPerturbationStrategy | None = None,
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

        if perturbation_strategy is not None and tensor_perturbation_strategy is not None:
            msg = (
                "Only one of perturbation_strategy and tensor_perturbation_strategy "
                "can be provided."
            )
            raise ValueError(msg)

        self.perturbation_type = perturbation_type
        self.perturbation_mode: Literal["text", "tensor"]

        if perturbation_type in TENSOR_PERTURBATION_STRATEGIES:
            if perturbation_strategy is not None:
                msg = (
                    f"perturbation_type={perturbation_type!r} is a tensor perturbation, "
                    "so perturbation_strategy must be None. "
                    "Use tensor_perturbation_strategy instead."
                )
                raise ValueError(msg)

            if tensor_perturbation_strategy is None:
                tensor_perturbation_strategy = create_tensor_perturbation_strategy(
                    strategy=perturbation_type,
                    tokenizer=tokenizer,
                )

            self.perturbation_mode = "tensor"
            self.perturbation_strategy = None
            self.tensor_perturbation_strategy = tensor_perturbation_strategy

        else:
            if tensor_perturbation_strategy is not None:
                msg = (
                    f"perturbation_type={perturbation_type!r} is a text perturbation, "
                    "so tensor_perturbation_strategy must be None. "
                    "Use perturbation_strategy instead."
                )
                raise ValueError(msg)

            if perturbation_strategy is None:
                perturbation_strategy = create_perturbation_strategy(
                    strategy=perturbation_type,
                    tokenizer=tokenizer,
                    mlm_model_name=mlm_model_name,
                    mlm_num_samples=mlm_num_samples,
                    device=self.device,
                )

            self.perturbation_mode = "text"
            self.perturbation_strategy = perturbation_strategy
            self.tensor_perturbation_strategy = None

        # MLM infilling currently supports only word, named-entity, and chunk players.

        if (
            self.perturbation_mode == "text"
            and isinstance(self.perturbation_strategy, MLMInfillingPerturbation)
            and self.player_level in {"sentence", "subword"}
        ):
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

        self._compute_reference_predictions()

    def coalition_to_text(
        self,
        coalition: np.ndarray,
    ) -> str:
        """Convert one coalition into a perturbed text.

        This method is only valid for text perturbation strategies. Tensor
        perturbations build model-ready inputs directly and must not be routed
        through this string-based path.
        """
        if self.perturbation_mode != "text":
            msg = (
                "coalition_to_text() can only be used with text perturbation strategies. "
                f"Got perturbation_mode={self.perturbation_mode!r}."
            )
            raise RuntimeError(msg)

        if self.perturbation_strategy is None:
            msg = "perturbation_strategy is required in text perturbation mode."
            raise RuntimeError(msg)

        return self.player_strategy.coalition_to_text(
            coalition,
            self.perturbation_strategy,
        )

    def _coalitions_to_texts(
        self,
        coalitions: np.ndarray,
    ) -> list[str]:
        """Convert coalition matrix into perturbed texts."""
        if self.perturbation_mode != "text":
            msg = (
                "_coalitions_to_texts() can only be used with text perturbation strategies. "
                f"Got perturbation_mode={self.perturbation_mode!r}."
            )
            raise RuntimeError(msg)

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

    def _evaluate_coalitions(
        self,
        coalitions: np.ndarray,
    ) -> np.ndarray:
        if self.perturbation_mode == "text" and isinstance(
            self.perturbation_strategy, MLMInfillingPerturbation
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
        if self.perturbation_mode == "tensor":
            if self.tensor_perturbation_strategy is None:
                msg = "tensor_perturbation_strategy is required in tensor perturbation mode."
                raise RuntimeError(msg)
            players = self.player_strategy.get_players()

            masked_inputs = self.tensor_perturbation_strategy.evaluate(
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

    def value_function(
        self,
        coalitions: np.ndarray,
    ) -> np.ndarray:
        """Evaluate one or more coalitions.

        For text perturbations, each coalition is converted to one perturbed text
        and scored once. For MLM infilling, this process is repeated
        ``mlm_num_samples`` times with fresh sampled replacements, and the returned
        value is the mean score across samples.

        For tensor perturbations, coalitions are converted directly into model-ready
        inputs and scored through the target callable's tensor-input interface.
        """
        coalitions = np.asarray(coalitions)

        if coalitions.ndim == 1:
            coalitions = coalitions.reshape(1, -1)

        if coalitions.shape[1] != self.n_features:
            msg = f"Expected coalition width {self.n_features}, got {coalitions.shape[1]}"
            raise ValueError(msg)

        scores = self._evaluate_coalitions(coalitions)
        empty_mask = ~np.any(coalitions, axis=1)
        scores[empty_mask] = self.empty_prediction
        return scores

    def _compute_reference_predictions(self) -> None:
        self.full_prediction = float(
            self._evaluate_coalitions(self.grand_coalition.reshape(1, -1))[0]
        )

        self.empty_prediction = float(
            self._evaluate_coalitions(self.empty_coalition.reshape(1, -1))[0]
        )
        self.normalization_value = self.empty_prediction
