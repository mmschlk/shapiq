"""Tensor perturbation strategies used by the TextImputer.

Tensor perturbations do not create perturbed strings. Instead, they build
model-ready tensor inputs, such as ``input_ids`` and ``attention_mask``.
They are intentionally separated from text perturbation strategies because
they use a different interface and should not be routed through
``_coalitions_to_texts``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

try:
    import torch
except ImportError as err:
    from ._error import _text_import_error

    raise _text_import_error from err

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

# =============================================================================
# Tensor PERTURBATION STRATEGIES
# =============================================================================


class BaseTensorPerturbationStrategy(ABC):
    """Base class for perturbations that produce model-ready tensor inputs.

    Unlike text perturbation strategies, tensor perturbations do not implement
    a string-in/string-out ``perturb`` method. They directly build model inputs
    for coalitions and are consumed by tensor-based prediction paths.
    """

    @abstractmethod
    def evaluate(
        self,
        players: list[str],
        coalitions: np.ndarray,
        *,
        model_type: str,
        prompt_template: str | None = None,
        player_separator: str = "",
    ) -> list[dict[str, torch.Tensor]]:
        """Build model-ready inputs for coalitions using attention masking.

        For encoder classifiers, players are tokenized as one input sequence. For
        causal LM and seq2seq models, players are inserted into ``prompt_template`` and
        only player tokens inside ``"{text}"`` are maskable.
        """


class AttentionMaskPerturbation(BaseTensorPerturbationStrategy):
    """Build model inputs by masking missing players in the attention mask.

    This perturbation does not create perturbed strings. Instead, it maps
    players to token spans and sets the corresponding attention mask entries
    of missing players to 0.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
    ) -> None:
        """Initialize attention-mask perturbation."""
        self.tokenizer = tokenizer

    @staticmethod
    def build_attention_mask_for_coalition(
        base_attention_mask: torch.Tensor,
        player_spans: list[tuple[int, int]],
        coalition: np.ndarray,
    ) -> torch.Tensor:
        """Build an attention mask for one coalition."""
        coalition = np.asarray(coalition, dtype=bool)

        if len(coalition) != len(player_spans):
            msg = (
                f"Coalition length {len(coalition)} does not match "
                f"number of player spans {len(player_spans)}."
            )
            raise ValueError(msg)

        attention_mask = base_attention_mask.clone()

        for keep, (start, end) in zip(coalition, player_spans, strict=False):
            if not keep:
                attention_mask[..., start:end] = 0

        return attention_mask

    @staticmethod
    def build_tokenized_players(
        players: list[str],
        tokenizer: PreTrainedTokenizerBase,
        player_separator: str = "",
    ) -> tuple[dict[str, torch.Tensor], list[tuple[int, int]]]:
        """Tokenize players into one sequence and record their token spans."""
        all_token_ids: list[int] = []
        player_spans: list[tuple[int, int]] = []

        for idx, player in enumerate(players):
            text_piece = player if idx == 0 else f"{player_separator}{player}"

            token_ids = tokenizer.encode(
                text_piece,
                add_special_tokens=False,
            )

            start = len(all_token_ids)
            all_token_ids.extend(token_ids)
            end = len(all_token_ids)

            player_spans.append((start, end))

        input_ids = torch.tensor(
            [all_token_ids],
            dtype=torch.long,
        )
        attention_mask = torch.ones_like(input_ids)

        return (
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            },
            player_spans,
        )

    @classmethod
    def build_inputs_for_coalitions(
        cls,
        tokenizer: PreTrainedTokenizerBase,
        players: list[str],
        coalitions: np.ndarray,
        player_separator: str = "",
    ) -> list[dict[str, torch.Tensor]]:
        """Build masked model inputs using attention masking.

        Args:
            tokenizer: HuggingFace tokenizer.
            players: Text players to explain.
            coalitions: Coalition matrix of shape ``(n_coalitions, n_players)``.
            player_separator: String inserted between adjacent players before tokenization.

        Returns:
            A list of model input dictionaries. Each dictionary contains:
            - input_ids: Encoded text.
            - attention_mask: Attention mask with missing-player tokens set to 0.
        """
        coalitions = np.asarray(coalitions, dtype=bool)

        if coalitions.ndim == 1:
            coalitions = coalitions.reshape(1, -1)

        encoded, player_spans = cls.build_tokenized_players(
            players=players,
            tokenizer=tokenizer,
            player_separator=player_separator,
        )

        if coalitions.shape[1] != len(player_spans):
            msg = f"Expected coalition width {len(player_spans)}, got {coalitions.shape[1]}."
            raise ValueError(msg)

        masked_inputs: list[dict[str, torch.Tensor]] = []

        for coalition in coalitions:
            attention_mask = cls.build_attention_mask_for_coalition(
                base_attention_mask=encoded["attention_mask"],
                player_spans=player_spans,
                coalition=coalition,
            )

            masked_inputs.append(
                {
                    "input_ids": encoded["input_ids"],
                    "attention_mask": attention_mask,
                },
            )

        return masked_inputs

    @staticmethod
    def build_tokenized_prompt_players(
        players: list[str],
        tokenizer: PreTrainedTokenizerBase,
        prompt_template: str,
        player_separator: str = "",
    ) -> tuple[dict[str, torch.Tensor], list[tuple[int, int]]]:
        """Tokenize prompt-wrapped players and record player token spans.

        This is for causal LM scoring. The prompt template must contain "{text}".
        Only tokens corresponding to players inside "{text}" are maskable.
        Prompt instruction tokens are always kept visible.
        """
        if "{text}" not in prompt_template:
            msg = "prompt_template must contain '{text}'."
            raise ValueError(msg)

        prefix, suffix = prompt_template.split("{text}", maxsplit=1)

        prefix_ids = tokenizer.encode(
            prefix,
            add_special_tokens=False,
        )

        suffix_ids = tokenizer.encode(
            suffix,
            add_special_tokens=False,
        )

        all_token_ids: list[int] = []
        player_spans: list[tuple[int, int]] = []

        all_token_ids.extend(prefix_ids)

        for idx, player in enumerate(players):
            text_piece = player if idx == 0 else f"{player_separator}{player}"

            token_ids = tokenizer.encode(
                text_piece,
                add_special_tokens=False,
            )

            start = len(all_token_ids)
            all_token_ids.extend(token_ids)
            end = len(all_token_ids)

            player_spans.append((start, end))

        all_token_ids.extend(suffix_ids)

        input_ids = torch.tensor(
            [all_token_ids],
            dtype=torch.long,
        )
        attention_mask = torch.ones_like(input_ids)

        return (
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            },
            player_spans,
        )

    @classmethod
    def build_prompt_inputs_for_coalitions(
        cls,
        tokenizer: PreTrainedTokenizerBase,
        players: list[str],
        coalitions: np.ndarray,
        prompt_template: str,
        player_separator: str = "",
    ) -> list[dict[str, torch.Tensor]]:
        """Build causal-LM prompt inputs using attention masking.

        The generated inputs represent prompt_template.format(text=players_text),
        but attention masking is applied only to player tokens inside "{text}".
        """
        coalitions = np.asarray(coalitions, dtype=bool)

        if coalitions.ndim == 1:
            coalitions = coalitions.reshape(1, -1)

        encoded, player_spans = cls.build_tokenized_prompt_players(
            players=players,
            tokenizer=tokenizer,
            prompt_template=prompt_template,
            player_separator=player_separator,
        )

        if coalitions.shape[1] != len(player_spans):
            msg = f"Expected coalition width {len(player_spans)}, got {coalitions.shape[1]}."
            raise ValueError(msg)

        masked_inputs: list[dict[str, torch.Tensor]] = []

        for coalition in coalitions:
            attention_mask = cls.build_attention_mask_for_coalition(
                base_attention_mask=encoded["attention_mask"],
                player_spans=player_spans,
                coalition=coalition,
            )

            masked_inputs.append(
                {
                    "input_ids": encoded["input_ids"],
                    "attention_mask": attention_mask,
                },
            )

        return masked_inputs

    def evaluate(
        self,
        players: list[str],
        coalitions: np.ndarray,
        *,
        model_type: str = "encoder_classifier",
        prompt_template: str | None = None,
        player_separator: str = "",
    ) -> list[dict[str, torch.Tensor]]:
        """Build masked model inputs using attention masking."""
        if model_type == "causal_lm":
            if prompt_template is None:
                msg = "prompt_template is required for causal_lm attention masking."
                raise ValueError(msg)

            return self.build_prompt_inputs_for_coalitions(
                tokenizer=self.tokenizer,
                players=players,
                coalitions=coalitions,
                prompt_template=prompt_template,
                player_separator=player_separator,
            )

        if model_type == "encoder_classifier":
            return self.build_inputs_for_coalitions(
                tokenizer=self.tokenizer,
                players=players,
                coalitions=coalitions,
                player_separator=player_separator,
            )

        if model_type == "seq2seq":
            if prompt_template is None:
                msg = "prompt_template is required for seq2seq attention masking."
                raise ValueError(msg)

            return self.build_prompt_inputs_for_coalitions(
                tokenizer=self.tokenizer,
                players=players,
                coalitions=coalitions,
                prompt_template=prompt_template,
                player_separator=player_separator,
            )

        msg = f"Unknown model_type for attention masking: {model_type}."
        raise ValueError(msg)


TENSOR_PERTURBATION_STRATEGIES: dict[
    str,
    type[BaseTensorPerturbationStrategy],
] = {
    "attention_mask": AttentionMaskPerturbation,
}


def create_tensor_perturbation_strategy(
    strategy: str,
    *,
    tokenizer: PreTrainedTokenizerBase,
) -> BaseTensorPerturbationStrategy:
    """Create a tensor perturbation strategy from a string identifier.

    This factory is intentionally separate from the text perturbation factory
    to avoid mixing string-returning and tensor-returning perturbations.
    """
    if strategy not in TENSOR_PERTURBATION_STRATEGIES:
        msg = (
            f"Unknown tensor perturbation strategy: {strategy}. "
            f"Available strategies: {list(TENSOR_PERTURBATION_STRATEGIES)}"
        )
        raise ValueError(msg)

    if strategy == "attention_mask":
        return AttentionMaskPerturbation(tokenizer=tokenizer)

    msg = f"Unhandled tensor perturbation strategy: {strategy}"
    raise RuntimeError(msg)
