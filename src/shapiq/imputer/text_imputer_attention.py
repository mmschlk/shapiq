"""Utilities for attention-mask-based text imputations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase


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


def build_tokenized_prompt_with_target(
    players: list[str],
    tokenizer: PreTrainedTokenizerBase,
    device: str,
    *,
    prompt_suffix: str = "",
    target_text: str = "",
    player_separator: str = "\n",
) -> tuple[dict[str, torch.Tensor], list[tuple[int, int]], tuple[int, int]]:
    """Tokenize players, suffix, and target text into one sequence."""
    all_token_ids: list[int] = []
    player_spans: list[tuple[int, int]] = []

    separator_ids = tokenizer.encode(
        player_separator,
        add_special_tokens=False,
    )

    for player_idx, player in enumerate(players):
        if player_idx > 0:
            all_token_ids.extend(separator_ids)

        token_ids = tokenizer.encode(
            player,
            add_special_tokens=False,
        )

        start = len(all_token_ids)
        all_token_ids.extend(token_ids)
        end = len(all_token_ids)

        player_spans.append((start, end))

    suffix_ids = tokenizer.encode(
        prompt_suffix,
        add_special_tokens=False,
    )
    all_token_ids.extend(suffix_ids)

    target_ids = tokenizer.encode(
        target_text,
        add_special_tokens=False,
    )

    target_start = len(all_token_ids)
    all_token_ids.extend(target_ids)
    target_end = len(all_token_ids)

    input_ids = torch.tensor(
        [all_token_ids],
        dtype=torch.long,
        device=device,
    )

    attention_mask = torch.ones_like(input_ids, device=device)

    return (
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        },
        player_spans,
        (target_start, target_end),
    )


def attention_mask_value_function(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    players: list[str],
    coalitions: np.ndarray,
    device: str,
    *,
    prompt_suffix: str = "",
    target_text: str = "",
    player_separator: str = "\n",
) -> np.ndarray:
    """Evaluate coalition values using attention masking.

    Args:
        model: Causal language model.
        tokenizer: HuggingFace tokenizer.
        players: Text players to explain.
        coalitions: Coalition matrix of shape ``(n_coalitions, n_players)``.
        device: Device for tensors.
        prompt_suffix: Fixed prompt part after players.
        target_text: Target continuation to score.
        player_separator: Separator inserted between players.

    Returns:
        One score for each coalition.
    """
    coalitions = np.asarray(coalitions, dtype=bool)

    if coalitions.ndim == 1:
        coalitions = coalitions.reshape(1, -1)

    encoded, player_spans, target_span = build_tokenized_prompt_with_target(
        players=players,
        tokenizer=tokenizer,
        device=device,
        prompt_suffix=prompt_suffix,
        target_text=target_text,
        player_separator=player_separator,
    )

    if coalitions.shape[1] != len(player_spans):
        msg = f"Expected coalition width {len(player_spans)}, got {coalitions.shape[1]}."
        raise ValueError(msg)

    scores: list[float] = []

    for coalition in coalitions:
        attention_mask = build_attention_mask_for_coalition(
            base_attention_mask=encoded["attention_mask"],
            player_spans=player_spans,
            coalition=coalition,
        )

        encoded_masked = {
            "input_ids": encoded["input_ids"],
            "attention_mask": attention_mask,
        }
        score = score_target_span(
            model=model,
            encoded=encoded_masked,
            target_span=target_span,
        )

        scores.append(float(score.detach().cpu()))

    return np.asarray(scores, dtype=np.float32)


def score_target_span(
    model: PreTrainedModel,
    encoded: dict[str, torch.Tensor],
    target_span: tuple[int, int],
) -> torch.Tensor:
    """Compute log-probability of a target span in a causal LM sequence."""
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    target_start, target_end = target_span

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    logits = outputs.logits
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    score = torch.tensor(0.0, device=input_ids.device)

    for token_pos in range(target_start, target_end):
        previous_pos = token_pos - 1
        token_id = input_ids[0, token_pos]
        score = score + log_probs[0, previous_pos, token_id]

    return score
