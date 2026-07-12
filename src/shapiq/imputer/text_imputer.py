"""Backward-compatible imports for the text imputer."""

from __future__ import annotations

from .text.callables import (
    BaseTargetCallable,
    CausalLMCallable,
    EncoderClassifierCallable,
    Seq2SeqCallable,
)
from .text.imputer import TextImputer
from .text.perturbations import (
    PERTURBATION_STRATEGIES,
    BasePerturbationStrategy,
    MaskTokenPerturbation,
    MLMInfillingPerturbation,
    NeutralPerturbation,
    PadTokenPerturbation,
    RemovalPerturbation,
    WordNetNeutralPerturbation,
    _get_neutral_replacement,
    _penn_to_wn,
    _require_nltk_resource,
    create_perturbation_strategy,
)
from .text.players import (
    PLAYER_STRATEGIES,
    BasePlayerStrategy,
    ChunkPlayerStrategy,
    NamedEntityPlayerStrategy,
    SentencePlayerStrategy,
    SubwordPlayerStrategy,
    WordPlayerStrategy,
    create_player_strategy,
)
from .text.tensor_perturbation import (
    TENSOR_PERTURBATION_STRATEGIES,
    AttentionMaskPerturbation,
    BaseTensorPerturbationStrategy,
    create_tensor_perturbation_strategy,
)

__all__ = [
    "PERTURBATION_STRATEGIES",
    "PLAYER_STRATEGIES",
    "TENSOR_PERTURBATION_STRATEGIES",
    "AttentionMaskPerturbation",
    "BasePerturbationStrategy",
    "BasePlayerStrategy",
    "BaseTargetCallable",
    "BaseTensorPerturbationStrategy",
    "CausalLMCallable",
    "ChunkPlayerStrategy",
    "EncoderClassifierCallable",
    "MLMInfillingPerturbation",
    "MaskTokenPerturbation",
    "NamedEntityPlayerStrategy",
    "NeutralPerturbation",
    "PadTokenPerturbation",
    "RemovalPerturbation",
    "SentencePlayerStrategy",
    "Seq2SeqCallable",
    "SubwordPlayerStrategy",
    "TextImputer",
    "WordNetNeutralPerturbation",
    "WordPlayerStrategy",
    "_get_neutral_replacement",
    "_penn_to_wn",
    "_require_nltk_resource",
    "create_perturbation_strategy",
    "create_player_strategy",
    "create_tensor_perturbation_strategy",
]
