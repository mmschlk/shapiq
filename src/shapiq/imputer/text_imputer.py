"""Backward-compatible imports for the text imputer."""

from .text.imputer import TextImputer

from .text.players import (
    BasePlayerStrategy,
    ChunkPlayerStrategy,
    NamedEntityPlayerStrategy,
    PLAYER_STRATEGIES,
    SentencePlayerStrategy,
    SubwordPlayerStrategy,
    WordPlayerStrategy,
    create_player_strategy,
)

from .text.perturbations import (
    BasePerturbationStrategy,
    MaskTokenPerturbation,
    MLMInfillingPerturbation,
    NeutralPerturbation,
    PERTURBATION_STRATEGIES,
    PadTokenPerturbation,
    RemovalPerturbation,
    WordNetNeutralPerturbation,
    _get_neutral_replacement,
    _penn_to_wn,
    _require_nltk_resource,
    create_perturbation_strategy,
)

from .text.tensor_perturbation import (
    BaseTensorPerturbationStrategy,
    TENSOR_PERTURBATION_STRATEGIES,
    AttentionMaskPerturbation,
    create_tensor_perturbation_strategy,
)

from .text.callables import (
    BaseTargetCallable,
    CausalLMCallable,
    EncoderClassifierCallable,
    Seq2SeqCallable,
)