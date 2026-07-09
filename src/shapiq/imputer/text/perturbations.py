"""Perturbation strategies used by the TextImputer."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, cast

import numpy as np

try:
    import nltk
    import torch
    from nltk.corpus import wordnet as wn
    from transformers import AutoModelForMaskedLM, AutoTokenizer
except ImportError as err:
    from ._error import _text_import_error
    raise _text_import_error from err

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase
    from transformers.modeling_outputs import BaseModelOutput

def _require_nltk_resource(resource_path: str, download_name: str) -> None:
    """Raise a helpful error if an NLTK resource is not installed."""
    try:
        nltk.data.find(resource_path)
    except LookupError as error:
        try:
            nltk.data.find(f"{resource_path}.zip")
        except LookupError:
            pass
        else:
            return

        msg = (
            f"Missing NLTK resource '{download_name}'. "
            "Install it once with:\n\n"
            "    import nltk\n"
            f"    nltk.download('{download_name}')\n"
        )
        raise LookupError(msg) from error

# =============================================================================
# PERTURBATION STRATEGIES
# =============================================================================


class BasePerturbationStrategy(ABC):
    """Abstract interface for representing missing text players.

    A perturbation strategy receives one missing player and returns replacement text.
    Simple strategies can ignore coalition-level context, whereas MLM infilling
    requires it to generate replacements consistently.
    """

    @abstractmethod
    def perturb(
        self,
        player: str,
        *,
        context: dict | None = None,
    ) -> str:
        """Return a replacement for one missing player.

        Parameters:
        player:
            Original text player that is absent from the coalition.
        context:
            Optional coalition-level information. When provided, it may contain
            ``players``, ``coalition``, and ``mask_index``. MLM infilling requires
            this context; simple replacement strategies ignore it.

        Returns:
        str:
            Replacement text. Returning an empty string removes the player.
        """

# =============================================================================
# [MASK] replacement
# =============================================================================


class MaskTokenPerturbation(BasePerturbationStrategy):
    """Replace a missing player with the tokenizer's mask token."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
    ) -> None:
        """Initialize mask-token replacement from the provided tokenizer."""
        self.mask_token = tokenizer.mask_token

        if self.mask_token is None:
            msg = (
                "Tokenizer does not define a mask token. "
                "MaskTokenPerturbation requires a masked language model tokenizer."
            )
            raise ValueError(msg)

    def perturb(
        self,
        player: str,
        *,
        context: dict | None = None,
    ) -> str:
        """Replace missing words with [MASK]."""
        del player
        del context
        return self.mask_token


# =============================================================================
# [PAD] replacement
# =============================================================================


class PadTokenPerturbation(BasePerturbationStrategy):
    """Replace missing players with the tokenizer's PAD token."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
    ) -> None:
        """Initialize PAD replacement strategy."""
        self.pad_token = tokenizer.pad_token

        if self.pad_token is None:
            msg = f"{tokenizer.__class__.__name__} does not define a pad token."
            raise ValueError(msg)

    def perturb(
        self,
        player: str,
        *,
        context: dict | None = None,
    ) -> str:
        """Return the PAD token."""
        del player
        del context
        return self.pad_token


# =============================================================================
# REMOVAL PERTURBATION
# =============================================================================


class RemovalPerturbation(BasePerturbationStrategy):
    """Remove a player by replacing it with an empty string."""

    def perturb(
        self,
        player: str,
        *,
        context: dict | None = None,
    ) -> str:
        """Remove a player by replacing it with an empty string."""
        del player
        del context
        """Return empty string."""
        return ""


# =============================================================================
# NEUTRAL PERTURBATION
# =============================================================================


class NeutralPerturbation(BasePerturbationStrategy):
    """Replace missing players with neutral placeholder text."""

    def __init__(
        self,
        neutral_text: str = "something",
    ) -> None:
        """Replace missing players with neutral placeholder text."""
        self.neutral_text = neutral_text

    def perturb(
        self,
        player: str,
        *,
        context: dict | None = None,
    ) -> str:
        """Return neutral replacement text."""
        del player
        del context
        return self.neutral_text


# =============================================================================
# WORDNET NEUTRAL PERTURBATION
# =============================================================================


def _penn_to_wn(tag: str) -> str | None:
    """Map a Penn Treebank POS tag to a WordNet POS tag when available."""
    if tag.startswith("N"):
        return wn.NOUN

    if tag.startswith("V"):
        return wn.VERB

    if tag.startswith("J"):
        return wn.ADJ

    if tag.startswith("R"):
        return wn.ADV

    return None


def _get_neutral_replacement(word: str, pos_tag: str) -> str:
    """Return a broad WordNet hypernym or a neutral fallback.

    The first WordNet synset and its first hypernym are used as a lightweight,
    deterministic approximation of a semantically broader replacement.
    If no suitable mapping exists, ``"something"`` is returned.
    """
    wn_pos = _penn_to_wn(pos_tag)

    if wn_pos is None:
        return "something"

    synsets = wn.synsets(word, pos=wn_pos)

    if not synsets:
        return "something"

    hypernyms = synsets[0].hypernyms()

    if not hypernyms:
        return "something"

    replacement = hypernyms[0].lemma_names()[0]

    return replacement.replace("_", " ").split()[0]


class WordNetNeutralPerturbation(BasePerturbationStrategy):
    """WordNet-based semantic neutral replacement."""

    def perturb(
        self,
        player: str,
        *,
        context: dict | None = None,
    ) -> str:
        """Replace a player with a semantic hypernym."""
        del context
        _require_nltk_resource("corpora/wordnet", "wordnet")
        _require_nltk_resource(
            "taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"
        )
        tag = nltk.pos_tag([player])[0][1]

        return _get_neutral_replacement(player, tag)


# =============================================================================
# MLM Infilling Perturbation
# support:
# - word-level players
# - named-entity-level players
# - chunk-level players
# =============================================================================


class MLMInfillingPerturbation(BasePerturbationStrategy):
    """Replace missing player spans with samples from a masked language model.

    For a coalition, all missing players are masked simultaneously and an MLM predicts replacements conditioned on the remaining players.
    Replacements are cached per coalition so that all missing players in the same coalition are generated from one MLM forward pass.
    This strategy currently supports player strategies whose players represent contiguous text spans:
    - ``WordPlayerStrategy``
    - ``NamedEntityPlayerStrategy``
    - ``ChunkPlayerStrategy``

    It does not support subword or sentence players in the current version. Subword masking can produce invalid token fragments,
    while sentence-level masking is not a suitable input representation for the current MLM setup.

    Parameters:
    model_name:
        Hugging Face masked-language-model checkpoint used for infilling.
    device:
        Device on which the MLM is evaluated.
    num_samples:
        Number of independently sampled infillings used by ``TextImputer`` to estimate a coalition value by Monte Carlo averaging.

    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        device: str = "cpu",
        num_samples: int = 100,
    ) -> None:
        """Initialize MLM-based infilling strategy."""
        self.tokenizer = cast(
            "PreTrainedTokenizerBase",
            AutoTokenizer.from_pretrained(model_name),
        )
        self.model_name = model_name
        self.device = device
        model = AutoModelForMaskedLM.from_pretrained(model_name)

        model = model.to(device)

        self.model = cast(
            "PreTrainedModel",
            model,
        )
        self.model.eval()

        self.mask_token = self.tokenizer.mask_token

        if self.mask_token is None:
            msg = f"{model_name} does not define a mask token."
            raise ValueError(msg)

        self._cache: dict[tuple, dict[int, str]] = {}

        self.num_samples = num_samples

    def clear_cache(self) -> None:
        """Discard cached replacements before generating a new MLM sample.

        ``TextImputer`` calls this once per Monte Carlo sample.Within one sample,
        the cache ensures that all missing players of the same coalitionuse replacements from the same MLM forward pass.

        """
        self._cache.clear()

    def _build_cache_key(
        self,
        players: list[str],
        coalition: np.ndarray,
    ) -> tuple:
        """Create a stable cache key for one player sequence and coalition."""
        return (tuple(players), tuple(coalition.tolist()))

    def _predict_masks(
        self,
        players: list[str],
        coalition: np.ndarray,
    ) -> dict[int, str]:
        """Predict one replacement for every missing player in a coalition.

        All absent players are replaced by the MLM mask token at once. Candidate
        tokens that decode to special tokens, empty strings, or WordPiece continuations
        are rejected. If no valid candidate is sampled after a fixed number of attempts,
        the neutral fallback ``"something"`` is used.

        """
        masked_players = []

        for keep, player in zip(coalition, players, strict=False):
            if keep:
                masked_players.append(player)

            else:
                masked_players.append(self.mask_token)

        text = " ".join(masked_players)

        encoded = self.tokenizer(text, return_tensors="pt")

        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = self.model(**encoded)

        logits = outputs.logits

        mask_positions = (encoded["input_ids"][0] == self.tokenizer.mask_token_id).nonzero(
            as_tuple=True
        )[0]

        replacements = {}

        for player_idx, token_pos in zip(np.where(coalition == 0)[0], mask_positions, strict=False):
            probs = torch.softmax(logits[0, token_pos], dim=-1)

            replacement = "something"
            max_sampling_attempts = 100

            for _ in range(max_sampling_attempts):
                candidate_id = int(
                    torch.multinomial(
                        probs,
                        num_samples=1,
                    ).item()
                )

                token = cast(
                    "str",
                    self.tokenizer.decode(
                        [candidate_id],
                        skip_special_tokens=True,
                    ),
                ).strip()

                if token == "":
                    continue

                if token in {
                    self.tokenizer.cls_token,
                    self.tokenizer.sep_token,
                    self.tokenizer.pad_token,
                    self.tokenizer.mask_token,
                }:
                    continue

                if token.startswith("##"):
                    continue

                replacement = token
                break

            replacements[player_idx] = replacement

        return replacements

    def perturb(
        self,
        player: str,
        *,
        context: dict | None = None,
    ) -> str:
        """Return the cached or newly generated MLM replacement for one player.

        The ``context`` dictionary must contain the complete player list, the
        coalition, and the index of the currently missing player. This allows one
        coalition-level MLM prediction to be shared across all missing players.

        """
        if context is None:
            msg = "MLMInfillingPerturbation requires context."
            raise ValueError(msg)

        players = context["players"]
        coalition = np.asarray(context["coalition"])
        mask_index = context["mask_index"]

        cache_key = self._build_cache_key(players, coalition)

        if cache_key not in self._cache:
            self._cache[cache_key] = self._predict_masks(players, coalition)

        replacements = self._cache[cache_key]

        return replacements.get(mask_index, player)


# =============================================================================
# ATTENTION PERTURBATION
# ============================================================================
class AttentionMaskPerturbation(BasePerturbationStrategy):
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

    def perturb(
        self,
        player: str,
        *,
        context: dict | None = None,
    ) -> str:
        """Return the original player.

        This method is required by the perturbation-strategy interface but is not
        used by attention masking.
        """
        del context
        return player

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


# =============================================================================
# PERTURBATION DICTIONARY AND FACTORY
# =============================================================================

PERTURBATION_STRATEGIES = {
    "mask": MaskTokenPerturbation,
    "pad": PadTokenPerturbation,
    "removal": RemovalPerturbation,
    "neutral": NeutralPerturbation,
    "wordnet_neutral": WordNetNeutralPerturbation,
    "mlm_infilling": MLMInfillingPerturbation,
    "attention_mask": AttentionMaskPerturbation,
}


def create_perturbation_strategy(
    strategy: str,
    tokenizer: PreTrainedTokenizerBase,
    mlm_model_name: str = "bert-base-uncased",
    mlm_num_samples: int = 100,
    device: str = "cpu",
) -> BasePerturbationStrategy:
    """Create a perturbation strategy from a string identifier."""
    if strategy not in PERTURBATION_STRATEGIES:
        msg = (
            f"Unknown perturbation strategy: {strategy}. "
            f"Available strategies: {list(PERTURBATION_STRATEGIES)}"
        )

        raise ValueError(msg)
    if strategy == "mask":
        return MaskTokenPerturbation(tokenizer)

    if strategy == "pad":
        return PadTokenPerturbation(tokenizer)

    if strategy == "attention_mask":
        return AttentionMaskPerturbation(tokenizer)

    if strategy == "removal":
        return RemovalPerturbation()

    if strategy == "neutral":
        return NeutralPerturbation()

    if strategy == "wordnet_neutral":
        return WordNetNeutralPerturbation()

    if strategy == "mlm_infilling":
        return MLMInfillingPerturbation(
            model_name=mlm_model_name,
            device=device,
            num_samples=mlm_num_samples,
        )
    msg_0 = f"Unhandled perturbation strategy: {strategy}"
    raise RuntimeError(msg_0)

