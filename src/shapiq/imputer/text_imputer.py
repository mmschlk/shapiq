"""Generalized text imputer for coalition-based text explanations."""

from __future__ import annotations  # noqa: I001

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
import numpy as np
import torch
import nltk
from nltk.corpus import wordnet as wn
from nltk.tree import Tree

from transformers import AutoModelForMaskedLM, AutoTokenizer

nltk.download("punkt")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger")
nltk.download("averaged_perceptron_tagger_eng")
nltk.download("maxent_ne_chunker")
nltk.download("words")
nltk.download("maxent_ne_chunker_tab")

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

# =============================================================================
# PLAYER STRATEGIES
# =============================================================================


class BasePlayerStrategy(ABC):
    """Abstract interface for defining players."""

    @abstractmethod
    def get_players(self) -> list[str]:
        """Return player list."""

    @abstractmethod
    def coalition_to_text(
        self,
        coalition: np.ndarray,
        perturbation_strategy: BasePerturbationStrategy,
    ) -> str:
        """Convert coalition into perturbed text."""

    @property
    @abstractmethod
    def n_players(self) -> int:
        """Return number of players."""


# =============================================================================
# TO DO:
# Current implementation:
# - sub-word-level players
# - word-level players
# - named-entity-level players
# - chunk-level players
# - sentence-level players

# Future:
# - additional player strategies
# =============================================================================

# =============================================================================
# SUBWORD-LEVEL PLAYER STRATEGY
# =============================================================================


class SubwordPlayerStrategy(BasePlayerStrategy):
    """Tokenizer/subword-level player strategy.

    Uses the provided HuggingFace tokenizer to define players as tokenizer tokens (WordPiece/BPE/SentencePiece, etc.).
    """

    def __init__(
        self,
        text: str,
        tokenizer: PreTrainedTokenizerBase,
    ) -> None:
        """Initialize subword-level player strategy."""
        self.text = text
        self.tokenizer = tokenizer

        self.subwords = tokenizer.tokenize(text)

    def get_players(self) -> list[str]:
        """Return subword players."""
        return self.subwords

    @property
    def n_players(self) -> int:
        """Return number of subword players."""
        return len(self.subwords)

    def coalition_to_text(
        self,
        coalition: np.ndarray,
        perturbation_strategy: BasePerturbationStrategy,
    ) -> str:
        """Convert coalition into perturbed subword text."""
        if len(coalition) != self.n_players:
            msg = f"Coalition length {len(coalition)} does not match n_players={self.n_players}"
            raise ValueError(msg)

        output_tokens: list[str] = []

        for idx, (keep, token) in enumerate(zip(coalition, self.subwords, strict=False)):
            if keep:
                output_tokens.append(token)

            else:
                replacement = perturbation_strategy.perturb(
                    token,
                    context={
                        "players": self.subwords,
                        "coalition": coalition,
                        "mask_index": idx,
                    },
                )

                if replacement != "":
                    output_tokens.append(replacement)

        return self.tokenizer.convert_tokens_to_string(output_tokens)


# =============================================================================
# WORD-LEVEL PLAYER STRATEGY
# =============================================================================


class WordPlayerStrategy(BasePlayerStrategy):
    """Word-level player strategy."""

    def __init__(
        self,
        text: str,
    ) -> None:
        """Initialize word-level player strategy."""
        self.text = text
        self.words = nltk.word_tokenize(text)

    def get_players(self) -> list[str]:
        """Return word players."""
        return self.words

    @property
    def n_players(self) -> int:
        """Return number of word players."""
        return len(self.words)

    def coalition_to_text(
        self,
        coalition: np.ndarray,
        perturbation_strategy: BasePerturbationStrategy,
    ) -> str:
        """Convert coalition into perturbed text."""
        if len(coalition) != self.n_players:
            msg = f"Coalition length {len(coalition)} does not match n_players={self.n_players}"
            raise ValueError(msg)

        output_words = []

        for idx, (keep, word) in enumerate(zip(coalition, self.words, strict=False)):
            if keep:
                output_words.append(word)

            else:
                replacement = perturbation_strategy.perturb(
                    word,
                    context={
                        "players": self.words,
                        "coalition": coalition,
                        "mask_index": idx,
                    },
                )

                if replacement != "":
                    output_words.append(replacement)

        return " ".join(output_words)


# =============================================================================
# NAMED-ENTITY PLAYER STRATEGY
# =============================================================================


class NamedEntityPlayerStrategy(BasePlayerStrategy):
    """Named-entity-level player strategy using NLTK NER."""

    def __init__(
        self,
        text: str,
    ) -> None:
        """Initialize named-entity player strategy."""
        self.text = text

        tokens = nltk.word_tokenize(text)
        pos_tags = nltk.pos_tag(tokens)
        ner_tree = nltk.ne_chunk(pos_tags)

        self.players: list[str] = []

        for node in ner_tree:
            if isinstance(node, Tree):
                entity = " ".join(word for word, _tag in node.leaves())

                self.players.append(entity)

            else:
                word, _tag = node
                self.players.append(word)

    def get_players(self) -> list[str]:
        """Return entity-aware players."""
        return self.players

    @property
    def n_players(self) -> int:
        """Return number of players."""
        return len(self.players)

    def coalition_to_text(
        self,
        coalition: np.ndarray,
        perturbation_strategy: BasePerturbationStrategy,
    ) -> str:
        """Convert coalition into perturbed text."""
        if len(coalition) != self.n_players:
            msg = f"Coalition length {len(coalition)} does not match n_players={self.n_players}"
            raise ValueError(msg)

        output_players: list[str] = []

        for idx, (keep, player) in enumerate(zip(coalition, self.players, strict=False)):
            if keep:
                output_players.append(player)

            else:
                replacement = perturbation_strategy.perturb(
                    player,
                    context={
                        "players": self.players,
                        "coalition": coalition,
                        "mask_index": idx,
                    },
                )

                if replacement != "":
                    output_players.append(replacement)

        return " ".join(output_players)


# =============================================================================
# CHUNK-LEVEL PLAYER STRATEGY
# =============================================================================


class ChunkPlayerStrategy(BasePlayerStrategy):
    """Chunk-level player strategy using POS-based phrase chunking."""

    def __init__(
        self,
        text: str,
    ) -> None:
        """Initialize chunk-level player strategy."""
        self.text = text

        tokens = nltk.word_tokenize(text)
        pos_tags = nltk.pos_tag(tokens)

        grammar = r"""
            NP: {<DT>?<JJ.*>*<NN.*>+}
            VP: {<VB.*><RB.*>*}
        """

        chunker = nltk.RegexpParser(grammar)
        tree = chunker.parse(pos_tags)

        self.chunks: list[str] = []

        for node in tree:
            if isinstance(node, Tree):
                phrase = " ".join(word for word, _tag in node.leaves())
                self.chunks.append(phrase)

            else:
                word, _tag = node
                self.chunks.append(word)

    def get_players(self) -> list[str]:
        """Return chunk players."""
        return self.chunks

    @property
    def n_players(self) -> int:
        """Return number of chunk players."""
        return len(self.chunks)

    def coalition_to_text(
        self,
        coalition: np.ndarray,
        perturbation_strategy: BasePerturbationStrategy,
    ) -> str:
        """Convert coalition into perturbed chunk text."""
        if len(coalition) != self.n_players:
            msg = f"Coalition length {len(coalition)} does not match n_players={self.n_players}"
            raise ValueError(msg)

        output_chunks: list[str] = []

        for idx, (keep, chunk) in enumerate(zip(coalition, self.chunks, strict=False)):
            if keep:
                output_chunks.append(chunk)

            else:
                replacement = perturbation_strategy.perturb(
                    chunk,
                    context={
                        "players": self.chunks,
                        "coalition": coalition,
                        "mask_index": idx,
                    },
                )

                if replacement != "":
                    output_chunks.append(replacement)

        return " ".join(output_chunks)


# =============================================================================
# SENTENCE-LEVEL PLAYER STRATEGY
# =============================================================================


class SentencePlayerStrategy(BasePlayerStrategy):
    """Sentence-level player strategy using NLTK sentence splitting."""

    def __init__(
        self,
        text: str,
    ) -> None:
        """Sentence-level player strategy using NLTK sentence splitting."""
        self.text = text
        self.sentences = nltk.sent_tokenize(text)

    def get_players(self) -> list[str]:
        """Return sentence players."""
        return self.sentences

    @property
    def n_players(self) -> int:
        """Return number of sentence players."""
        return len(self.sentences)

    def coalition_to_text(
        self,
        coalition: np.ndarray,
        perturbation_strategy: BasePerturbationStrategy,
    ) -> str:
        """Convert coalition into perturbed sentence-level text."""
        if len(coalition) != self.n_players:
            msg = f"Coalition length {len(coalition)} does not match n_players={self.n_players}"
            raise ValueError(msg)

        perturbed_sentences = []

        for keep, sentence in zip(coalition, self.sentences, strict=False):
            if keep:
                perturbed_sentences.append(sentence)

            else:
                replacement = perturbation_strategy.perturb(sentence, context=None)

                if replacement != "":
                    perturbed_sentences.append(replacement)

        return " ".join(perturbed_sentences)


# =============================================================================
# PLAYER DICTIONARY AND FACTORY
# =============================================================================

PLAYER_STRATEGIES = {
    "subword": SubwordPlayerStrategy,
    "word": WordPlayerStrategy,
    "named_entity": NamedEntityPlayerStrategy,
    "chunk": ChunkPlayerStrategy,
    "sentence": SentencePlayerStrategy,
}


def create_player_strategy(
    level: str,
    text: str,
    tokenizer: PreTrainedTokenizerBase,
) -> BasePlayerStrategy:
    """Create a player strategy from a string identifier."""
    if level not in PLAYER_STRATEGIES:
        msg = f"Unknown player level: {level}. Available levels: {list(PLAYER_STRATEGIES)}"
        raise ValueError(msg)

    strategy_cls = PLAYER_STRATEGIES[level]

    if level == "subword":
        return strategy_cls(
            text=text,
            tokenizer=tokenizer,
        )

    return strategy_cls(text=text)


# =============================================================================
# PERTURBATION STRATEGIES
# =============================================================================
class BasePerturbationStrategy:
    """Base class for all perturbation strategies."""


class BaseStringPerturbationStrategy(BasePerturbationStrategy, ABC):
    """Abstract perturbation strategy."""

    @abstractmethod
    def perturb(
        self,
        player: str,
        *,
        context: dict | None = None,
    ) -> str:
        """Perturb a player representation."""


class BaseAttentionMaskPerturbationStrategy(BasePerturbationStrategy, ABC):
    """Tensor-level perturbation strategy using attention masks.

    These strategies do not return perturbed text. Instead, they evaluate
    coalitions by modifying the attention mask.
    """

    @abstractmethod
    def evaluate(
        self,
        players: list[str],
        coalitions: np.ndarray,
    ) -> np.ndarray:
        """Evaluate coalition values."""


# =============================================================================
# TO DO:
# Current implementation:
# - [MASK] replacement
# - [PAD] replacement
# - removal perturbation
# - neutral perturbation
# - WordNet Neutral Perturbation
# - MLM infilling
# - Attention perturbation

# Future:
# - Attention Masking
# =============================================================================

# =============================================================================
# [MASK] replacement
# =============================================================================


class MaskTokenPerturbation(BaseStringPerturbationStrategy):
    """Replace missing words with [MASK]."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
    ) -> None:
        """Replace missing words with [MASK]."""
        self.mask_token = tokenizer.mask_token

        if self.mask_token is None:
            msg = (
                "Tokenizer does not define a mask token. "
                "MaskTokenPerturbation requires a masked language model tokenizer."
            )
            raise ValueError(msg)

    def perturb(
        self,
        _player: str,
        *,
        context: dict | None = None,  # noqa: ARG002
    ) -> str:
        """Replace missing words with [MASK]."""
        return self.mask_token


# =============================================================================
# [PAD] replacement
# =============================================================================


class PadTokenPerturbation(BaseStringPerturbationStrategy):
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
        _player: str,
        *,
        context: dict | None = None,  # noqa: ARG002
    ) -> str:
        """Return the PAD token."""
        return self.pad_token


# =============================================================================
# REMOVAL PERTURBATION
# =============================================================================


class RemovalPerturbation(BaseStringPerturbationStrategy):
    """Remove a player by replacing it with an empty string."""

    def perturb(
        self,
        _player: str,
        *,
        context: dict | None = None,  # noqa: ARG002
    ) -> str:
        """Return empty string."""
        return ""


# =============================================================================
# NEUTRAL PERTURBATION
# =============================================================================


class NeutralPerturbation(BaseStringPerturbationStrategy):
    """Replace missing players with neutral placeholder text."""

    def __init__(
        self,
        neutral_text: str = "something",
    ) -> None:
        """Replace missing players with neutral placeholder text."""
        self.neutral_text = neutral_text

    def perturb(
        self,
        _player: str,
        *,
        context: dict | None = None,  # noqa: ARG002
    ) -> str:
        """Return neutral replacement text."""
        return self.neutral_text


# =============================================================================
# WORDNET NEUTRAL PERTURBATION
# =============================================================================


def _penn_to_wn(tag: str) -> str | None:
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
    """Generate a semantic neutral replacement using WordNet hypernyms."""
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


class WordNetNeutralPerturbation(BaseStringPerturbationStrategy):
    """WordNet-based semantic neutral replacement."""

    def perturb(
        self,
        _player: str,
        *,
        context: dict | None = None,  # noqa: ARG002
    ) -> str:
        """Replace a player with a semantic hypernym."""
        tag = nltk.pos_tag([_player])[0][1]

        return _get_neutral_replacement(_player, tag)


# =============================================================================
# MLM Infilling Perturbation
# support:
# - word-level players
# - named-entity-level players
# - chunk-level players
# =============================================================================


class MLMInfillingPerturbation(BaseStringPerturbationStrategy):
    """Fill missing players using a masked language model."""

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        device: str = "cpu",
    ) -> None:
        """Initialize MLM-based infilling strategy."""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForMaskedLM.from_pretrained(model_name)

        self.device = device
        self.model.to(device)
        self.model.eval()

        self.mask_token = self.tokenizer.mask_token

        if self.mask_token is None:
            msg = f"{model_name} does not define a mask token."
            raise ValueError(msg)

        self._cache: dict[tuple, dict[int, str]] = {}

    def _build_cache_key(
        self,
        players: list[str],
        coalition: np.ndarray,
    ) -> tuple:
        return (tuple(players), tuple(coalition.tolist()))

    def _predict_masks(
        self,
        players: list[str],
        coalition: np.ndarray,
    ) -> dict[int, str]:
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
            top_k = 10

            candidate_ids = logits[0, token_pos].topk(top_k).indices.tolist()
            replacement = "something"

            for candidate_id in candidate_ids:
                token = self.tokenizer.convert_ids_to_tokens(candidate_id)

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
        """Generate an MLM-based replacement for a missing player."""
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
# =============================================================================
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


def build_tokenized_players(
    players: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> tuple[dict[str, torch.Tensor], list[tuple[int, int]]]:
    """Tokenize players, suffix, and target text into one sequence."""
    all_token_ids: list[int] = []
    player_spans: list[tuple[int, int]] = []

    for player_idx, player in enumerate(players):

        token_ids = tokenizer.encode(
            player,
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


def attention_mask_value_function(
    tokenizer: PreTrainedTokenizerBase,
    players: list[str],
    coalitions: np.ndarray,
) -> np.ndarray:
    """Evaluate coalition values using attention masking.

    Args:
        tokenizer: HuggingFace tokenizer.
        players: Text players to explain.
        coalitions: Coalition matrix of shape ``(n_coalitions, n_players)``.

    Returns:
        input_ids: Encoded Text
        attention_mask: attention_mask for llm
    """
    coalitions = np.asarray(coalitions, dtype=bool)

    if coalitions.ndim == 1:
        coalitions = coalitions.reshape(1, -1)

    encoded, player_spans = build_tokenized_players(
        players=players,
        tokenizer=tokenizer,
    )

    if coalitions.shape[1] != len(player_spans):
        msg = f"Expected coalition width {len(player_spans)}, got {coalitions.shape[1]}."
        raise ValueError(msg)

    masked_inputs: list[dict[str, torch.Tensor]] = []

    for coalition in coalitions:
        attention_mask = build_attention_mask_for_coalition(
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
class AttentionMaskPerturbation(BaseAttentionMaskPerturbationStrategy):
    """Evaluate missing players by masking their attention.

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
    def evaluate(
        self,
        players: list[str],
        coalitions: np.ndarray,
    ) -> np.ndarray:
        """Evaluate coalition values using attention masking."""
        return attention_mask_value_function(
            tokenizer=self.tokenizer,
            players=players,
            coalitions=coalitions,
        )


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
) -> BasePerturbationStrategy:
    """Create a perturbation strategy from a string identifier."""
    if strategy not in PERTURBATION_STRATEGIES:
        msg = (
            f"Unknown perturbation strategy: {strategy}. "
            f"Available strategies: "
            f"{list(PERTURBATION_STRATEGIES)}"
        )
        raise ValueError(msg)

    strategy_cls = PERTURBATION_STRATEGIES[strategy]

    if strategy in {"mask", "pad", "attention_mask"}:
        return strategy_cls(tokenizer)

    if strategy == "mlm_infilling":
        return strategy_cls()

    return strategy_cls()


# =============================================================================
# TARGET CALLABLES
# =============================================================================


class BaseTargetCallable(ABC):
    """Abstract interface for model-specific scoring."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        device: str,
    ) -> None:
        """Abstract interface for model-specific scoring."""
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    @abstractmethod
    def predict(
        self,
        texts: list[str],
    ) -> np.ndarray:
        """Return scalar scores."""


# =============================================================================
# Encoder only support
# =============================================================================


class EncoderClassifierCallable(BaseTargetCallable):
    """Encoder classifier scoring."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        device: str,
        class_idx: int = 1,
        output_type: str = "logit",
    ) -> None:
        """Encoder classifier scoring."""
        super().__init__(model, tokenizer, device)

        self.class_idx = class_idx
        self.output_type = output_type

        if output_type not in {"logit", "probability"}:
            msg = "output_type must be 'logit' or 'probability'"
            raise ValueError(msg)

    def predict(
        self,
        texts: list[str],
    ) -> np.ndarray:
        """Run encoder classifier inference."""
        encoded = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

        encoded = {key: value.to(self.device) for key, value in encoded.items()}

        with torch.no_grad():
            outputs = self.model(**encoded)
            logits = outputs.logits

        if self.output_type == "logit":
            scores = logits[:, self.class_idx]

        else:
            probs = torch.softmax(logits, dim=-1)
            scores = probs[:, self.class_idx]

        return scores.detach().cpu().numpy()


# =============================================================================
# Causal LM support
# =============================================================================


class CausalLMCallable(BaseTargetCallable):
    """Causal language model scoring."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        device: str,
        target_label: str = "good",
        prompt_template: str = ("Review: {text}\n\nSentiment:"),
    ) -> None:
        """Causal language model scoring."""
        super().__init__(model, tokenizer, device)
        self.prompt_template = prompt_template
        self.target_token_ids = tokenizer.encode(target_label, add_special_tokens=False)

        if len(self.target_token_ids) == 0:
            msg = f"Target label '{target_label}' produced no tokens."
            raise ValueError(msg)

        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                msg = "Tokenizer must define either a pad token or eos token."
                raise ValueError(msg)

            tokenizer.pad_token = tokenizer.eos_token

        tokenizer.padding_side = "left"

    def _build_prompt(
        self,
        text: str,
    ) -> str:
        """Construct a prompt for causal LM scoring."""
        return self.prompt_template.format(text=text)

    def _score_target_sequence(
        self,
        prompt: str,
    ) -> float:
        """Compute log-probability of target sequence."""
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        target_ids = self.target_token_ids
        total_log_prob = 0.0

        for i in range(len(target_ids)):
            prefix_ids = target_ids[:i]
            input_ids = prompt_ids + prefix_ids
            encoded = {"input_ids": torch.tensor([input_ids], device=self.device)}

            with torch.no_grad():
                outputs = self.model(**encoded)

            logits = outputs.logits
            next_token_logits = logits[:, -1, :]
            log_probs = torch.log_softmax(next_token_logits, dim=-1)

            total_log_prob += log_probs[0, target_ids[i]].item()

        return total_log_prob

    def predict(
        self,
        texts: list[str],
    ) -> np.ndarray:
        """Score texts using target-sequence log probabilities."""
        scores = []
        for text in texts:
            prompt = self._build_prompt(text)
            score = self._score_target_sequence(prompt)
            scores.append(score)

        return np.asarray(scores, dtype=np.float32)


# =============================================================================
# TO DO:
# seq2seq support
# =============================================================================


class Seq2SeqCallable(BaseTargetCallable):
    """Placeholder for future seq2seq support."""

    def predict(
        self,
        texts: list[str],
    ) -> np.ndarray:
        """Placeholder for future seq2seq support."""
        msg = "Seq2SeqCallable is not implemented yet."
        raise NotImplementedError(msg)


# =============================================================================
# TEXT IMPUTER
# =============================================================================


class TextImputer:
    """Generalized text imputer."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        text: str,
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
        # architecture selection
        # ---------------------------------------------------------------------
        player_level: str = "word",
        perturbation_type: str = "mask",
        player_strategy: BasePlayerStrategy | None = None,
        perturbation_strategy: BasePerturbationStrategy | None = None,
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

        elif model_type == "causal_lm":
            self.target_callable = CausalLMCallable(
                model=model,
                tokenizer=tokenizer,
                device=device,
                target_label=target_label,
                prompt_template=prompt_template,
            )

        elif model_type == "seq2seq":
            self.target_callable = Seq2SeqCallable(
                model=model,
                tokenizer=tokenizer,
                device=device,
            )

        else:
            msg = "model_type must be one of:\n- 'encoder_classifier'\n- 'causal_lm'\n- 'seq2seq'"
            raise ValueError(msg)

    @property
    def n_players(self) -> int:
        """Return number of players."""
        return self.player_strategy.n_players

    def coalition_to_text(
        self,
        coalition: np.ndarray,
    ) -> str:
        """Convert coalition into perturbed text."""
        if not isinstance(self.perturbation_strategy, BaseStringPerturbationStrategy):
            msg = (
                f"{self.perturbation_type!r} does not produce perturbed text. "
                "Use value_function() to evaluate coalitions."
            )
            raise TypeError(msg)

        return self.player_strategy.coalition_to_text(
            coalition,
            self.perturbation_strategy,
        )

    def _coalitions_to_texts(
        self,
        coalitions: np.ndarray,
    ) -> list[str]:
        """Convert coalition matrix into perturbed texts."""
        if not isinstance(self.perturbation_strategy, BaseStringPerturbationStrategy):
            msg = (
                f"{self.perturbation_type!r} does not produce perturbed text. "
                "Use value_function() to evaluate coalitions."
            )
            raise TypeError(msg)

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

    def value_function(
        self,
        coalitions: np.ndarray,
    ) -> np.ndarray:
        """Evaluate coalition values."""
        coalitions = np.asarray(coalitions)

        if coalitions.ndim == 1:
            coalitions = coalitions.reshape(1, -1)

        if coalitions.shape[1] != self.n_players:
            msg = f"Expected coalition width {self.n_players}, got {coalitions.shape[1]}"
            raise ValueError(msg)

        if isinstance(
            self.perturbation_strategy,
            BaseAttentionMaskPerturbationStrategy,
        ):
            players = self.player_strategy.get_players()

            return self.perturbation_strategy.evaluate(
                players=players,
                coalitions=coalitions,
            )

        if isinstance(
            self.perturbation_strategy,
            BaseStringPerturbationStrategy,
        ):
            texts = self._coalitions_to_texts(coalitions)
            return self._batched_predict(texts)

        msg = f"Unsupported perturbation strategy type: {type(self.perturbation_strategy).__name__}"
        raise TypeError(msg)

    def __call__(
        self,
        coalitions: np.ndarray,
    ) -> np.ndarray:
        """Make TextImputer compatible with shapiq approximators."""
        return self.value_function(coalitions)

    def full_prediction(self) -> float:
        """Score of original unperturbed text."""
        if isinstance(
            self.perturbation_strategy,
            BaseAttentionMaskPerturbationStrategy,
        ):
            full_coalition = np.ones((1, self.n_players), dtype=bool)
            score = self.value_function(full_coalition)[0]
            return float(score)

        score = self._predict_batch([self.text])[0]
        return float(score)
