"""Generalized text imputer for coalition-based text explanations."""

from __future__ import annotations  # noqa: I001

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, cast
import numpy as np
import torch
import nltk
from nltk.corpus import wordnet as wn
from nltk.tree import Tree

from transformers import AutoModelForMaskedLM, AutoTokenizer

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
# PLAYER STRATEGIES
# =============================================================================


class BasePlayerStrategy(ABC):
    """Abstract interface for converting text into Shapley players.

    A player strategy defines the feature granularity used for attribution.
    For example, a text can be represented by subwords, words, named entities, syntactic chunks, or sentences.

    Implementations must provide:
    - ``get_players`` to expose the extracted text units;
    - ``n_players`` to report the number of units;
    - ``coalition_to_text`` to reconstruct a perturbed text for a coalition.

    A coalition uses ``1`` for a kept player and ``0`` for a missing player.
    Missing players are replaced by the supplied perturbation strategy.
    """

    @abstractmethod
    def get_players(self) -> list[str]:
        """Return player list."""

    @abstractmethod
    def coalition_to_text(
        self,
        coalition: np.ndarray,
        perturbation_strategy: BasePerturbationStrategy,
    ) -> str:
        """Reconstruct text for a coalition using a perturbation strategy.

        Parameters:
        coalition:
            One-dimensional binary vector. A value of ``1`` keeps the corresponding player; a value of ``0`` replaces it using ``perturbation_strategy``.
        perturbation_strategy:
            Strategy that determines how missing players are represented.

        Returns:
        str: The perturbed text corresponding to the coalition.
        """

    @property
    @abstractmethod
    def n_players(self) -> int:
        """Return number of players."""


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
        _require_nltk_resource("tokenizers/punkt_tab", "punkt_tab")
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

        _require_nltk_resource("tokenizers/punkt_tab", "punkt_tab")
        _require_nltk_resource(
            "taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"
        )
        _require_nltk_resource("chunkers/maxent_ne_chunker_tab", "maxent_ne_chunker_tab")
        _require_nltk_resource("corpora/words", "words")

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
        _require_nltk_resource("tokenizers/punkt_tab", "punkt_tab")
        _require_nltk_resource(
            "taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"
        )
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
        _require_nltk_resource("tokenizers/punkt_tab", "punkt_tab")
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
    if level == "subword":
        return SubwordPlayerStrategy(
            text=text,
            tokenizer=tokenizer,
        )

    if level == "word":
        return WordPlayerStrategy(text=text)

    if level == "named_entity":
        return NamedEntityPlayerStrategy(text=text)

    if level == "chunk":
        return ChunkPlayerStrategy(text=text)

    if level == "sentence":
        return SentencePlayerStrategy(text=text)

    msg = (
        f"Unknown player level: {level}. "
        "Available levels: "
        "['subword', 'word', 'named_entity', 'chunk', 'sentence']"
    )

    raise ValueError(msg)


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
# Future:
# - Attention Masking
# =============================================================================

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

    def predict_from_inputs(
        self,
        inputs: list[dict[str, torch.Tensor]],
    ) -> np.ndarray:
        """Return scalar scores from pre-tokenized model inputs."""
        msg = f"{self.__class__.__name__} does not support pre-tokenized inputs."
        raise NotImplementedError(msg)


# =============================================================================
# Encoder only support
# =============================================================================


class EncoderClassifierCallable(BaseTargetCallable):
    """Score text with an encoder-only sequence-classification model.

    The callable returns either the selected class logit or its softmax probability.
    It is suitable for models such as BERT, RoBERTa, or DistilBERT with a sequence-classification head.
    """

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
        """Return one score per text from the configured classifier output.

        Texts are tokenized as a padded batch and evaluated in inference mode.
        Depending on ``output_type``, the returned score is either the raw logit or
        the softmax probability for ``class_idx``.
        """
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

    def predict_from_inputs(
        self,
        inputs: list[dict[str, torch.Tensor]],
    ) -> np.ndarray:
        """Run encoder classifier inference from pre-tokenized inputs."""
        input_ids = torch.cat(
            [item["input_ids"] for item in inputs],
            dim=0,
        ).to(self.device)

        attention_mask = torch.cat(
            [item["attention_mask"] for item in inputs],
            dim=0,
        ).to(self.device)

        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        if "token_type_ids" in inputs[0]:
            model_inputs["token_type_ids"] = torch.cat(
                [item["token_type_ids"] for item in inputs],
                dim=0,
            ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**model_inputs)
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
    """Score text using the log-probability of a target causal-LM continuation.

    Each input text is inserted into ``prompt_template``. The score is the
    autoregressive log-probability of ``target_label`` after that prompt.
    Multi-token target labels are scored token by token, conditioned on the
    prompt and all preceding target tokens.

    This supports decoder-only models such as Gemma, Llama, GPT, and Qwen,
    provided their tokenizer defines either a padding token or an EOS token.
    """

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
        """Compute the autoregressive log-probability of the target label.

        For each target token, the model receives the prompt followed by preceding target tokens.
        The final-position distribution is then used to score the next target token.
        """
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

    def predict_from_inputs(
        self,
        inputs: list[dict[str, torch.Tensor]],
    ) -> np.ndarray:
        """Score target sequence from pre-tokenized causal-LM prompt inputs."""
        scores = []

        for item in inputs:
            prompt_input_ids = item["input_ids"].to(self.device)
            prompt_attention_mask = item["attention_mask"].to(self.device)

            total_log_prob = 0.0

            for i, target_token_id in enumerate(self.target_token_ids):
                prefix_ids = self.target_token_ids[:i]

                if len(prefix_ids) > 0:
                    prefix_tensor = torch.tensor(
                        [prefix_ids],
                        dtype=torch.long,
                        device=self.device,
                    )

                    input_ids = torch.cat(
                        [prompt_input_ids, prefix_tensor],
                        dim=1,
                    )

                    prefix_attention_mask = torch.ones_like(prefix_tensor)

                    attention_mask = torch.cat(
                        [prompt_attention_mask, prefix_attention_mask],
                        dim=1,
                    )
                else:
                    input_ids = prompt_input_ids
                    attention_mask = prompt_attention_mask

                with torch.no_grad():
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )

                logits = outputs.logits
                next_token_logits = logits[:, -1, :]
                log_probs = torch.log_softmax(next_token_logits, dim=-1)

                total_log_prob += log_probs[0, target_token_id].item()

            scores.append(total_log_prob)

        return np.asarray(scores, dtype=np.float32)


# =============================================================================
# seq2seq support
# =============================================================================


class Seq2SeqCallable(BaseTargetCallable):
    """Score a fixed target sequence with an encoder-decoder model.

    For each input text, this callable computes the conditional log-probability
    of generating ``target_label`` using teacher forcing:

        log P(target_label | input_text)

    A multi-token target is scored token by token. By default, the final score
    is the mean token log-probability, so targets of different lengths are more
    comparable.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        device: str,
        target_label: str = "positive",
        prompt_template: str = "{text}",
        *,
        normalize: bool = True,
    ) -> None:
        """Initialize seq2seq target-sequence scoring."""
        super().__init__(model, tokenizer, device)

        if not getattr(model.config, "is_encoder_decoder", False):
            msg = (
                "Seq2SeqCallable requires an encoder-decoder model with "
                "model.config.is_encoder_decoder=True."
            )
            raise ValueError(msg)

        self.target_label = target_label
        self.prompt_template = prompt_template
        self.normalize = normalize

        self.target_token_ids: list[int] = tokenizer.encode(
            target_label,
            add_special_tokens=False,
        )
        if not self.target_token_ids:
            msg_0 = f"Target label {target_label!r} produced no tokens after encoding."
            raise ValueError(msg_0)

        decoder_start_token_id = model.config.decoder_start_token_id
        if decoder_start_token_id is None:
            decoder_start_token_id = tokenizer.pad_token_id

        if decoder_start_token_id is None:
            msg_1 = (
                "Cannot determine decoder_start_token_id: neither "
                "model.config.decoder_start_token_id nor tokenizer.pad_token_id "
                "is available."
            )
            raise ValueError(msg_1)

        self.decoder_start_token_id = decoder_start_token_id

    def _build_prompt(self, text: str) -> str:
        """Wrap the original text into a prompt template."""
        return self.prompt_template.format(text=text)

    def _encode_inputs(self, texts: list[str]) -> dict[str, torch.Tensor]:
        """Encode a list of texts into encoder input tensors."""
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        return {key: value.to(self.device) for key, value in encoded.items()}

    def _compute_log_prob_for_target(
        self,
        encoder_outputs: BaseModelOutput,
        attention_mask: torch.Tensor,
        batch_size: int,
    ) -> np.ndarray:
        """Compute the log-probability of the decoder generating the target token sequence."""
        total_log_probs = torch.zeros(batch_size, device=self.device)

        decoder_input_ids = torch.full(
            (batch_size, 1),
            self.decoder_start_token_id,
            dtype=torch.long,
            device=self.device,
        )

        for target_token_id in self.target_token_ids:
            with torch.no_grad():
                outputs = self.model(
                    encoder_outputs=encoder_outputs,
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids,
                )

            log_probs = torch.log_softmax(outputs.logits[:, -1, :], dim=-1)
            total_log_probs += log_probs[:, target_token_id]

            next_token = torch.full(
                (batch_size, 1),
                target_token_id,
                dtype=torch.long,
                device=self.device,
            )
            decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=1)

        if self.normalize:
            total_log_probs /= len(self.target_token_ids)

        return total_log_probs.cpu().numpy()

    def predict(self, texts: list[str]) -> np.ndarray:
        """Compute log-probability scores of the Seq2Seq target sequence for a batch of texts."""
        prompts = [self._build_prompt(text) for text in texts]
        encoder_inputs = self._encode_inputs(prompts)

        encoder = self.model.get_encoder()
        with torch.no_grad():
            encoder_outputs = encoder(
                input_ids=encoder_inputs["input_ids"],
                attention_mask=encoder_inputs["attention_mask"],
                return_dict=True,
            )

        return self._compute_log_prob_for_target(
            encoder_outputs=encoder_outputs,
            attention_mask=encoder_inputs["attention_mask"],
            batch_size=len(prompts),
        )

    def predict_from_inputs(
        self,
        inputs: list[dict[str, torch.Tensor]],
    ) -> np.ndarray:
        """Score target sequence from pre-tokenized encoder inputs."""
        input_ids = torch.cat(
            [item["input_ids"] for item in inputs],
            dim=0,
        ).to(self.device)

        attention_mask = torch.cat(
            [item["attention_mask"] for item in inputs],
            dim=0,
        ).to(self.device)

        encoder = self.model.get_encoder()

        with torch.no_grad():
            encoder_outputs = encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )

        return self._compute_log_prob_for_target(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            batch_size=input_ids.shape[0],
        )


# =============================================================================
# TEXT IMPUTER
# =============================================================================


class TextImputer:
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

    @property
    def n_players(self) -> int:
        """Return number of players."""
        return self.player_strategy.n_players

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

        if coalitions.shape[1] != self.n_players:
            msg = f"Expected coalition width {self.n_players}, got {coalitions.shape[1]}"
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

    def __call__(
        self,
        coalitions: np.ndarray,
    ) -> np.ndarray:
        """Make TextImputer compatible with shapiq approximators."""
        return self.value_function(coalitions)

    def full_prediction(self) -> float:
        """Score of original unperturbed text."""
        full_coalition = np.ones((1, self.n_players), dtype=bool)
        score = self.value_function(full_coalition)[0]
        return float(score)
