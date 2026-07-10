"""Player strategies used by the TextImputer."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

try:
    import nltk
    from nltk.tree import Tree
except ImportError as err:
    from ._error import _text_import_error

    raise _text_import_error from err

if TYPE_CHECKING:
    import numpy as np
    from transformers import PreTrainedTokenizerBase

    from .perturbations import BasePerturbationStrategy


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

    _passes_context = True

    @abstractmethod
    def get_players(self) -> list[str]:
        """Return player list."""

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
        if len(coalition) != self.n_players:
            msg = f"Coalition length {len(coalition)} does not match n_players={self.n_players}"
            raise ValueError(msg)

        players = self.get_players()
        output_parts: list[str] = []

        for idx, (keep, player) in enumerate(zip(coalition, players, strict=False)):
            if keep:
                output_parts.append(player)
            else:
                context = (
                    {
                        "players": players,
                        "coalition": coalition,
                        "mask_index": idx,
                    }
                    if self._passes_context
                    else None
                )

                replacement = perturbation_strategy.perturb(
                    player,
                    context=context,
                )

                if replacement != "":
                    output_parts.append(replacement)
        return self._join(output_parts)

    @abstractmethod
    def _join(
        self,
        parts: list[str],
    ) -> str:
        """Join perturbed players into the final text."""

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

    def _join(
        self,
        parts: list[str],
    ) -> str:
        """Join subword tokens into text."""
        return self.tokenizer.convert_tokens_to_string(parts)


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

    def _join(
        self,
        parts: list[str],
    ) -> str:
        """Join words into text."""
        return " ".join(parts)


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

    def _join(
        self,
        parts: list[str],
    ) -> str:
        """Join entity into text."""
        return " ".join(parts)


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

    def _join(
        self,
        parts: list[str],
    ) -> str:
        """Join chunks into text."""
        return " ".join(parts)


# =============================================================================
# SENTENCE-LEVEL PLAYER STRATEGY
# =============================================================================


class SentencePlayerStrategy(BasePlayerStrategy):
    """Sentence-level player strategy using NLTK sentence splitting."""

    # Sentence perturbations do not use context.
    _passes_context = False

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

    def _join(
        self,
        parts: list[str],
    ) -> str:
        """Join sentences into text."""
        return " ".join(parts)


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
