"""Generalized text imputer for coalition-based text explanations."""

from __future__ import annotations  # noqa: I001

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
import numpy as np
import torch
import nltk
from nltk.corpus import wordnet as wn
from nltk.tree import Tree

nltk.download("punkt")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger")
nltk.download("averaged_perceptron_tagger_eng")
nltk.download("maxent_ne_chunker")
nltk.download("words")

if TYPE_CHECKING:
    from transformers import (
        PreTrainedModel,
        PreTrainedTokenizerBase,
    )


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

# Yinan Feng
class SubwordPlayerStrategy(BasePlayerStrategy):
    """Tokenizer/subword-level player strategy.

    Uses the provided HuggingFace tokenizer to define players as
    tokenizer tokens (WordPiece/BPE/SentencePiece, etc.).
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
            msg = (
                f"Coalition length {len(coalition)} "
                f"does not match n_players={self.n_players}"
            )
            raise ValueError(msg)

        output_tokens: list[str] = []

        for keep, token in zip(
            coalition,
            self.subwords,
            strict=False,
        ):
            if keep:
                output_tokens.append(token)

            else:
                replacement = perturbation_strategy.perturb(token)

                if replacement != "":
                    output_tokens.append(replacement)

        return self.tokenizer.convert_tokens_to_string(
            output_tokens,
        )

# =============================================================================

# WORD-LEVEL PLAYER STRATEGY

# =============================================================================
# week 1

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
            msg = (
                f"Coalition length {len(coalition)} "
                f"does not match n_players={self.n_players}"
            )
            raise ValueError(msg)

        output_words = []

        for keep, word in zip(
            coalition,
            self.words,
            strict=False,
        ):
            if keep:
                output_words.append(word)
            else:
                replacement = perturbation_strategy.perturb(word)

                if replacement != "":
                    output_words.append(replacement)

        return " ".join(output_words)

# =============================================================================
# NAMED-ENTITY PLAYER STRATEGY
# =============================================================================

# yinan feng

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

            # Named entity
            if isinstance(node, Tree):
                entity = " ".join(
                    word
                    for word, _tag in node.leaves()
                )

                self.players.append(entity)

            # Non-entity token
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
            msg = (
                f"Coalition length {len(coalition)} "
                f"does not match n_players={self.n_players}"
            )
            raise ValueError(msg)

        output_players: list[str] = []

        for keep, player in zip(
            coalition,
            self.players,
            strict=False,
        ):
            if keep:
                output_players.append(player)

            else:
                replacement = perturbation_strategy.perturb(
                    player,
                )

                if replacement != "":
                    output_players.append(
                        replacement,
                    )

        return " ".join(output_players)

# =============================================================================
# CHUNK-LEVEL PLAYER STRATEGY
# =============================================================================

# yinan feng

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

        # yinan
        grammar = r"""
            NP: {<DT>?<JJ.*>*<NN.*>+}
            VP: {<VB.*><RB.*>*}
        """

        chunker = nltk.RegexpParser(grammar)
        tree = chunker.parse(pos_tags)

        self.chunks: list[str] = []

        for node in tree:
            # phrase chunk
            if isinstance(node, Tree):
                phrase = " ".join(
                    word
                    for word, _tag in node.leaves()
                )
                self.chunks.append(phrase)

            # single token
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
            msg = (
                f"Coalition length {len(coalition)} "
                f"does not match n_players={self.n_players}"
            )
            raise ValueError(msg)

        output_chunks: list[str] = []

        for keep, chunk in zip(
            coalition,
            self.chunks,
            strict=False,
        ):
            if keep:
                output_chunks.append(chunk)

            else:
                replacement = perturbation_strategy.perturb(
                    chunk,
                )

                if replacement != "":
                    output_chunks.append(
                        replacement,
                    )

        return " ".join(output_chunks)


# =============================================================================

# SENTENCE-LEVEL PLAYER STRATEGY

# =============================================================================

# xupeng wang

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
            msg = (
                f"Coalition length {len(coalition)} "
                f"does not match n_players={self.n_players}"
            )
            raise ValueError(msg)

        perturbed_sentences = []

        for keep, sentence in zip(coalition,self.sentences, strict=False):
            if keep:
                perturbed_sentences.append(sentence)
            else:
                replacement = perturbation_strategy.perturb(
                    sentence,
                )
                if replacement != "":
                    perturbed_sentences.append(
                        replacement,
                    )

        return " ".join(perturbed_sentences)

# =============================================================================
# PLAYER STRATEGY
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
    """Create a player strategy from a string identifier.

    Args:
        level: Player granularity level.
        text: Input text to explain.
        tokenizer: Tokenizer required for subword-level players.
    """
    if level not in PLAYER_STRATEGIES:
        msg = (
            f"Unknown player level: {level}. "
            f"Available levels: {list(PLAYER_STRATEGIES)}"
        )
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


class BasePerturbationStrategy(ABC):
    """Abstract perturbation strategy."""

    @abstractmethod
    def perturb(
        self,
        token: str,
    ) -> str:
        """Replace or modify missing feature."""


# =============================================================================
# TO DO:
# Current implementation:
# - [MASK] replacement
# - removal perturbation
# - neutral perturbation

# Future:
# - [PAD] replacement
# - MLM infilling
# - semantic replacement
# =============================================================================



# =============================================================================

# MASK TOKEN PERTURBATION

# =============================================================================


class MaskTokenPerturbation(BasePerturbationStrategy):
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
                "MaskTokenPerturbation requires "
                "a masked language model tokenizer."
            )
            raise ValueError(msg)

    def perturb(
        self,
        _token: str,
    ) -> str:
        """Replace missing words with [MASK]."""
        return self.mask_token

# =============================================================================

# REMOVAL PERTURBATION

# =============================================================================

# xupeng wang

class RemovalPerturbation(

    BasePerturbationStrategy,

):
    """Remove missing players entirely."""

    def perturb(

        self,

        _player: str,

    ) -> str:
        """Return empty string."""
        return ""

# =============================================================================

# NEUTRAL PERTURBATION

# =============================================================================

# xupeng wang

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

        _player: str,

    ) -> str:
        """Return neutral replacement text."""
        return self.neutral_text


# =============================================================================

# WORDNET NEUTRAL PERTURBATION

# =============================================================================

# Xufan Dong
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

def get_neutral_replacement( word: str, pos_tag: str) -> str:
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

class WordNetNeutralPerturbation(BasePerturbationStrategy):
    """Semantic neutral replacement.

    Examples:
    dog -> animal
    car -> vehicle
    apple -> fruit
    """

    def perturb(self, player: str) -> str:
        """Replace a player with a semantic hypernym."""
        tag = nltk.pos_tag([player])[0][1]

        return get_neutral_replacement(player, tag)

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
# TO DO:
# Current implementation:
# - encoder classifiers
# - BERT
# - RoBERTa
# - DistilBERT
#
# Future:
# - additional encoder architectures
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
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        encoded = {
            key: value.to(self.device)
            for key, value in encoded.items()
        }

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
# TO DO:
# Future causal LM support.
#
# Planned models:
# - Gemma
# - Llama
# - GPT-style models
#
# Planned scoring:
# - next-token likelihood
# - continuation likelihood
# - log-prob scoring
# =============================================================================

# Xufan Dong
class CausalLMCallable(BaseTargetCallable):
    """Causal language model scoring."""
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        device: str,
        target_label: str = "good",

        prompt_template: str = (
            "Review: {text}\n\n"
            "Sentiment:"
            ),
    ) -> None:
        """Causal language model scoring."""
        super().__init__(model, tokenizer, device)

        self.prompt_template = prompt_template

        target_ids = tokenizer.encode(target_label, add_special_tokens=False)

        if len(target_ids) != 1:
            msg = (
                f"Target label '{target_label}' "
                "must be single-token."
            )
            raise ValueError(msg)

        self.target_token_id = target_ids[0]

        if tokenizer.pad_token is None:

            tokenizer.pad_token = tokenizer.eos_token

        tokenizer.padding_side = "left"

    def _build_prompt(
        self,
        text: str,
    ) -> str:
        """Construct a prompt for causal LM scoring."""
        return self.prompt_template.format(text=text)

    def predict(
        self,
        texts: list[str],
    ) -> np.ndarray:
        """Score texts using next-token log probabilities."""
        prompts = [
            self._build_prompt(text)
            for text in texts
        ]

        encoded = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        encoded = {
            key: value.to(self.device)
            for key, value in encoded.items()
            }

        with torch.no_grad():
            outputs = self.model(**encoded)
            logits = outputs.logits

        next_token_logits = logits[:, -1, :]

        log_probs = torch.log_softmax(
            next_token_logits,
            dim=-1,
        )

        scores = log_probs[
            :,
            self.target_token_id,
        ]

        return (
            scores
            .detach()
            .cpu()
            .float()
            .numpy()
        )
# =============================================================================
# TO DO:
# Future seq2seq support.
#
# Planned models:
# - T5
# - BART
#
# Planned scoring:
# - decoder likelihood
# - target sequence probability
# =============================================================================


class Seq2SeqCallable(BaseTargetCallable):
    """Placeholder for future seq2seq support."""

    def predict(
        self,
        texts: list[str],
    ) -> np.ndarray:
        """Placeholder for future seq2seq support."""
        msg = "Seq2SeqCallable is not implemented yet."
        raise NotImplementedError(
            msg
        )


# =============================================================================
# TEXT IMPUTER
# =============================================================================


class TextImputer:
    """Generalized text imputer.

    Current support:
    - word-level players
    - [MASK] perturbation
    - encoder classifiers
    - batched coalition evaluation

    Architecture goals:
    - pluggable player strategies
    - pluggable perturbation strategies
    - pluggable target callables
    """

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
        # Xufan Dong

        target_label: str = "good",

        prompt_template: str = (
            "Review: {text}\n\n"
            "Sentiment:"
        ),

        # ---------------------------------------------------------------------
        # architecture selection
        # ---------------------------------------------------------------------
        player_level: str = "word",
        player_strategy: BasePlayerStrategy | None = None,
        perturbation_strategy: BasePerturbationStrategy | None = None,

        # =============================================================================
        # TO DO:
        # Generalize target callable support.
        #
        # Future:
        # - "causal_lm"
        # - "seq2seq"
        # =============================================================================
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

        # Xufan Dong
        if not hasattr(self.model, "hf_device_map"):
            self.model = self.model.to(self.device)

        self.model.eval()

        # =============================================================================
        # PLAYER STRATEGY
        # =============================================================================

        if player_strategy is None:
            player_strategy = create_player_strategy(
                level=player_level,
                text=text,
                tokenizer=tokenizer,
            )

        self.player_level = player_level
        self.player_strategy = player_strategy

        # =============================================================================
        # PERTURBATION STRATEGY
        # =============================================================================

        if perturbation_strategy is None:

            # TO DO:
            # future configurable perturbation defaults
            perturbation_strategy = MaskTokenPerturbation(
                tokenizer,
            )

        self.perturbation_strategy = perturbation_strategy

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
            msg = (
                "model_type must be one of:\n"
                "- 'encoder_classifier'\n"
                "- 'causal_lm'\n"
                "- 'seq2seq'"
            )
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
        return self.player_strategy.coalition_to_text(
            coalition,
            self.perturbation_strategy,
        )

    def _coalitions_to_texts(
        self,
        coalitions: np.ndarray,
    ) -> list[str]:
        """Convert coalition matrix into perturbed texts."""
        return [
            self.coalition_to_text(coalition)
            for coalition in coalitions
        ]

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

        for start in range(
            0,
            len(texts),
            self.batch_size,
        ):

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
            msg = (
                f"Expected coalition width {self.n_players}, "
                f"got {coalitions.shape[1]}"
            )
            raise ValueError(msg)

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
        score = self._predict_batch([self.text])[0]

        return float(score)
