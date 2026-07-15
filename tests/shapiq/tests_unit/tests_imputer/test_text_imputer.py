"""Fast mock-based unit tests for shapiq TextImputer."""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest
from nltk.tree import Tree

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402

from shapiq.imputer.text_imputer import (  # noqa: E402
    CausalLMCallable,
    ChunkPlayerStrategy,
    EncoderClassifierCallable,
    MaskTokenPerturbation,
    MLMInfillingPerturbation,
    NamedEntityPlayerStrategy,
    NeutralPerturbation,
    PadTokenPerturbation,
    RemovalPerturbation,
    SentencePlayerStrategy,
    SubwordPlayerStrategy,
    TextImputer,
    WordNetNeutralPerturbation,
    WordPlayerStrategy,
    _get_neutral_replacement,
    _penn_to_wn,
    _require_nltk_resource,
    create_perturbation_strategy,
    create_player_strategy,
)

MODULE = "shapiq.imputer.text_imputer"
PLAYERS_MODULE = "shapiq.imputer.text.players"
PERTURBATIONS_MODULE = "shapiq.imputer.text.perturbations"
CALLABLES_MODULE = "shapiq.imputer.text.callables"


class DummyTokenizer:
    """Small tokenizer substitute used by fast unit tests."""

    mask_token = "[MASK]"
    mask_token_id = 99
    pad_token = "[PAD]"
    pad_token_id = 0
    eos_token = "</s>"
    eos_token_id = 2
    cls_token = "[CLS]"
    sep_token = "[SEP]"
    padding_side = "right"

    def __init__(self) -> None:
        self.encode_return_values: list[list[int]] = []

    def tokenize(self, text: str) -> list[str]:
        return ["un", "##happy"]

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        return " ".join(tokens).replace(" ##", "")

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
    ) -> list[int]:
        return self.encode_return_values.pop(0)

    def __call__(
        self,
        texts,
        *,
        padding: bool = False,
        truncation: bool = False,
        return_tensors: str = "pt",
    ) -> dict[str, torch.Tensor]:
        if isinstance(texts, str):
            texts = [texts]

        return {
            "input_ids": torch.tensor(
                [[10, 99, 11] for _ in texts],
                dtype=torch.long,
            ),
            "attention_mask": torch.ones(
                (len(texts), 3),
                dtype=torch.long,
            ),
        }

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        mapping = {
            0: "[PAD]",
            1: "[CLS]",
            2: "[SEP]",
            3: "[MASK]",
            4: "##suffix",
            5: "great",
            6: "thing",
        }
        return mapping[token_ids[0]]


@pytest.fixture
def tokenizer() -> DummyTokenizer:
    return DummyTokenizer()


@pytest.fixture
def model() -> MagicMock:
    model = MagicMock()
    model.to.return_value = model
    model.return_value = SimpleNamespace(
        logits=torch.tensor([[1.0, 2.0]]),
    )

    return model


@pytest.fixture
def no_nltk_resource_check():
    """Avoid touching local NLTK data in tests."""
    with (
        patch("shapiq.imputer.text.players._require_nltk_resource"),
        patch("shapiq.imputer.text.perturbations._require_nltk_resource"),
    ):
        yield


# ============================================================================
# NLTK helper
# ============================================================================


def test_require_nltk_resource_passes_when_resource_exists() -> None:
    with patch(f"{PLAYERS_MODULE}.nltk.data.find") as find:
        _require_nltk_resource("tokenizers/punkt_tab", "punkt_tab")

    find.assert_called_once_with("tokenizers/punkt_tab")


def test_require_nltk_resource_passes_when_zip_resource_exists() -> None:
    with patch(
        f"{PLAYERS_MODULE}.nltk.data.find",
        side_effect=[LookupError("not installed"), None],
    ) as find:
        _require_nltk_resource("tokenizers/punkt_tab", "punkt_tab")

    assert find.call_count == 2

    assert find.call_args_list == [
        call("tokenizers/punkt_tab"),
        call("tokenizers/punkt_tab.zip"),
    ]


def test_require_nltk_resource_has_helpful_error_when_missing() -> None:
    with (
        patch(
            f"{PLAYERS_MODULE}.nltk.data.find",
            side_effect=LookupError("not installed"),
        ),
        pytest.raises(LookupError, match=r"nltk\.download\('punkt_tab'\)"),
    ):
        _require_nltk_resource("tokenizers/punkt_tab", "punkt_tab")


# ============================================================================
# Player strategies
# ============================================================================


def test_subword_player_strategy(tokenizer: DummyTokenizer) -> None:
    strategy = SubwordPlayerStrategy("unhappy", tokenizer)

    assert strategy.get_players() == ["un", "##happy"]
    assert strategy.n_players == 2

    text = strategy.coalition_to_text(
        np.array([1, 0]),
        NeutralPerturbation("X"),
    )

    assert text == "un X"


def test_subword_player_strategy_rejects_wrong_coalition_length(
    tokenizer: DummyTokenizer,
) -> None:
    strategy = SubwordPlayerStrategy("unhappy", tokenizer)

    with pytest.raises(ValueError, match="does not match n_players=2"):
        strategy.coalition_to_text(
            np.array([1]),
            NeutralPerturbation(),
        )


def test_word_player_strategy_with_mocked_nltk(
    no_nltk_resource_check,
) -> None:
    with patch(
        f"{PLAYERS_MODULE}.nltk.word_tokenize",
        return_value=["I", "love", "cats"],
    ):
        strategy = WordPlayerStrategy("I love cats")

    assert strategy.get_players() == ["I", "love", "cats"]
    assert strategy.n_players == 3

    assert (
        strategy.coalition_to_text(
            np.array([1, 0, 1]),
            MaskTokenPerturbation(DummyTokenizer()),
        )
        == "I [MASK] cats"
    )

    assert (
        strategy.coalition_to_text(
            np.array([1, 0, 1]),
            RemovalPerturbation(),
        )
        == "I cats"
    )


def test_word_player_strategy_passes_context_to_perturbation(
    no_nltk_resource_check,
) -> None:
    with patch(
        f"{PLAYERS_MODULE}.nltk.word_tokenize",
        return_value=["I", "love", "cats"],
    ):
        strategy = WordPlayerStrategy("I love cats")

    perturbation = MagicMock()
    perturbation.perturb.return_value = "X"

    assert (
        strategy.coalition_to_text(
            np.array([1, 0, 1]),
            perturbation,
        )
        == "I X cats"
    )

    perturbation.perturb.assert_called_once()
    args, kwargs = perturbation.perturb.call_args
    assert args == ("love",)
    context = kwargs["context"]
    assert context["players"] == ["I", "love", "cats"]
    np.testing.assert_array_equal(
        context["coalition"],
        np.array([1, 0, 1]),
    )

    assert context["mask_index"] == 1


def test_named_entity_player_strategy_groups_entities(
    no_nltk_resource_check,
) -> None:
    ner_tree = [
        Tree("PERSON", [("John", "NNP"), ("Smith", "NNP")]),
        ("visited", "VBD"),
        Tree("GPE", [("Berlin", "NNP")]),
    ]

    with (
        patch(f"{PLAYERS_MODULE}.nltk.word_tokenize", return_value=["ignored"]),
        patch(f"{PLAYERS_MODULE}.nltk.pos_tag", return_value=[("ignored", "NN")]),
        patch(f"{PLAYERS_MODULE}.nltk.ne_chunk", return_value=ner_tree),
    ):
        strategy = NamedEntityPlayerStrategy("John Smith visited Berlin")

    assert strategy.get_players() == ["John Smith", "visited", "Berlin"]
    assert (
        strategy.coalition_to_text(
            np.array([1, 0, 1]),
            NeutralPerturbation("something"),
        )
        == "John Smith something Berlin"
    )


def test_chunk_player_strategy_groups_phrases(
    no_nltk_resource_check,
) -> None:
    parsed_tree = [
        Tree("NP", [("the", "DT"), ("movie", "NN")]),
        ("was", "VBD"),
        Tree("ADJP", [("very", "RB"), ("good", "JJ")]),
    ]

    parser = MagicMock()
    parser.parse.return_value = parsed_tree

    with (
        patch(f"{PLAYERS_MODULE}.nltk.word_tokenize", return_value=["ignored"]),
        patch(f"{PLAYERS_MODULE}.nltk.pos_tag", return_value=[("ignored", "NN")]),
        patch(f"{PLAYERS_MODULE}.nltk.RegexpParser", return_value=parser),
    ):
        strategy = ChunkPlayerStrategy("the movie was very good")

    assert strategy.get_players() == ["the movie", "was", "very good"]
    assert (
        strategy.coalition_to_text(
            np.array([0, 1, 1]),
            NeutralPerturbation("something"),
        )
        == "something was very good"
    )


def test_sentence_player_strategy_with_mocked_nltk(
    no_nltk_resource_check,
) -> None:
    with patch(
        f"{PLAYERS_MODULE}.nltk.sent_tokenize",
        return_value=["First.", "Second."],
    ):
        strategy = SentencePlayerStrategy("First. Second.")

    assert strategy.get_players() == ["First.", "Second."]
    assert (
        strategy.coalition_to_text(
            np.array([1, 0]),
            PadTokenPerturbation(DummyTokenizer()),
        )
        == "First. [PAD]"
    )


def test_player_factory_creates_correct_strategy(
    tokenizer: DummyTokenizer,
    no_nltk_resource_check,
) -> None:
    assert isinstance(
        create_player_strategy("subword", "unhappy", tokenizer),
        SubwordPlayerStrategy,
    )

    with patch(f"{PLAYERS_MODULE}.nltk.word_tokenize", return_value=["hello"]):
        assert isinstance(
            create_player_strategy("word", "hello", tokenizer),
            WordPlayerStrategy,
        )

    with (
        patch(
            f"{PLAYERS_MODULE}.NamedEntityPlayerStrategy",
            return_value=MagicMock(),
        ) as named_entity_strategy,
        patch(
            f"{PLAYERS_MODULE}.ChunkPlayerStrategy",
            return_value=MagicMock(),
        ) as chunk_strategy,
        patch(
            f"{PLAYERS_MODULE}.SentencePlayerStrategy",
            return_value=MagicMock(),
        ) as sentence_strategy,
    ):
        create_player_strategy("named_entity", "hello", tokenizer)
        create_player_strategy("chunk", "hello", tokenizer)
        create_player_strategy("sentence", "hello", tokenizer)

    named_entity_strategy.assert_called_once_with(text="hello")
    chunk_strategy.assert_called_once_with(text="hello")
    sentence_strategy.assert_called_once_with(text="hello")


def test_player_factory_rejects_unknown_level(tokenizer: DummyTokenizer) -> None:
    with pytest.raises(ValueError, match="Unknown player level"):
        create_player_strategy("not_real", "hello", tokenizer)


# ============================================================================
# Basic perturbations and WordNet perturbation
# ============================================================================


def test_mask_and_pad_perturbations(tokenizer: DummyTokenizer) -> None:
    assert MaskTokenPerturbation(tokenizer).perturb("word") == "[MASK]"
    assert PadTokenPerturbation(tokenizer).perturb("word") == "[PAD]"
    assert RemovalPerturbation().perturb("word") == ""
    assert NeutralPerturbation("neutral").perturb("word") == "neutral"


def test_mask_and_pad_require_special_tokens() -> None:
    tokenizer_without_mask = DummyTokenizer()
    tokenizer_without_mask.mask_token = None

    with pytest.raises(ValueError, match="does not define a mask token"):
        MaskTokenPerturbation(tokenizer_without_mask)

    tokenizer_without_pad = DummyTokenizer()
    tokenizer_without_pad.pad_token = None

    with pytest.raises(ValueError, match="does not define a pad token"):
        PadTokenPerturbation(tokenizer_without_pad)


@pytest.mark.parametrize(
    ("tag", "expected"),
    [
        ("NN", "n"),
        ("VBZ", "v"),
        ("JJ", "a"),
        ("RB", "r"),
        ("IN", None),
    ],
)
def test_penn_to_wordnet_mapping(tag: str, expected: str | None) -> None:
    fake_wn = SimpleNamespace(
        NOUN="n",
        VERB="v",
        ADJ="a",
        ADV="r",
    )

    with patch(f"{PERTURBATIONS_MODULE}.wn", fake_wn):
        assert _penn_to_wn(tag) == expected


def test_get_neutral_replacement_uses_hypernym() -> None:
    hypernym = MagicMock()
    hypernym.lemma_names.return_value = ["living_thing"]

    synset = MagicMock()
    synset.hypernyms.return_value = [hypernym]

    fake_wn = SimpleNamespace(
        NOUN="n",
        VERB="v",
        ADJ="a",
        ADV="r",
        synsets=MagicMock(return_value=[synset]),
    )
    with patch(f"{PERTURBATIONS_MODULE}.wn", fake_wn):
        assert _get_neutral_replacement("cat", "NN") == "living"


@pytest.mark.parametrize("tag", ["IN", "NN"])
def test_get_neutral_replacement_falls_back_to_something(tag: str) -> None:
    if tag == "IN":
        assert _get_neutral_replacement("of", tag) == "something"
    else:
        fake_wn = SimpleNamespace(
            NOUN="n",
            VERB="v",
            ADJ="a",
            ADV="r",
            synsets=MagicMock(return_value=[]),
        )
        with patch(f"{PERTURBATIONS_MODULE}.wn", fake_wn):
            assert _get_neutral_replacement("unknown", tag) == "something"


def test_get_neutral_replacement_falls_back_when_no_hypernym() -> None:
    synset = MagicMock()
    synset.hypernyms.return_value = []

    fake_wn = SimpleNamespace(
        NOUN="n",
        VERB="v",
        ADJ="a",
        ADV="r",
        synsets=MagicMock(return_value=[synset]),
    )

    with patch(f"{PERTURBATIONS_MODULE}.wn", fake_wn):
        assert _get_neutral_replacement("cat", "NN") == "something"


def test_wordnet_neutral_perturbation(
    no_nltk_resource_check,
) -> None:
    with (
        patch(f"{PERTURBATIONS_MODULE}.nltk.pos_tag", return_value=[("cat", "NN")]),
        patch(
            f"{PERTURBATIONS_MODULE}._get_neutral_replacement",
            return_value="animal",
        ),
    ):
        result = WordNetNeutralPerturbation().perturb("cat")

    assert result == "animal"


def test_perturbation_factory(tokenizer: DummyTokenizer) -> None:
    assert isinstance(
        create_perturbation_strategy("mask", tokenizer),
        MaskTokenPerturbation,
    )
    assert isinstance(
        create_perturbation_strategy("pad", tokenizer),
        PadTokenPerturbation,
    )
    assert isinstance(
        create_perturbation_strategy("removal", tokenizer),
        RemovalPerturbation,
    )
    assert isinstance(
        create_perturbation_strategy("neutral", tokenizer),
        NeutralPerturbation,
    )
    assert isinstance(
        create_perturbation_strategy("wordnet_neutral", tokenizer),
        WordNetNeutralPerturbation,
    )

    with pytest.raises(ValueError, match="Unknown perturbation strategy"):
        create_perturbation_strategy("not_real", tokenizer)


def test_perturbation_factory_creates_mlm_infilling(
    tokenizer: DummyTokenizer,
) -> None:
    with patch(
        f"{PERTURBATIONS_MODULE}.MLMInfillingPerturbation",
    ) as mlm_strategy:
        result = create_perturbation_strategy(
            "mlm_infilling",
            tokenizer,
            mlm_model_name="fake-mlm",
            mlm_num_samples=5,
            device="cpu",
        )

    mlm_strategy.assert_called_once_with(
        model_name="fake-mlm",
        device="cpu",
        num_samples=5,
    )
    assert result is mlm_strategy.return_value


# ============================================================================
# MLM infilling: all model calls are mocked
# ============================================================================


def test_mlm_infilling_initializes_model_and_tokenizer() -> None:
    tokenizer = MagicMock()
    tokenizer.mask_token = "[MASK]"

    model = MagicMock()
    model.to.return_value = model

    with (
        patch(
            f"{PERTURBATIONS_MODULE}.AutoTokenizer.from_pretrained",
            return_value=tokenizer,
        ) as tokenizer_loader,
        patch(
            f"{PERTURBATIONS_MODULE}.AutoModelForMaskedLM.from_pretrained",
            return_value=model,
        ) as model_loader,
    ):
        perturbation = MLMInfillingPerturbation(
            model_name="fake-mlm",
            device="cpu",
            num_samples=5,
        )

    tokenizer_loader.assert_called_once_with("fake-mlm")
    model_loader.assert_called_once_with("fake-mlm")
    model.to.assert_called_once_with("cpu")
    model.eval.assert_called_once()

    assert perturbation.tokenizer is tokenizer
    assert perturbation.model is model
    assert perturbation.model_name == "fake-mlm"
    assert perturbation.device == "cpu"
    assert perturbation.mask_token == "[MASK]"
    assert perturbation._cache == {}
    assert perturbation.num_samples == 5


def test_mlm_infilling_rejects_tokenizer_without_mask_token() -> None:
    tokenizer = MagicMock()
    tokenizer.mask_token = None

    model = MagicMock()
    model.to.return_value = model

    with (
        patch(
            f"{PERTURBATIONS_MODULE}.AutoTokenizer.from_pretrained",
            return_value=tokenizer,
        ),
        patch(
            f"{PERTURBATIONS_MODULE}.AutoModelForMaskedLM.from_pretrained",
            return_value=model,
        ),
        pytest.raises(ValueError, match="does not define a mask token"),
    ):
        MLMInfillingPerturbation(
            model_name="fake-mlm",
            device="cpu",
        )


def make_mlm_without_constructor(
    tokenizer: DummyTokenizer,
    model: MagicMock,
) -> MLMInfillingPerturbation:
    """Create an MLM perturbation without downloading a Hugging Face model."""
    perturbation = object.__new__(MLMInfillingPerturbation)
    perturbation.tokenizer = tokenizer
    perturbation.model = model
    perturbation.model_name = "fake-mlm"
    perturbation.device = "cpu"
    perturbation.mask_token = "[MASK]"
    perturbation._cache = {}
    perturbation.num_samples = 3
    return perturbation


def test_mlm_infilling_requires_context(
    tokenizer: DummyTokenizer,
    model: MagicMock,
) -> None:
    perturbation = make_mlm_without_constructor(tokenizer, model)

    with pytest.raises(ValueError, match="requires context"):
        perturbation.perturb("movie")


def test_mlm_infilling_caches_one_prediction_per_coalition(
    tokenizer: DummyTokenizer,
    model: MagicMock,
) -> None:
    perturbation = make_mlm_without_constructor(tokenizer, model)
    perturbation._predict_masks = MagicMock(
        return_value={1: "great"},
    )

    context = {
        "players": ["This", "movie", "works"],
        "coalition": np.array([1, 0, 1]),
        "mask_index": 1,
    }

    assert perturbation.perturb("movie", context=context) == "great"
    assert perturbation.perturb("movie", context=context) == "great"

    perturbation._predict_masks.assert_called_once()
    perturbation.clear_cache()
    assert perturbation._cache == {}


def test_mlm_infilling_returns_original_player_if_index_missing(
    tokenizer: DummyTokenizer,
    model: MagicMock,
) -> None:
    perturbation = make_mlm_without_constructor(tokenizer, model)
    perturbation._predict_masks = MagicMock(return_value={})

    assert (
        perturbation.perturb(
            "movie",
            context={
                "players": ["movie"],
                "coalition": np.array([0]),
                "mask_index": 0,
            },
        )
        == "movie"
    )


def test_mlm_predict_masks_filters_invalid_tokens(
    tokenizer: DummyTokenizer,
    model: MagicMock,
) -> None:
    perturbation = make_mlm_without_constructor(tokenizer, model)

    logits = torch.zeros((1, 3, 10))
    model.return_value = SimpleNamespace(logits=logits)

    # Invalid candidates: [PAD], [CLS], [SEP], [MASK], ##suffix.
    # First valid candidate is "great".
    sampled_ids = iter([0, 1, 2, 3, 4, 5])

    with patch(
        f"{PERTURBATIONS_MODULE}.torch.multinomial",
        side_effect=lambda *args, **kwargs: torch.tensor([next(sampled_ids)]),
    ):
        replacements = perturbation._predict_masks(
            players=["This", "movie", "works"],
            coalition=np.array([1, 0, 1]),
        )

    assert replacements == {1: "great"}
    model.assert_called_once()


def test_mlm_predict_masks_falls_back_after_failed_sampling(
    tokenizer: DummyTokenizer,
    model: MagicMock,
) -> None:
    perturbation = make_mlm_without_constructor(tokenizer, model)

    logits = torch.zeros((1, 3, 10))
    model.return_value = SimpleNamespace(logits=logits)

    with patch(
        f"{PERTURBATIONS_MODULE}.torch.multinomial",
        return_value=torch.tensor([0]),  # always [PAD]
    ):
        replacements = perturbation._predict_masks(
            players=["This", "movie", "works"],
            coalition=np.array([1, 0, 1]),
        )

    assert replacements == {1: "something"}


# ============================================================================
# Target callables
# ============================================================================


def test_encoder_callable_returns_logits(
    tokenizer: DummyTokenizer,
    model: MagicMock,
) -> None:
    model.return_value = SimpleNamespace(
        logits=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
    )

    callable_ = EncoderClassifierCallable(
        model=model,
        tokenizer=tokenizer,
        device="cpu",
        class_idx=1,
        output_type="logit",
    )

    np.testing.assert_allclose(
        callable_.predict(["a", "b"]),
        np.array([2.0, 4.0]),
    )


def test_encoder_callable_returns_probabilities(
    tokenizer: DummyTokenizer,
    model: MagicMock,
) -> None:
    model.return_value = SimpleNamespace(
        logits=torch.tensor([[0.0, 0.0]]),
    )

    callable_ = EncoderClassifierCallable(
        model=model,
        tokenizer=tokenizer,
        device="cpu",
        class_idx=1,
        output_type="probability",
    )

    np.testing.assert_allclose(
        callable_.predict(["a"]),
        np.array([0.5]),
    )


def test_encoder_callable_rejects_invalid_output_type(
    tokenizer: DummyTokenizer,
    model: MagicMock,
) -> None:
    with pytest.raises(ValueError, match="output_type"):
        EncoderClassifierCallable(
            model=model,
            tokenizer=tokenizer,
            device="cpu",
            output_type="not_real",
        )


def test_causal_callable_scores_multi_token_target(
    tokenizer: DummyTokenizer,
    model: MagicMock,
) -> None:
    # First encode call is target label; following calls are prompt encodings.
    tokenizer.encode_return_values = [
        [5, 6],  # target label
        [10, 11],  # prompt
    ]

    logits = torch.zeros((1, 2, 20))
    logits[0, -1, 5] = 3.0
    logits[0, -1, 6] = 4.0
    model.return_value = SimpleNamespace(logits=logits)

    callable_ = CausalLMCallable(
        model=model,
        tokenizer=tokenizer,
        device="cpu",
        target_label="very good",
    )

    scores = callable_.predict(["nice review"])

    assert scores.shape == (1,)
    assert model.call_count == 2
    assert callable_._build_prompt("nice") == ("Review: nice\n\nSentiment:")


def test_causal_callable_uses_eos_as_pad_when_pad_is_missing(
    model: MagicMock,
) -> None:
    tokenizer = DummyTokenizer()
    tokenizer.pad_token_id = None
    tokenizer.encode_return_values = [[5]]

    CausalLMCallable(
        model=model,
        tokenizer=tokenizer,
        device="cpu",
    )

    assert tokenizer.pad_token == "</s>"
    assert tokenizer.padding_side == "left"


def test_causal_callable_rejects_empty_target(
    tokenizer: DummyTokenizer,
    model: MagicMock,
) -> None:
    tokenizer.encode_return_values = [[]]

    with pytest.raises(ValueError, match="produced no tokens"):
        CausalLMCallable(
            model=model,
            tokenizer=tokenizer,
            device="cpu",
        )


# ============================================================================
# TextImputer orchestration
# ============================================================================


def make_player_strategy() -> MagicMock:
    strategy = MagicMock()
    strategy.n_players = 2
    strategy.coalition_to_text.side_effect = [
        "full-text",
        "empty-text",
        "text-1",
        "text-2",
        "text-3",
    ]
    return strategy


def test_text_imputer_creates_default_player_strategy(
    tokenizer: DummyTokenizer,
    model: MagicMock,
) -> None:
    player_strategy = make_player_strategy()

    with patch(
        "shapiq.imputer.text.imputer.create_player_strategy",
        return_value=player_strategy,
    ) as create_strategy:
        TextImputer(
            model=model,
            tokenizer=tokenizer,
            text="original",
            perturbation_strategy=NeutralPerturbation(),
        )

    create_strategy.assert_called_once_with(
        level="word",
        text="original",
        tokenizer=tokenizer,
    )


def test_text_imputer_creates_default_perturbation_strategy(
    tokenizer: DummyTokenizer,
    model: MagicMock,
) -> None:
    player_strategy = make_player_strategy()
    perturbation = NeutralPerturbation()

    with patch(
        "shapiq.imputer.text.imputer.create_perturbation_strategy",
        return_value=perturbation,
    ) as create_strategy:
        TextImputer(
            model=model,
            tokenizer=tokenizer,
            text="original",
            player_strategy=player_strategy,
            perturbation_type="neutral",
            mlm_model_name="test-mlm",
            mlm_num_samples=5,
            device="cpu",
        )

    create_strategy.assert_called_once_with(
        strategy="neutral",
        tokenizer=tokenizer,
        mlm_model_name="test-mlm",
        mlm_num_samples=5,
        device="cpu",
    )


def test_text_imputer_rejects_both_perturbation_strategies(
    tokenizer: DummyTokenizer,
    model: MagicMock,
) -> None:
    player_strategy = make_player_strategy()

    with pytest.raises(
        ValueError,
        match="Only one of perturbation_strategy and tensor_perturbation_strategy",
    ):
        TextImputer(
            model=model,
            tokenizer=tokenizer,
            text="original",
            player_strategy=player_strategy,
            perturbation_strategy=NeutralPerturbation(),
            tensor_perturbation_strategy=MagicMock(),
        )


def test_text_imputer_rejects_text_strategy_for_tensor_perturbation(
    tokenizer: DummyTokenizer,
    model: MagicMock,
) -> None:
    player_strategy = make_player_strategy()

    with pytest.raises(ValueError, match="is a tensor perturbation"):
        TextImputer(
            model=model,
            tokenizer=tokenizer,
            text="original",
            player_strategy=player_strategy,
            perturbation_type="attention_mask",
            perturbation_strategy=NeutralPerturbation(),
        )


def test_text_imputer_rejects_tensor_strategy_for_text_perturbation(
    tokenizer: DummyTokenizer,
    model: MagicMock,
) -> None:
    player_strategy = make_player_strategy()

    with pytest.raises(ValueError, match="is a text perturbation"):
        TextImputer(
            model=model,
            tokenizer=tokenizer,
            text="original",
            player_strategy=player_strategy,
            perturbation_type="mask",
            tensor_perturbation_strategy=MagicMock(),
        )


def test_text_imputer_rejects_coalition_to_text_in_tensor_mode(
    tokenizer: DummyTokenizer,
    model: MagicMock,
) -> None:
    player_strategy = make_player_strategy()
    tensor_perturbation_strategy = MagicMock()
    tensor_perturbation_strategy.evaluate.return_value = [{"input_ids": torch.tensor([[1, 2]])}]

    with patch.object(
        EncoderClassifierCallable,
        "predict_from_inputs",
        return_value=np.array([0.5]),
    ):
        imputer = TextImputer(
            model=model,
            tokenizer=tokenizer,
            text="original",
            player_strategy=player_strategy,
            perturbation_type="attention_mask",
            tensor_perturbation_strategy=tensor_perturbation_strategy,
        )

    with pytest.raises(
        RuntimeError,
        match=r"coalition_to_text\(\) can only be used with text perturbation strategies",
    ):
        imputer.coalition_to_text(np.array([1, 0]))


def test_text_imputer_rejects_coalitions_to_texts_in_tensor_mode(
    tokenizer: DummyTokenizer,
    model: MagicMock,
) -> None:
    player_strategy = make_player_strategy()
    tensor_perturbation_strategy = MagicMock()
    tensor_perturbation_strategy.evaluate.return_value = [{"input_ids": torch.tensor([[1, 2]])}]

    with patch.object(
        EncoderClassifierCallable,
        "predict_from_inputs",
        return_value=np.array([0.5]),
    ):
        imputer = TextImputer(
            model=model,
            tokenizer=tokenizer,
            text="original",
            player_strategy=player_strategy,
            perturbation_type="attention_mask",
            tensor_perturbation_strategy=tensor_perturbation_strategy,
        )

    with pytest.raises(
        RuntimeError,
        match=r"_coalitions_to_texts\(\) can only be used with text perturbation strategies",
    ):
        imputer._coalitions_to_texts(np.array([[1, 0]]))


def test_text_imputer_batches_and_returns_scores(
    tokenizer: DummyTokenizer,
    model: MagicMock,
) -> None:
    player_strategy = make_player_strategy()
    perturbation = NeutralPerturbation()

    imputer = TextImputer(
        model=model,
        tokenizer=tokenizer,
        text="original",
        batch_size=2,
        player_strategy=player_strategy,
        perturbation_strategy=perturbation,
    )

    imputer.target_callable = MagicMock()
    imputer.target_callable.predict.side_effect = [
        np.array([0.1, 0.2]),
        np.array([0.3]),
    ]

    scores = imputer.value_function(
        np.array([[1, 0], [0, 1], [0, 0]]),
    )

    np.testing.assert_allclose(scores, np.array([0.1, 0.2, imputer.empty_prediction]))
    assert imputer.target_callable.predict.call_args_list == [
        call(["text-1", "text-2"]),
        call(["text-3"]),
    ]


def test_text_imputer_accepts_one_dimensional_coalition(
    tokenizer: DummyTokenizer,
    model: MagicMock,
) -> None:
    player_strategy = make_player_strategy()

    imputer = TextImputer(
        model=model,
        tokenizer=tokenizer,
        text="original",
        player_strategy=player_strategy,
        perturbation_strategy=NeutralPerturbation(),
    )

    imputer.target_callable = MagicMock()
    imputer.target_callable.predict.return_value = np.array([0.7])

    np.testing.assert_allclose(
        imputer.value_function(np.array([1, 0])),
        np.array([0.7]),
    )


def test_text_imputer_rejects_wrong_coalition_width(
    tokenizer: DummyTokenizer,
    model: MagicMock,
) -> None:
    player_strategy = make_player_strategy()

    imputer = TextImputer(
        model=model,
        tokenizer=tokenizer,
        text="original",
        player_strategy=player_strategy,
        perturbation_strategy=NeutralPerturbation(),
    )

    with pytest.raises(ValueError, match="Expected coalition width 2"):
        imputer.value_function(np.array([[1, 0, 1]]))


def test_text_imputer_mlm_averages_multiple_samples(
    tokenizer: DummyTokenizer,
    model: MagicMock,
) -> None:
    player_strategy = MagicMock()
    player_strategy.n_players = 1
    player_strategy.coalition_to_text.side_effect = [
        # full prediction
        "full-1",
        "full-2",
        "full-3",
        # empty prediction
        "empty-1",
        "empty-2",
        "empty-3",
        # value_function
        "sample-1",
        "sample-2",
        "sample-3",
    ]

    mlm = make_mlm_without_constructor(tokenizer, model)
    mlm.num_samples = 3
    mlm.clear_cache = MagicMock()

    imputer = TextImputer(
        model=model,
        tokenizer=tokenizer,
        text="original",
        player_strategy=player_strategy,
        perturbation_strategy=mlm,
        player_level="word",
    )

    imputer.target_callable = MagicMock()
    imputer.target_callable.predict.side_effect = [
        np.array([1.0]),
        np.array([2.0]),
        np.array([3.0]),
    ]

    np.testing.assert_allclose(
        imputer.value_function(np.array([[0]])),
        np.array([2.0]),
    )

    assert mlm.clear_cache.call_count == mlm.num_samples * 3
    assert imputer._last_generated_texts == [
        "sample-1",
        "sample-2",
        "sample-3",
    ]


@pytest.mark.parametrize("player_level", ["subword", "sentence"])
def test_text_imputer_rejects_unsupported_mlm_player_levels(
    tokenizer: DummyTokenizer,
    model: MagicMock,
    player_level: str,
) -> None:
    mlm = make_mlm_without_constructor(tokenizer, model)
    player_strategy = MagicMock()
    player_strategy.n_players = 1

    with pytest.raises(ValueError, match="supports only word"):
        TextImputer(
            model=model,
            tokenizer=tokenizer,
            text="original",
            player_level=player_level,
            player_strategy=player_strategy,
            perturbation_strategy=mlm,
        )


def test_text_imputer_rejects_unknown_model_type(
    tokenizer: DummyTokenizer,
    model: MagicMock,
) -> None:
    player_strategy = MagicMock()
    player_strategy.n_players = 1

    with pytest.raises(ValueError, match="model_type must be one of"):
        TextImputer(
            model=model,
            tokenizer=tokenizer,
            text="original",
            player_strategy=player_strategy,
            perturbation_strategy=NeutralPerturbation(),
            model_type="not_real",
        )


def test_text_imputer_full_prediction_and_call(
    tokenizer: DummyTokenizer,
    model: MagicMock,
) -> None:
    player_strategy = MagicMock()
    player_strategy.n_players = 1
    player_strategy.coalition_to_text.return_value = "perturbed"

    imputer = TextImputer(
        model=model,
        tokenizer=tokenizer,
        text="original",
        player_strategy=player_strategy,
        perturbation_strategy=NeutralPerturbation(),
    )

    assert imputer.full_prediction == 2.0


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("RUN_SLOW_TESTS") != "1",
    reason="Set RUN_SLOW_TESTS=1 to run slow end-to-end tests.",
)
def test_text_imputer_end_to_end_with_tiny_checkpoint() -> None:
    """Run TextImputer end-to-end with a real tiny Hugging Face checkpoint."""
    model_name = "hf-internal-testing/tiny-random-bert"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    imputer = TextImputer(
        model=model,
        tokenizer=tokenizer,
        text="This movie is surprisingly good.",
        player_level="subword",
        perturbation_type="mask",
        model_type="encoder_classifier",
        class_idx=1,
        output_type="logit",
        device="cpu",
    )

    coalitions = np.stack(
        [
            imputer.empty_coalition,
            imputer.grand_coalition,
        ]
    )

    scores = imputer(coalitions)

    assert scores.shape == (2,)
    assert np.all(np.isfinite(scores))
    assert scores[0] == pytest.approx(0.0)
