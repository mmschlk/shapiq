# ============================================================================
# Pytest unit tests for Seq2SeqCallable
# Coverage: model type validation, single-token target, multi-token
#           teacher forcing, normalisation, prompt template, end-to-end
#           integration with TextImputer
# ============================================================================
from __future__ import annotations

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from shapiq.imputer.text_imputer import (  # noqa: E402
    NeutralPerturbation,
    Seq2SeqCallable,
    TextImputer,
)

MODULE = "shapiq.imputer.text_imputer"


# ============================================================================
# Mock fixtures — no real model downloads; all model calls go through MagicMock
# ============================================================================


def make_seq2seq_tokenizer() -> MagicMock:
    """Return a minimal seq2seq tokenizer substitute.

    encode() return values are controlled via the encode_queue list:
    each call to encode() pops from the front of the queue.
    """
    tok = MagicMock()
    tok.pad_token = "[PAD]"
    tok.pad_token_id = 0
    tok.eos_token = "</s>"
    tok.eos_token_id = 2

    tok.encode_queue = []
    tok.encode.side_effect = lambda text, **kwargs: tok.encode_queue.pop(0)

    tok.return_value = {
        "input_ids": torch.tensor([[10, 11, 12]]),
        "attention_mask": torch.ones((1, 3), dtype=torch.long),
    }
    return tok


def make_seq2seq_model(decoder_start_token_id: int = 0) -> MagicMock:
    """Return a minimal seq2seq model substitute.

    config.is_encoder_decoder is set to True to mimic T5 / BART.
    model.get_encoder() returns an encoder mock whose output is a
    SimpleNamespace with a last_hidden_state tensor.
    The return value of model(**kwargs) can be overridden per test.
    """
    model = MagicMock()
    model.to.return_value = model

    model.config.is_encoder_decoder = True
    model.config.decoder_start_token_id = decoder_start_token_id

    encoder_mock = MagicMock()
    encoder_mock.return_value = SimpleNamespace(
        last_hidden_state=torch.zeros((1, 3, 16)),
    )
    model.get_encoder.return_value = encoder_mock

    return model


@pytest.fixture
def seq2seq_tokenizer() -> MagicMock:
    return make_seq2seq_tokenizer()


@pytest.fixture
def seq2seq_model() -> MagicMock:
    return make_seq2seq_model()


# ============================================================================
# TEST 1 — Model type validation
# ============================================================================
# Seq2SeqCallable.__init__ reads model.config.is_encoder_decoder.
# If the flag is False or absent, a ValueError mentioning
# "is_encoder_decoder" must be raised.
# ============================================================================


class TestModelTypeValidation:
    def test_rejects_model_with_is_encoder_decoder_false(
        self,
        seq2seq_tokenizer: MagicMock,
    ) -> None:
        """ValueError must be raised when is_encoder_decoder=False."""
        seq2seq_tokenizer.encode_queue = [[1]]

        bad_model = make_seq2seq_model()
        bad_model.config.is_encoder_decoder = False

        with pytest.raises(ValueError, match="is_encoder_decoder"):
            Seq2SeqCallable(
                model=bad_model,
                tokenizer=seq2seq_tokenizer,
                device="cpu",
            )

    def test_rejects_model_without_is_encoder_decoder_attribute(
        self,
        seq2seq_tokenizer: MagicMock,
    ) -> None:
        """When the config attribute is absent, getattr defaults to False and
        the constructor must raise ValueError."""
        seq2seq_tokenizer.encode_queue = [[1]]

        bad_model = make_seq2seq_model()
        del bad_model.config.is_encoder_decoder

        with pytest.raises(ValueError, match="is_encoder_decoder"):
            Seq2SeqCallable(
                model=bad_model,
                tokenizer=seq2seq_tokenizer,
                device="cpu",
            )

    def test_accepts_valid_seq2seq_model(
        self,
        seq2seq_model: MagicMock,
        seq2seq_tokenizer: MagicMock,
    ) -> None:
        """No exception must be raised when is_encoder_decoder=True."""
        seq2seq_tokenizer.encode_queue = [[1]]

        Seq2SeqCallable(
            model=seq2seq_model,
            tokenizer=seq2seq_tokenizer,
            device="cpu",
        )

    def test_rejects_empty_target_label(
        self,
        seq2seq_model: MagicMock,
        seq2seq_tokenizer: MagicMock,
    ) -> None:
        """ValueError must be raised when the target label encodes to an empty list."""
        seq2seq_tokenizer.encode_queue = [[]]

        with pytest.raises(ValueError, match="produced no tokens"):
            Seq2SeqCallable(
                model=seq2seq_model,
                tokenizer=seq2seq_tokenizer,
                device="cpu",
                target_label="",
            )

    def test_rejects_when_no_decoder_start_token_available(
        self,
        seq2seq_tokenizer: MagicMock,
    ) -> None:
        """ValueError must be raised when both decoder_start_token_id and
        pad_token_id are None."""
        seq2seq_tokenizer.encode_queue = [[1]]
        seq2seq_tokenizer.pad_token_id = None

        bad_model = make_seq2seq_model()
        bad_model.config.decoder_start_token_id = None

        with pytest.raises(ValueError, match="decoder_start_token_id"):
            Seq2SeqCallable(
                model=bad_model,
                tokenizer=seq2seq_tokenizer,
                device="cpu",
            )

    def test_falls_back_to_pad_token_id_when_config_missing(
        self,
        seq2seq_tokenizer: MagicMock,
    ) -> None:
        """When config.decoder_start_token_id is None, tokenizer.pad_token_id
        must be used as the fallback."""
        seq2seq_tokenizer.encode_queue = [[1]]
        seq2seq_tokenizer.pad_token_id = 7

        model = make_seq2seq_model()
        model.config.decoder_start_token_id = None

        callable_obj = Seq2SeqCallable(
            model=model,
            tokenizer=seq2seq_tokenizer,
            device="cpu",
        )

        assert callable_obj.decoder_start_token_id == 7


# ============================================================================
# TEST 2 — Single-token target: output shape and dtype
# ============================================================================
# predict([text]) must return a numpy array of shape (1,) and dtype float32.
# The scalar value must be a finite negative number (log-probability).
# ============================================================================


class TestSingleTokenTarget:
    def _make_callable(
        self,
        seq2seq_model: MagicMock,
        seq2seq_tokenizer: MagicMock,
        token_id: int = 42,
        logit_value: float = 2.0,
        *,
        normalize: bool = True,
    ) -> Seq2SeqCallable:
        """Build a Seq2SeqCallable with a single-token target.

        logit_value is placed at the target token position; all other
        logits are zero. Because log_softmax(2.0) < 0, the score is
        always negative.
        """
        seq2seq_tokenizer.encode_queue = [[token_id]]

        vocab_size = 100
        logits = torch.zeros((1, 1, vocab_size))
        logits[0, 0, token_id] = logit_value
        seq2seq_model.return_value = SimpleNamespace(logits=logits)

        return Seq2SeqCallable(
            model=seq2seq_model,
            tokenizer=seq2seq_tokenizer,
            device="cpu",
            target_label="positive",
            normalize=normalize,
        )

    def test_output_is_numpy_array(
        self,
        seq2seq_model: MagicMock,
        seq2seq_tokenizer: MagicMock,
    ) -> None:
        callable_obj = self._make_callable(seq2seq_model, seq2seq_tokenizer)
        scores = callable_obj.predict(["text"])
        assert isinstance(scores, np.ndarray)

    def test_output_shape_is_one(
        self,
        seq2seq_model: MagicMock,
        seq2seq_tokenizer: MagicMock,
    ) -> None:
        callable_obj = self._make_callable(seq2seq_model, seq2seq_tokenizer)
        scores = callable_obj.predict(["text"])
        assert scores.shape == (1,)

    def test_output_dtype_is_float32(
        self,
        seq2seq_model: MagicMock,
        seq2seq_tokenizer: MagicMock,
    ) -> None:
        callable_obj = self._make_callable(seq2seq_model, seq2seq_tokenizer)
        scores = callable_obj.predict(["text"])
        assert scores.dtype == np.float32

    def test_output_value_is_finite(
        self,
        seq2seq_model: MagicMock,
        seq2seq_tokenizer: MagicMock,
    ) -> None:
        callable_obj = self._make_callable(seq2seq_model, seq2seq_tokenizer)
        scores = callable_obj.predict(["text"])
        assert np.isfinite(scores[0])

    def test_output_value_is_negative(
        self,
        seq2seq_model: MagicMock,
        seq2seq_tokenizer: MagicMock,
    ) -> None:
        """A log-probability must always be non-positive."""
        callable_obj = self._make_callable(seq2seq_model, seq2seq_tokenizer)
        scores = callable_obj.predict(["text"])
        assert scores[0] < 0

    def test_batch_output_shape_matches_input_length(
        self,
        seq2seq_model: MagicMock,
        seq2seq_tokenizer: MagicMock,
    ) -> None:
        """When predict receives N texts, the output shape must be (N,)."""
        token_id = 42
        n_texts = 3
        vocab_size = 100

        seq2seq_tokenizer.encode_queue = [[token_id]]
        seq2seq_tokenizer.return_value = {
            "input_ids": torch.tensor([[10, 11, 12]] * n_texts),
            "attention_mask": torch.ones((n_texts, 3), dtype=torch.long),
        }

        logits = torch.zeros((n_texts, 1, vocab_size))
        logits[:, 0, token_id] = 2.0
        seq2seq_model.return_value = SimpleNamespace(logits=logits)

        callable_obj = Seq2SeqCallable(
            model=seq2seq_model,
            tokenizer=seq2seq_tokenizer,
            device="cpu",
            target_label="positive",
        )

        scores = callable_obj.predict(["a", "b", "c"])
        assert scores.shape == (n_texts,)


# ============================================================================
# TEST 3 — Multi-token target: teacher-forcing loop
# ============================================================================
# When the target label contains N tokens, the decoder loop runs N times.
# At each step decoder_input_ids grows by one token, and the final score
# equals the sum of per-token log-probabilities.
# ============================================================================


class TestMultiTokenTarget:
    def _make_two_token_callable(
        self,
        seq2seq_model: MagicMock,
        seq2seq_tokenizer: MagicMock,
        token_ids: list[int],
        logit_value: float = 3.0,
        *,
        normalize: bool = False,
    ) -> Seq2SeqCallable:
        """Build a callable with a two-token target.

        Every model() call returns the same logits tensor;
        the target token positions are set to logit_value, the rest to zero.
        """
        seq2seq_tokenizer.encode_queue = [token_ids]

        vocab_size = 100
        logits = torch.zeros((1, len(token_ids), vocab_size))
        for tid in token_ids:
            logits[:, :, tid] = logit_value
        seq2seq_model.return_value = SimpleNamespace(logits=logits)

        return Seq2SeqCallable(
            model=seq2seq_model,
            tokenizer=seq2seq_tokenizer,
            device="cpu",
            target_label="very positive",
            normalize=normalize,
        )

    def test_model_called_once_per_target_token(
        self,
        seq2seq_model: MagicMock,
        seq2seq_tokenizer: MagicMock,
    ) -> None:
        """model() must be called exactly N times for an N-token target."""
        token_ids = [10, 20]
        callable_obj = self._make_two_token_callable(seq2seq_model, seq2seq_tokenizer, token_ids)

        callable_obj.predict(["text"])

        assert seq2seq_model.call_count == len(token_ids)

    def test_scores_are_summed_across_tokens(
        self,
        seq2seq_model: MagicMock,
        seq2seq_tokenizer: MagicMock,
    ) -> None:
        """With normalize=False the score must equal the sum of per-token log-probs."""
        token_ids = [10, 20]
        logit_value = 3.0
        vocab_size = 100

        seq2seq_tokenizer.encode_queue = [token_ids]

        def make_logits(tid: int) -> SimpleNamespace:
            logits = torch.zeros((1, 1, vocab_size))
            logits[0, 0, tid] = logit_value
            return SimpleNamespace(logits=logits)

        seq2seq_model.side_effect = [make_logits(10), make_logits(20)]

        callable_obj = Seq2SeqCallable(
            model=seq2seq_model,
            tokenizer=seq2seq_tokenizer,
            device="cpu",
            target_label="very positive",
            normalize=False,
        )

        scores = callable_obj.predict(["text"])

        ref = 0.0
        for tid in token_ids:
            logits = torch.zeros(vocab_size)
            logits[tid] = logit_value
            ref += torch.log_softmax(logits, dim=-1)[tid].item()

        assert abs(float(scores[0]) - ref) < 1e-5

    def test_decoder_input_ids_grow_by_one_per_step(
        self,
        seq2seq_model: MagicMock,
        seq2seq_tokenizer: MagicMock,
    ) -> None:
        """The length of decoder_input_ids passed to model() must increase by
        one at each teacher-forcing step."""
        token_ids = [10, 20, 30]
        vocab_size = 100

        seq2seq_tokenizer.encode_queue = [token_ids]

        captured_decoder_lengths: list[int] = []

        def capture_call(**kwargs) -> SimpleNamespace:
            length = kwargs["decoder_input_ids"].shape[1]
            captured_decoder_lengths.append(length)
            logits = torch.zeros((1, length, vocab_size))
            return SimpleNamespace(logits=logits)

        seq2seq_model.side_effect = capture_call

        callable_obj = Seq2SeqCallable(
            model=seq2seq_model,
            tokenizer=seq2seq_tokenizer,
            device="cpu",
            target_label="three tokens",
            normalize=False,
        )

        callable_obj.predict(["text"])

        # Step 1: decoder_input_ids = [start]           → length 1
        # Step 2: decoder_input_ids = [start, t1]       → length 2
        # Step 3: decoder_input_ids = [start, t1, t2]   → length 3
        assert captured_decoder_lengths == list(range(1, len(token_ids) + 1))


# ============================================================================
# TEST 4 — Normalisation: normalize=True vs normalize=False
# ============================================================================
# normalize=False → score = sum of per-token log-probs
# normalize=True  → score = mean of per-token log-probs = sum / n_tokens
# ============================================================================


class TestNormalization:
    def _get_score(
        self,
        seq2seq_model: MagicMock,
        seq2seq_tokenizer: MagicMock,
        token_ids: list[int],
        logit_value: float,
        *,
        normalize: bool,
    ) -> float:
        """Compute the callable score for a given token_ids target."""
        vocab_size = 100

        seq2seq_tokenizer.encode_queue = [token_ids]

        def make_logits(**kwargs) -> SimpleNamespace:
            dec_len = kwargs["decoder_input_ids"].shape[1]
            logits = torch.zeros((1, dec_len, vocab_size))
            for tid in token_ids:
                logits[:, :, tid] = logit_value
            return SimpleNamespace(logits=logits)

        seq2seq_model.side_effect = make_logits

        callable_obj = Seq2SeqCallable(
            model=seq2seq_model,
            tokenizer=seq2seq_tokenizer,
            device="cpu",
            target_label="label",
            normalize=normalize,
        )

        return float(callable_obj.predict(["text"])[0])

    def test_single_token_normalize_equals_raw(
        self,
        seq2seq_model: MagicMock,
        seq2seq_tokenizer: MagicMock,
    ) -> None:
        """For a single-token target, normalize=True and normalize=False
        must produce the same score (dividing by 1 is a no-op)."""
        raw = self._get_score(seq2seq_model, seq2seq_tokenizer, [5], 2.0, normalize=False)

        seq2seq_model.reset_mock()
        norm = self._get_score(seq2seq_model, seq2seq_tokenizer, [5], 2.0, normalize=True)

        assert abs(raw - norm) < 1e-6

    def test_multi_token_norm_equals_raw_divided_by_n(
        self,
        seq2seq_model: MagicMock,
        seq2seq_tokenizer: MagicMock,
    ) -> None:
        """For a multi-token target, norm_score must equal raw_score / n_tokens."""
        token_ids = [5, 6, 7]

        raw = self._get_score(seq2seq_model, seq2seq_tokenizer, token_ids, 2.0, normalize=False)

        seq2seq_model.reset_mock()
        norm = self._get_score(seq2seq_model, seq2seq_tokenizer, token_ids, 2.0, normalize=True)

        expected = raw / len(token_ids)
        assert abs(norm - expected) < 1e-5

    def test_multi_token_normalized_greater_than_raw(
        self,
        seq2seq_model: MagicMock,
        seq2seq_tokenizer: MagicMock,
    ) -> None:
        """For a multi-token target, the normalised score must be greater than
        the raw score, because dividing a negative number by n > 1 makes it
        less negative."""
        token_ids = [5, 6]

        raw = self._get_score(seq2seq_model, seq2seq_tokenizer, token_ids, 2.0, normalize=False)

        seq2seq_model.reset_mock()
        norm = self._get_score(seq2seq_model, seq2seq_tokenizer, token_ids, 2.0, normalize=True)

        assert norm > raw


# ============================================================================
# TEST 5 — Prompt template
# ============================================================================
# _build_prompt inserts the input text into the template string via .format().
# Different templates change the text sent to the encoder, which must be
# reflected in what the tokenizer receives.
# ============================================================================


class TestPromptTemplate:
    def test_default_template_is_noop(
        self,
        seq2seq_model: MagicMock,
        seq2seq_tokenizer: MagicMock,
    ) -> None:
        """With the default template '{text}', _build_prompt must return
        the original text unchanged."""
        seq2seq_tokenizer.encode_queue = [[1]]

        callable_obj = Seq2SeqCallable(
            model=seq2seq_model,
            tokenizer=seq2seq_tokenizer,
            device="cpu",
            target_label="positive",
            prompt_template="{text}",
        )

        assert callable_obj._build_prompt("hello world") == "hello world"

    @pytest.mark.parametrize(
        "template,text,expected",
        [
            (
                "sst2 sentence: {text}",
                "great film",
                "sst2 sentence: great film",
            ),
            (
                "Sentiment of '{text}':",
                "great film",
                "Sentiment of 'great film':",
            ),
            (
                "Q: {text}\nA:",
                "great film",
                "Q: great film\nA:",
            ),
        ],
    )
    def test_build_prompt_formats_correctly(
        self,
        seq2seq_model: MagicMock,
        seq2seq_tokenizer: MagicMock,
        template: str,
        text: str,
        expected: str,
    ) -> None:
        """_build_prompt must return the correctly formatted string for
        various template styles."""
        seq2seq_tokenizer.encode_queue = [[1]]

        callable_obj = Seq2SeqCallable(
            model=seq2seq_model,
            tokenizer=seq2seq_tokenizer,
            device="cpu",
            target_label="positive",
            prompt_template=template,
        )

        assert callable_obj._build_prompt(text) == expected

    def test_prompt_is_passed_to_tokenizer(
        self,
        seq2seq_model: MagicMock,
        seq2seq_tokenizer: MagicMock,
    ) -> None:
        """predict() must pass the rendered prompt — not the raw input text —
        to the tokenizer."""
        seq2seq_tokenizer.encode_queue = [[1]]

        vocab_size = 100
        seq2seq_model.return_value = SimpleNamespace(logits=torch.zeros((1, 1, vocab_size)))

        template = "sst2 sentence: {text}"
        input_text = "great film"
        expected_prompt = "sst2 sentence: great film"

        callable_obj = Seq2SeqCallable(
            model=seq2seq_model,
            tokenizer=seq2seq_tokenizer,
            device="cpu",
            target_label="positive",
            prompt_template=template,
        )

        callable_obj.predict([input_text])

        call_args = seq2seq_tokenizer.call_args
        actual_texts = call_args[0][0]
        assert actual_texts == [expected_prompt]

    def test_different_templates_reach_encoder(
        self,
        seq2seq_model: MagicMock,
        seq2seq_tokenizer: MagicMock,
    ) -> None:
        """Different prompt templates must cause the tokenizer to receive
        different input texts."""
        vocab_size = 100
        seq2seq_model.return_value = SimpleNamespace(logits=torch.zeros((1, 1, vocab_size)))

        prompts_seen: list[list[str]] = []

        def capture_tokenizer(texts, **kwargs):
            prompts_seen.append(texts)
            return {
                "input_ids": torch.tensor([[10, 11, 12]]),
                "attention_mask": torch.ones((1, 3), dtype=torch.long),
            }

        seq2seq_tokenizer.side_effect = capture_tokenizer

        templates = ["{text}", "sst2 sentence: {text}"]

        for template in templates:
            seq2seq_tokenizer.encode_queue = [[1]]
            callable_obj = Seq2SeqCallable(
                model=seq2seq_model,
                tokenizer=seq2seq_tokenizer,
                device="cpu",
                target_label="positive",
                prompt_template=template,
            )
            callable_obj.predict(["great film"])

        assert prompts_seen[0] != prompts_seen[1]


def test_seq2seq_callable_predicts_from_pre_tokenized_inputs(
    seq2seq_model: MagicMock,
    seq2seq_tokenizer: MagicMock,
) -> None:
    seq2seq_tokenizer.encode_queue = [[5, 6]]

    vocab_size = 100
    seq2seq_model.return_value = SimpleNamespace(
        logits=torch.zeros((2, 1, vocab_size)),
    )

    callable_obj = Seq2SeqCallable(
        model=seq2seq_model,
        tokenizer=seq2seq_tokenizer,
        device="cpu",
        target_label="positive",
    )

    inputs = [
        {
            "input_ids": torch.tensor([[10, 11, 12]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        },
        {
            "input_ids": torch.tensor([[20, 21, 22]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        },
    ]

    scores = callable_obj.predict_from_inputs(inputs)

    assert isinstance(scores, np.ndarray)
    assert scores.shape == (2,)
    assert np.all(np.isfinite(scores))

    encoder = seq2seq_model.get_encoder.return_value
    encoder.assert_called_once()

    encoder_inputs = encoder.call_args.kwargs
    assert encoder_inputs["input_ids"].shape == (2, 3)
    assert encoder_inputs["attention_mask"].shape == (2, 3)
    assert encoder_inputs["return_dict"] is True


# ============================================================================
# TEST 6 — TextImputer end-to-end integration
# ============================================================================
# When TextImputer is initialised with model_type="seq2seq", the internal
# target_callable must be a Seq2SeqCallable instance, and both
# full_prediction() and value_function() must return finite values.
# ============================================================================


class TestSeq2SeqTextImputer:
    def _make_imputer(
        self,
        seq2seq_model: MagicMock,
        seq2seq_tokenizer: MagicMock,
        player_level: str = "word",
        perturbation_type: str = "neutral",
        target_label: str = "positive",
        prompt_template: str = "{text}",
        *,
        normalize: bool = True,
    ) -> TextImputer:
        """Build a seq2seq TextImputer with minimal configuration."""
        seq2seq_tokenizer.encode_queue = [[1]]

        player_strategy = MagicMock()
        player_strategy.n_players = 3
        player_strategy.coalition_to_text.return_value = "perturbed text"

        vocab_size = 100
        seq2seq_model.return_value = SimpleNamespace(logits=torch.zeros((1, 1, vocab_size)))

        return TextImputer(
            model=seq2seq_model,
            tokenizer=seq2seq_tokenizer,
            text="original text",
            model_type="seq2seq",
            target_label=target_label,
            prompt_template=prompt_template,
            player_strategy=player_strategy,
            perturbation_strategy=NeutralPerturbation(),
            normalize_target_logprob=normalize,
            device="cpu",
        )

    def test_target_callable_is_seq2seq_callable(
        self,
        seq2seq_model: MagicMock,
        seq2seq_tokenizer: MagicMock,
    ) -> None:
        """With model_type='seq2seq', target_callable must be a
        Seq2SeqCallable instance."""
        imputer = self._make_imputer(seq2seq_model, seq2seq_tokenizer)
        assert isinstance(imputer.target_callable, Seq2SeqCallable)

    def test_target_label_forwarded_to_callable(
        self,
        seq2seq_model: MagicMock,
        seq2seq_tokenizer: MagicMock,
    ) -> None:
        """target_label must be forwarded correctly to the internal callable."""
        imputer = self._make_imputer(seq2seq_model, seq2seq_tokenizer, target_label="negative")
        assert imputer.target_callable.target_label == "negative"

    def test_prompt_template_forwarded_to_callable(
        self,
        seq2seq_model: MagicMock,
        seq2seq_tokenizer: MagicMock,
    ) -> None:
        """prompt_template must be forwarded correctly to the internal callable."""
        template = "sst2 sentence: {text}"
        imputer = self._make_imputer(seq2seq_model, seq2seq_tokenizer, prompt_template=template)
        assert imputer.target_callable.prompt_template == template

    def test_normalize_forwarded_to_callable(
        self,
        seq2seq_model: MagicMock,
        seq2seq_tokenizer: MagicMock,
    ) -> None:
        """normalize_target_logprob must be forwarded correctly to the
        internal callable."""
        imputer = self._make_imputer(seq2seq_model, seq2seq_tokenizer, normalize=False)
        assert imputer.target_callable.normalize is False

    def test_full_prediction_returns_finite_float(
        self,
        seq2seq_model: MagicMock,
        seq2seq_tokenizer: MagicMock,
    ) -> None:
        """full_prediction must be a finite float."""
        imputer = self._make_imputer(seq2seq_model, seq2seq_tokenizer)
        score = imputer.full_prediction
        assert isinstance(score, float)
        assert np.isfinite(score)

    def test_value_function_returns_correct_shape(
        self,
        seq2seq_model: MagicMock,
        seq2seq_tokenizer: MagicMock,
    ) -> None:
        """value_function([[coalition]]) must return a numpy array of shape (1,)."""
        imputer = self._make_imputer(seq2seq_model, seq2seq_tokenizer)

        imputer.target_callable = MagicMock()
        imputer.target_callable.predict.return_value = np.array([-0.8], dtype=np.float32)

        coalition = np.array([[1, 0, 1]])
        scores = imputer.value_function(coalition)

        assert isinstance(scores, np.ndarray)
        assert scores.shape == (1,)
        assert np.isfinite(scores[0])

    def test_value_function_returns_finite_scores_for_all_zero_coalition(
        self,
        seq2seq_model: MagicMock,
        seq2seq_tokenizer: MagicMock,
    ) -> None:
        """An all-zero coalition (all players masked) must still produce a
        finite score."""
        imputer = self._make_imputer(seq2seq_model, seq2seq_tokenizer)

        imputer.target_callable = MagicMock()
        imputer.target_callable.predict.return_value = np.array([-2.0], dtype=np.float32)

        coalition = np.array([[0, 0, 0]])
        scores = imputer.value_function(coalition)

        assert np.isfinite(scores[0])

    def test_encoder_reuse_across_target_tokens(
        self,
        seq2seq_model: MagicMock,
        seq2seq_tokenizer: MagicMock,
    ) -> None:
        """For a batch of texts, the encoder must be called exactly once
        regardless of the number of target tokens."""
        target_token_ids = [10, 20, 30]
        vocab_size = 100

        seq2seq_tokenizer.encode_queue = [target_token_ids]

        encoder_mock = MagicMock()
        encoder_mock.return_value = SimpleNamespace(last_hidden_state=torch.zeros((1, 3, 16)))
        seq2seq_model.get_encoder.return_value = encoder_mock

        seq2seq_model.return_value = SimpleNamespace(logits=torch.zeros((1, 1, vocab_size)))

        callable_obj = Seq2SeqCallable(
            model=seq2seq_model,
            tokenizer=seq2seq_tokenizer,
            device="cpu",
            target_label="three token label",
            normalize=False,
        )

        callable_obj.predict(["text"])

        # Encoder called once; model called once per target token
        assert encoder_mock.call_count == 1
        assert seq2seq_model.call_count == len(target_token_ids)
