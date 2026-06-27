# ============================================================================
# Seq2SeqCallable 的 pytest 单元测试
# 覆盖：模型类型校验、单 token 目标、多 token teacher forcing、
#        归一化、prompt 模板、端到端集成
# ============================================================================

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from shapiq.imputer.text_imputer_seq2seq import (
    Seq2SeqCallable,
    TextImputer,
    NeutralPerturbation,
)

MODULE = "shapiq.imputer.text_imputer"


# ============================================================================
# Mock fixtures — 不下载真实模型，所有模型调用均通过 MagicMock 完成
# ============================================================================

def make_seq2seq_tokenizer() -> MagicMock:
    """构造一个模拟 seq2seq tokenizer。

    encode() 的返回值通过 encode_queue 队列控制，
    每次调用 encode() 会从队列头部弹出一个值。
    """
    tok = MagicMock()
    tok.pad_token    = "[PAD]"
    tok.pad_token_id = 0
    tok.eos_token    = "</s>"
    tok.eos_token_id = 2

    # encode_queue：测试中按需填入，模拟 tokenizer.encode() 的返回值
    tok.encode_queue = []
    tok.encode.side_effect = lambda text, **kwargs: tok.encode_queue.pop(0)

    # __call__ 返回形状一致的 input_ids 和 attention_mask
    tok.return_value = {
        "input_ids":      torch.tensor([[10, 11, 12]]),
        "attention_mask": torch.ones((1, 3), dtype=torch.long),
    }
    return tok


def make_seq2seq_model(decoder_start_token_id: int = 0) -> MagicMock:
    """构造一个模拟 seq2seq 模型。

    config.is_encoder_decoder = True，模拟 T5/BART 类模型。
    model.get_encoder() 返回一个 encoder mock，
    encoder 的输出是一个带 last_hidden_state 的 SimpleNamespace。
    model(**kwargs) 的返回值由各测试按需覆盖。
    """
    model = MagicMock()
    model.to.return_value = model

    # 标记为 encoder-decoder 架构
    model.config.is_encoder_decoder        = True
    model.config.decoder_start_token_id    = decoder_start_token_id

    # encoder mock：返回包含 last_hidden_state 的对象
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
# TEST 1 — 模型类型校验
# ============================================================================
# Seq2SeqCallable.__init__ 读取 model.config.is_encoder_decoder。
# 若该标志为 False 或缺失，必须抛出包含 "is_encoder_decoder" 的 ValueError。
# ============================================================================

class TestModelTypeValidation:

    def test_rejects_model_with_is_encoder_decoder_false(
        self,
        seq2seq_tokenizer: MagicMock,
    ) -> None:
        """is_encoder_decoder=False 时必须抛出 ValueError。"""
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
        """config 没有 is_encoder_decoder 属性时，getattr 默认为 False，应报错。"""
        seq2seq_tokenizer.encode_queue = [[1]]

        bad_model = make_seq2seq_model()
        # 删除属性，使 getattr(..., False) 返回 False
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
        """is_encoder_decoder=True 时不应报错。"""
        seq2seq_tokenizer.encode_queue = [[1]]

        # 不抛出异常即为通过
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
        """目标标签编码为空列表时必须抛出 ValueError。"""
        seq2seq_tokenizer.encode_queue = [[]]   # 编码结果为空

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
        """decoder_start_token_id 和 pad_token_id 均为 None 时必须报错。"""
        seq2seq_tokenizer.encode_queue  = [[1]]
        seq2seq_tokenizer.pad_token_id  = None

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
        """config 没有 decoder_start_token_id 时应退回使用 tokenizer.pad_token_id。"""
        seq2seq_tokenizer.encode_queue  = [[1]]
        seq2seq_tokenizer.pad_token_id  = 7

        model = make_seq2seq_model()
        model.config.decoder_start_token_id = None

        callable_obj = Seq2SeqCallable(
            model=model,
            tokenizer=seq2seq_tokenizer,
            device="cpu",
        )

        assert callable_obj.decoder_start_token_id == 7


# ============================================================================
# TEST 2 — 单 token 目标：输出格式与数据类型
# ============================================================================
# predict([text]) 必须返回形状 (1,)、dtype float32 的 numpy 数组，
# 且数值为有限负数（log-probability）。
# ============================================================================

class TestSingleTokenTarget:

    def _make_callable(
        self,
        seq2seq_model: MagicMock,
        seq2seq_tokenizer: MagicMock,
        token_id: int = 42,
        logit_value: float = 2.0,
        normalize: bool = True,
    ) -> Seq2SeqCallable:
        """辅助：构造一个单 token 目标的 Seq2SeqCallable。

        logit_value：目标 token 位置的 logit，其余位置为 0。
        由于 log_softmax(2.0) < 0，返回值必然是负数。
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
        """log-probability 必须为负数。"""
        callable_obj = self._make_callable(seq2seq_model, seq2seq_tokenizer)
        scores = callable_obj.predict(["text"])
        assert scores[0] < 0

    def test_batch_output_shape_matches_input_length(
        self,
        seq2seq_model: MagicMock,
        seq2seq_tokenizer: MagicMock,
    ) -> None:
        """predict 接收 N 条文本时，输出形状应为 (N,)。"""
        token_id   = 42
        n_texts    = 3
        vocab_size = 100

        seq2seq_tokenizer.encode_queue = [[token_id]]
        seq2seq_tokenizer.return_value = {
            "input_ids":      torch.tensor([[10, 11, 12]] * n_texts),
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
# TEST 3 — 多 token 目标：teacher forcing 循环
# ============================================================================
# 目标标签包含 N 个 token 时，decoder 循环 N 次，
# 每次 decoder_input_ids 增长 1 个 token，
# 最终分数等于各 token log-prob 之和。
# ============================================================================

class TestMultiTokenTarget:

    def _make_two_token_callable(
        self,
        seq2seq_model: MagicMock,
        seq2seq_tokenizer: MagicMock,
        token_ids: list[int],
        logit_value: float = 3.0,
        normalize: bool = False,
    ) -> Seq2SeqCallable:
        """构造一个两 token 目标的 callable。

        每次调用 model() 都返回相同的 logits，
        两个目标 token 的 logit 均设为 logit_value，其余为 0。
        """
        seq2seq_tokenizer.encode_queue = [token_ids]

        vocab_size = 100
        logits     = torch.zeros((1, len(token_ids), vocab_size))
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
        """N 个目标 token 应触发 N 次 model() 调用。"""
        token_ids    = [10, 20]
        callable_obj = self._make_two_token_callable(
            seq2seq_model, seq2seq_tokenizer, token_ids
        )

        callable_obj.predict(["text"])

        # model() 被调用的次数等于目标 token 数
        assert seq2seq_model.call_count == len(token_ids)

    def test_scores_are_summed_across_tokens(
        self,
        seq2seq_model: MagicMock,
        seq2seq_tokenizer: MagicMock,
    ) -> None:
        """normalize=False 时，最终分数应等于各 token log-prob 之和。"""
        token_ids    = [10, 20]
        logit_value  = 3.0
        vocab_size   = 100

        seq2seq_tokenizer.encode_queue = [token_ids]

        # 两次 model() 调用各返回一个独立的 logits
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

        # 手动计算参考值
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
        """验证每步传入 model() 的 decoder_input_ids 长度递增 1。"""
        token_ids  = [10, 20, 30]
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

        # 第 1 步：decoder_input_ids = [start]         → 长度 1
        # 第 2 步：decoder_input_ids = [start, t1]     → 长度 2
        # 第 3 步：decoder_input_ids = [start, t1, t2] → 长度 3
        assert captured_decoder_lengths == list(range(1, len(token_ids) + 1))


# ============================================================================
# TEST 4 — 归一化：normalize=True 与 normalize=False
# ============================================================================
# normalize=False → 分数 = 各 token log-prob 之和
# normalize=True  → 分数 = 各 token log-prob 的平均值 = 总和 / n_tokens
# ============================================================================

class TestNormalization:

    def _get_score(
        self,
        seq2seq_model: MagicMock,
        seq2seq_tokenizer: MagicMock,
        token_ids: list[int],
        logit_value: float,
        normalize: bool,
    ) -> float:
        """辅助：对 token_ids 定义的目标标签计算得分。"""
        vocab_size = 100

        seq2seq_tokenizer.encode_queue = [token_ids]

        def make_logits(**kwargs) -> SimpleNamespace:
            dec_len = kwargs["decoder_input_ids"].shape[1]
            logits  = torch.zeros((1, dec_len, vocab_size))
            # 把所有目标 token 的 logit 都设为相同值，简化参考值计算
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
        """单 token 目标：normalize=True 与 normalize=False 结果相同（除以 1 不变）。"""
        raw  = self._get_score(seq2seq_model, seq2seq_tokenizer, [5], 2.0, False)

        seq2seq_model.reset_mock()
        norm = self._get_score(seq2seq_model, seq2seq_tokenizer, [5], 2.0, True)

        assert abs(raw - norm) < 1e-6

    def test_multi_token_norm_equals_raw_divided_by_n(
        self,
        seq2seq_model: MagicMock,
        seq2seq_tokenizer: MagicMock,
    ) -> None:
        """多 token 目标：norm_score == raw_score / n_tokens。"""
        token_ids = [5, 6, 7]

        raw  = self._get_score(seq2seq_model, seq2seq_tokenizer, token_ids, 2.0, False)

        seq2seq_model.reset_mock()
        norm = self._get_score(seq2seq_model, seq2seq_tokenizer, token_ids, 2.0, True)

        expected = raw / len(token_ids)
        assert abs(norm - expected) < 1e-5

    def test_multi_token_normalized_greater_than_raw(
        self,
        seq2seq_model: MagicMock,
        seq2seq_tokenizer: MagicMock,
    ) -> None:
        """多 token 目标：归一化分数 > 原始分数（负数除以 n>1，绝对值缩小）。"""
        token_ids = [5, 6]

        raw  = self._get_score(seq2seq_model, seq2seq_tokenizer, token_ids, 2.0, False)

        seq2seq_model.reset_mock()
        norm = self._get_score(seq2seq_model, seq2seq_tokenizer, token_ids, 2.0, True)

        assert norm > raw


# ============================================================================
# TEST 5 — Prompt 模板
# ============================================================================
# _build_prompt 通过 .format() 将输入文本插入模板字符串。
# 不同模板会改变送入 encoder 的文本，从而产生不同的分数。
# ============================================================================

class TestPromptTemplate:

    def test_default_template_is_noop(
        self,
        seq2seq_model: MagicMock,
        seq2seq_tokenizer: MagicMock,
    ) -> None:
        """默认模板 '{text}' 下，_build_prompt 返回原始文本本身。"""
        seq2seq_tokenizer.encode_queue = [[1]]

        callable_obj = Seq2SeqCallable(
            model=seq2seq_model,
            tokenizer=seq2seq_tokenizer,
            device="cpu",
            target_label="positive",
            prompt_template="{text}",
        )

        assert callable_obj._build_prompt("hello world") == "hello world"

    @pytest.mark.parametrize("template,text,expected", [
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
    ])
    def test_build_prompt_formats_correctly(
        self,
        seq2seq_model: MagicMock,
        seq2seq_tokenizer: MagicMock,
        template: str,
        text: str,
        expected: str,
    ) -> None:
        """_build_prompt 对各种模板格式均返回正确字符串。"""
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
        """predict() 调用 tokenizer 时，传入的应是 prompt 而非原始文本。"""
        seq2seq_tokenizer.encode_queue = [[1]]

        vocab_size = 100
        seq2seq_model.return_value = SimpleNamespace(
            logits=torch.zeros((1, 1, vocab_size))
        )

        template     = "sst2 sentence: {text}"
        input_text   = "great film"
        expected_prompt = "sst2 sentence: great film"

        callable_obj = Seq2SeqCallable(
            model=seq2seq_model,
            tokenizer=seq2seq_tokenizer,
            device="cpu",
            target_label="positive",
            prompt_template=template,
        )

        callable_obj.predict([input_text])

        # tokenizer.__call__ 的第一个位置参数应是包含 prompt 的列表
        call_args = seq2seq_tokenizer.call_args
        actual_texts = call_args[0][0]          # 第一个位置参数
        assert actual_texts == [expected_prompt]

    def test_different_templates_reach_encoder(
        self,
        seq2seq_model: MagicMock,
        seq2seq_tokenizer: MagicMock,
    ) -> None:
        """不同模板应使 tokenizer 收到不同的文本。"""
        vocab_size = 100
        seq2seq_model.return_value = SimpleNamespace(
            logits=torch.zeros((1, 1, vocab_size))
        )

        prompts_seen: list[list[str]] = []

        def capture_tokenizer(texts, **kwargs):
            prompts_seen.append(texts)
            return {
                "input_ids":      torch.tensor([[10, 11, 12]]),
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

        # 两次调用 tokenizer 收到的文本列表应不同
        assert prompts_seen[0] != prompts_seen[1]


# ============================================================================
# TEST 6 — TextImputer 端到端集成
# ============================================================================
# TextImputer 以 model_type="seq2seq" 初始化时，
# 内部 target_callable 必须是 Seq2SeqCallable 实例，
# full_prediction() 和 value_function() 应返回有限数值。
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
        normalize: bool = True,
    ) -> TextImputer:
        """辅助：以最小配置构造一个 seq2seq TextImputer。"""
        seq2seq_tokenizer.encode_queue = [[1]]   # target_label token id

        player_strategy = MagicMock()
        player_strategy.n_players = 3
        player_strategy.coalition_to_text.return_value = "perturbed text"

        vocab_size = 100
        seq2seq_model.return_value = SimpleNamespace(
            logits=torch.zeros((1, 1, vocab_size))
        )

        imputer = TextImputer(
            model=seq2seq_model,
            tokenizer=seq2seq_tokenizer,
            text="original text",
            model_type="seq2seq",
            target_label=target_label,
            prompt_template=prompt_template,
            player_strategy=player_strategy,
            perturbation_strategy=NeutralPerturbation(),
            normalize_target_logprob=normalize,
        )
        return imputer

    def test_target_callable_is_seq2seq_callable(
        self,
        seq2seq_model: MagicMock,
        seq2seq_tokenizer: MagicMock,
    ) -> None:
        """model_type='seq2seq' 时，target_callable 应为 Seq2SeqCallable 实例。"""
        imputer = self._make_imputer(seq2seq_model, seq2seq_tokenizer)
        assert isinstance(imputer.target_callable, Seq2SeqCallable)

    def test_target_label_forwarded_to_callable(
        self,
        seq2seq_model: MagicMock,
        seq2seq_tokenizer: MagicMock,
    ) -> None:
        """target_label 参数应正确传递给内部 Seq2SeqCallable。"""
        imputer = self._make_imputer(
            seq2seq_model, seq2seq_tokenizer, target_label="negative"
        )
        assert imputer.target_callable.target_label == "negative"

    def test_prompt_template_forwarded_to_callable(
        self,
        seq2seq_model: MagicMock,
        seq2seq_tokenizer: MagicMock,
    ) -> None:
        """prompt_template 参数应正确传递给内部 Seq2SeqCallable。"""
        template = "sst2 sentence: {text}"
        imputer  = self._make_imputer(
            seq2seq_model, seq2seq_tokenizer, prompt_template=template
        )
        assert imputer.target_callable.prompt_template == template

    def test_normalize_forwarded_to_callable(
        self,
        seq2seq_model: MagicMock,
        seq2seq_tokenizer: MagicMock,
    ) -> None:
        """normalize_target_logprob 参数应正确传递给内部 Seq2SeqCallable。"""
        imputer = self._make_imputer(
            seq2seq_model, seq2seq_tokenizer, normalize=False
        )
        assert imputer.target_callable.normalize is False

    def test_full_prediction_returns_finite_float(
        self,
        seq2seq_model: MagicMock,
        seq2seq_tokenizer: MagicMock,
    ) -> None:
        """full_prediction() 应返回有限浮点数。"""
        imputer = self._make_imputer(seq2seq_model, seq2seq_tokenizer)

        imputer.target_callable = MagicMock()
        imputer.target_callable.predict.return_value = np.array([-1.5],
                                                                  dtype=np.float32)

        score = imputer.full_prediction()
        assert isinstance(score, float)
        assert np.isfinite(score)

    def test_value_function_returns_correct_shape(
        self,
        seq2seq_model: MagicMock,
        seq2seq_tokenizer: MagicMock,
    ) -> None:
        """value_function([[coalition]]) 应返回形状 (1,) 的数组。"""
        imputer = self._make_imputer(seq2seq_model, seq2seq_tokenizer)

        imputer.target_callable = MagicMock()
        imputer.target_callable.predict.return_value = np.array([-0.8],
                                                                  dtype=np.float32)

        coalition = np.array([[1, 0, 1]])
        scores    = imputer.value_function(coalition)

        assert isinstance(scores, np.ndarray)
        assert scores.shape == (1,)
        assert np.isfinite(scores[0])

    def test_value_function_returns_finite_scores_for_all_zero_coalition(
        self,
        seq2seq_model: MagicMock,
        seq2seq_tokenizer: MagicMock,
    ) -> None:
        """全零联盟（所有词被扰动）也应返回有限分数。"""
        imputer = self._make_imputer(seq2seq_model, seq2seq_tokenizer)

        imputer.target_callable = MagicMock()
        imputer.target_callable.predict.return_value = np.array([-2.0],
                                                                  dtype=np.float32)

        coalition = np.array([[0, 0, 0]])
        scores    = imputer.value_function(coalition)

        assert np.isfinite(scores[0])

    def test_encoder_reuse_across_target_tokens(
        self,
        seq2seq_model: MagicMock,
        seq2seq_tokenizer: MagicMock,
    ) -> None:
        """对一批文本，encoder 只应被调用一次（不随 target token 数增加）。"""
        target_token_ids = [10, 20, 30]   # 3 个 token 的目标标签
        vocab_size       = 100

        seq2seq_tokenizer.encode_queue = [target_token_ids]

        encoder_mock = MagicMock()
        encoder_mock.return_value = SimpleNamespace(
            last_hidden_state=torch.zeros((1, 3, 16))
        )
        seq2seq_model.get_encoder.return_value = encoder_mock

        seq2seq_model.return_value = SimpleNamespace(
            logits=torch.zeros((1, 1, vocab_size))
        )

        callable_obj = Seq2SeqCallable(
            model=seq2seq_model,
            tokenizer=seq2seq_tokenizer,
            device="cpu",
            target_label="three token label",
            normalize=False,
        )

        callable_obj.predict(["text"])

        # encoder 只被调用一次，而不是 n_tokens 次
        assert encoder_mock.call_count == 1
        # model() 被调用 n_tokens 次（每个 target token 一次）
        assert seq2seq_model.call_count == len(target_token_ids)