"""
Seq2SeqCallable 及其与 TextImputer 集成的提交前测试。

测试顺序：
    1. 模型类型校验  — 拒绝非 seq2seq 模型（BERT、Gemma）
    2. 单 token 目标 — 输出格式与数据类型
    3. 多 token 目标 — 跨多个 token 的 teacher forcing 循环
    4. 归一化        — normalize=True 与 normalize=False 的对比
    5. Prompt 模板   — prompt 在编码前是否被正确应用
    6. 端到端集成    — 所有 player level × 所有 perturbation type 组合
"""

import os
import types

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
from transformers import (
    AutoModelForMaskedLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

from shapiq.imputer.text_imputer_seq2seq import Seq2SeqCallable, TextImputer

# ---------------------------------------------------------------------------
# 共享测试配置
# ---------------------------------------------------------------------------

SEQ2SEQ_MODEL_NAME = "google/flan-t5-small"
BERT_MODEL_NAME    = "bert-base-uncased"

INPUT_TEXT = "The movie was surprisingly good and very entertaining."

# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

PASS = "通过"
FAIL = "失败"


def _header(title: str) -> None:
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


def _subheader(title: str) -> None:
    print("\n  " + "-" * 60)
    print(f"  {title}")
    print("  " + "-" * 60)


def _result(label: str, ok: bool, detail: str = "") -> None:
    tag  = PASS if ok else FAIL
    line = f"  [{tag}]  {label}"
    if detail:
        line += f"  →  {detail}"
    print(line)


# ===========================================================================
# 测试 1 — 模型类型校验
# ===========================================================================
# 预期行为
# --------
# Seq2SeqCallable.__init__ 读取 model.config.is_encoder_decoder。
# 若该标志缺失或为 False，构造函数必须抛出 ValueError，
# 且错误信息中需包含 "is_encoder_decoder"。
#
# 测试模型：
#   • BERT（仅编码器）  → is_encoder_decoder = False → 必须报错
#   • Gemma（因果 LM） → is_encoder_decoder = False → 必须报错
#   • FLAN-T5（seq2seq）→ is_encoder_decoder = True  → 不得报错
# ===========================================================================

def test_1_model_type_validation(seq2seq_model, seq2seq_tokenizer) -> None:
    _header("测试 1 — 模型类型校验")

    # ------------------------------------------------------------------
    # 1a  BERT 编码器模型必须被拒绝
    # ------------------------------------------------------------------
    _subheader("1a  BERT（仅编码器）— 预期抛出 ValueError")

    bert_model     = AutoModelForMaskedLM.from_pretrained(BERT_MODEL_NAME)
    bert_tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)

    print(f"  BERT config.is_encoder_decoder = "
          f"{getattr(bert_model.config, 'is_encoder_decoder', False)}")

    raised    = False
    error_msg = ""
    try:
        Seq2SeqCallable(
            model=bert_model,
            tokenizer=bert_tokenizer,
            device="cpu",
        )
    except ValueError as exc:
        raised    = True
        error_msg = str(exc)

    _result(
        "BERT 抛出 ValueError",
        raised,
        error_msg[:80] if raised else "未抛出异常",
    )
    _result(
        "错误信息包含 'is_encoder_decoder'",
        "is_encoder_decoder" in error_msg,
        error_msg[:80],
    )

    # ------------------------------------------------------------------
    # 1b  合成的因果 LM 配置必须被拒绝
    #     （通过 monkey-patch 模拟 Gemma，避免下载大模型）
    # ------------------------------------------------------------------
    _subheader("1b  合成因果 LM 配置（模拟 Gemma）— 预期抛出 ValueError")

    # 临时将 seq2seq 模型的配置标志改为 False，模拟因果 LM
    original_flag = seq2seq_model.config.is_encoder_decoder
    seq2seq_model.config.is_encoder_decoder = False
    print(f"  已将 config.is_encoder_decoder 临时修改为 "
          f"{seq2seq_model.config.is_encoder_decoder}")

    raised    = False
    error_msg = ""
    try:
        Seq2SeqCallable(
            model=seq2seq_model,
            tokenizer=seq2seq_tokenizer,
            device="cpu",
        )
    except ValueError as exc:
        raised    = True
        error_msg = str(exc)
    finally:
        # 还原原始标志，避免影响后续测试
        seq2seq_model.config.is_encoder_decoder = original_flag

    _result(
        "因果 LM 配置抛出 ValueError",
        raised,
        error_msg[:80] if raised else "未抛出异常",
    )

    # ------------------------------------------------------------------
    # 1c  FLAN-T5（真正的 seq2seq）必须被接受
    # ------------------------------------------------------------------
    _subheader("1c  FLAN-T5（seq2seq）— 预期不报错")

    print(f"  FLAN-T5 config.is_encoder_decoder = "
          f"{getattr(seq2seq_model.config, 'is_encoder_decoder', False)}")

    accepted  = False
    error_msg = ""
    try:
        Seq2SeqCallable(
            model=seq2seq_model,
            tokenizer=seq2seq_tokenizer,
            device="cpu",
        )
        accepted = True
    except Exception as exc:
        error_msg = str(exc)

    _result(
        "FLAN-T5 被成功接受，无异常",
        accepted,
        "正常" if accepted else error_msg[:80],
    )


# ===========================================================================
# 测试 2 — 单 token 目标：输出格式与数据类型
# ===========================================================================
# 预期行为
# --------
# predict([text]) 返回形状为 (1,)、dtype 为 float32 的 numpy 数组。
# 数值必须是有限的负数（因为是 log-probability）。
#
# "positive" 在 FLAN-T5 词汇表中编码为单个 token。
# 测试中显式验证这一假设，使测试具有自解释性。
# ===========================================================================

def test_2_single_token_target(seq2seq_model, seq2seq_tokenizer) -> None:
    _header("测试 2 — 单 token 目标：输出格式与数据类型")

    target_label = "positive"
    token_ids    = seq2seq_tokenizer.encode(target_label, add_special_tokens=False)
    n_tokens     = len(token_ids)

    print(f"  目标标签  = '{target_label}'")
    print(f"  token_ids = {token_ids}  (共 {n_tokens} 个)")

    _result(
        f"'{target_label}' 编码为恰好 1 个 token",
        n_tokens == 1,
        f"token_ids={token_ids}",
    )

    callable_obj = Seq2SeqCallable(
        model=seq2seq_model,
        tokenizer=seq2seq_tokenizer,
        device="cpu",
        target_label=target_label,
        normalize=True,
    )

    scores = callable_obj.predict([INPUT_TEXT])

    print(f"\n  predict([INPUT_TEXT]) 的输出：")
    print(f"    类型   : {type(scores)}")
    print(f"    形状   : {scores.shape}")
    print(f"    dtype  : {scores.dtype}")
    print(f"    数值   : {scores[0]:.6f}")

    _result("返回类型为 np.ndarray",            isinstance(scores, np.ndarray))
    _result("形状为 (1,)",                       scores.shape == (1,),          str(scores.shape))
    _result("dtype 为 float32",                  scores.dtype == np.float32,    str(scores.dtype))
    _result("数值有限（非 NaN/Inf）",             np.isfinite(scores[0]),        f"{scores[0]:.4f}")
    _result("数值为负数（符合 log-prob 特性）",   scores[0] < 0,                 f"{scores[0]:.4f}")


# ===========================================================================
# 测试 3 — 多 token 目标：teacher forcing 循环
# ===========================================================================
# 预期行为
# --------
# 多 token 目标（如 "very positive"）逐 token 打分。
# 验证方法：
#   a) 比较 Seq2SeqCallable 的输出与测试内手动实现的 teacher forcing 参考值。
#      两者误差必须小于 1e-5。
#   b) 追踪每步 decoder_input_ids 的长度，确认循环每步增长 1。
# ===========================================================================

def test_3_multi_token_target(seq2seq_model, seq2seq_tokenizer) -> None:
    _header("测试 3 — 多 token 目标：teacher forcing 循环")

    target_label = "very positive"
    token_ids    = seq2seq_tokenizer.encode(target_label, add_special_tokens=False)

    print(f"  目标标签  = '{target_label}'")
    print(f"  token_ids = {token_ids}  (共 {len(token_ids)} 个)")

    _result(
        f"'{target_label}' 编码为多于 1 个 token",
        len(token_ids) > 1,
        f"n_tokens={len(token_ids)}",
    )

    callable_obj = Seq2SeqCallable(
        model=seq2seq_model,
        tokenizer=seq2seq_tokenizer,
        device="cpu",
        target_label=target_label,
        normalize=False,     # 使用原始累加值，便于与手动参考值比较
    )

    scores      = callable_obj.predict([INPUT_TEXT])
    model_score = float(scores[0])

    # ------------------------------------------------------------------
    # 参考分数：在测试内独立实现的手动 teacher forcing 循环
    # ------------------------------------------------------------------
    encoded = seq2seq_tokenizer(
        INPUT_TEXT, return_tensors="pt", truncation=True
    )
    encoded = {k: v.to("cpu") for k, v in encoded.items()}

    with torch.no_grad():
        encoder_outputs = seq2seq_model.get_encoder()(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            return_dict=True,
        )

    start_id          = seq2seq_model.config.decoder_start_token_id
    decoder_input_ids = torch.tensor([[start_id]])
    ref_log_prob      = 0.0
    step_sizes        = []   # 记录每步 decoder 序列长度

    for tid in token_ids:
        step_sizes.append(decoder_input_ids.shape[1])

        with torch.no_grad():
            out = seq2seq_model(
                encoder_outputs=encoder_outputs,
                attention_mask=encoded["attention_mask"],
                decoder_input_ids=decoder_input_ids,
            )
        lp            = torch.log_softmax(out.logits[:, -1, :], dim=-1)
        ref_log_prob += lp[0, tid].item()
        decoder_input_ids = torch.cat(
            [decoder_input_ids, torch.tensor([[tid]])], dim=1
        )

    print(f"\n  每步 decoder_input_ids 长度变化 : {step_sizes}")
    print(f"  Seq2SeqCallable 分数             = {model_score:.8f}")
    print(f"  手动参考分数                     = {ref_log_prob:.8f}")
    print(f"  绝对误差                         = {abs(model_score - ref_log_prob):.2e}")

    _result(
        "每步 decoder 序列长度增长 1",
        step_sizes == list(range(1, len(token_ids) + 1)),
        str(step_sizes),
    )
    _result(
        "Seq2SeqCallable 与手动参考值一致（误差 < 1e-5）",
        abs(model_score - ref_log_prob) < 1e-5,
        f"误差={abs(model_score - ref_log_prob):.2e}",
    )


# ===========================================================================
# 测试 4 — 归一化：normalize=True 与 normalize=False
# ===========================================================================
# 预期行为
# --------
# normalize=False：分数等于各 token log-prob 之和。
# normalize=True ：分数等于各 token log-prob 的平均值。
#
# 验证项：
#   a) raw_score  == 手动求和         （normalize=False 与手动结果一致）
#   b) norm_score == raw_score / n_tokens
#   c) 单 token 目标：raw == norm     （除以 1 不改变结果）
#   d) 多 token 目标：norm > raw      （负数除以 n > 1，绝对值变小）
# ===========================================================================

def test_4_normalization(seq2seq_model, seq2seq_tokenizer) -> None:
    _header("测试 4 — 归一化：normalize=True 与 normalize=False")

    labels = {
        "单 token   'positive'":      "positive",
        "多 token   'very positive'": "very positive",
    }

    print(f"  {'标签':<28} {'token数':>8} {'原始分数':>18} "
          f"{'归一化分数':>18} {'raw/n':>18} {'一致?':>8}")
    print("  " + "-" * 100)

    for label_name, label in labels.items():
        n_tokens = len(
            seq2seq_tokenizer.encode(label, add_special_tokens=False)
        )

        raw_callable = Seq2SeqCallable(
            model=seq2seq_model,
            tokenizer=seq2seq_tokenizer,
            device="cpu",
            target_label=label,
            normalize=False,
        )
        norm_callable = Seq2SeqCallable(
            model=seq2seq_model,
            tokenizer=seq2seq_tokenizer,
            device="cpu",
            target_label=label,
            normalize=True,
        )

        raw_score  = float(raw_callable.predict([INPUT_TEXT])[0])
        norm_score = float(norm_callable.predict([INPUT_TEXT])[0])
        expected   = raw_score / n_tokens
        match      = abs(norm_score - expected) < 1e-5

        print(f"  {label_name:<28} {n_tokens:>8} {raw_score:>18.6f} "
              f"{norm_score:>18.6f} {expected:>18.6f} {'是' if match else '否':>8}")

        _result(
            f"norm == raw / n_tokens  对标签 '{label}'",
            match,
            f"误差={abs(norm_score - expected):.2e}",
        )

    # 额外检验：多 token 目标的归一化分数必须大于原始分数
    # （负数除以大于 1 的整数，绝对值变小，即数值更大）
    raw_multi = float(
        Seq2SeqCallable(
            model=seq2seq_model,
            tokenizer=seq2seq_tokenizer,
            device="cpu",
            target_label="very positive",
            normalize=False,
        ).predict([INPUT_TEXT])[0]
    )
    norm_multi = float(
        Seq2SeqCallable(
            model=seq2seq_model,
            tokenizer=seq2seq_tokenizer,
            device="cpu",
            target_label="very positive",
            normalize=True,
        ).predict([INPUT_TEXT])[0]
    )

    _result(
        "多 token 目标：归一化分数 > 原始分数（负数绝对值缩小）",
        norm_multi > raw_multi,
        f"原始={raw_multi:.4f}  归一化={norm_multi:.4f}",
    )


# ===========================================================================
# 测试 5 — Prompt 模板
# ===========================================================================
# 预期行为
# --------
# _build_prompt 通过 .format() 将输入文本插入模板字符串。
# 验证项：
#   a) 对多种模板，_build_prompt 返回正确的字符串。
#   b) 不同模板产生不同的预测分数（说明 encoder 输入确实发生了变化）。
#   c) 默认模板 "{text}" 是 no-op（prompt 等于原始文本）。
# ===========================================================================

def test_5_prompt_template(seq2seq_model, seq2seq_tokenizer) -> None:
    _header("测试 5 — Prompt 模板")

    templates = {
        "默认（无操作）":   "{text}",
        "SST-2 格式":       "sst2 sentence: {text}",
        "问答格式":         "Is this review positive or negative?\n{text}\nAnswer:",
        "指令格式":         "Classify sentiment:\n{text}\nLabel:",
    }

    # ------------------------------------------------------------------
    # 5a  _build_prompt 字符串构造验证
    # ------------------------------------------------------------------
    _subheader("5a  _build_prompt 字符串构造验证")

    for template_name, template in templates.items():
        callable_obj  = Seq2SeqCallable(
            model=seq2seq_model,
            tokenizer=seq2seq_tokenizer,
            device="cpu",
            target_label="positive",
            prompt_template=template,
        )
        actual_prompt = callable_obj._build_prompt(INPUT_TEXT)
        expected      = template.format(text=INPUT_TEXT)
        ok            = actual_prompt == expected

        print(f"\n  模板名称 : {template_name}")
        print(f"  预期结果 : {expected[:80]}")
        print(f"  实际结果 : {actual_prompt[:80]}")
        _result("_build_prompt 与 .format() 结果一致", ok)

    # ------------------------------------------------------------------
    # 5b  默认模板为 no-op 验证
    # ------------------------------------------------------------------
    _subheader("5b  默认模板 '{text}' 为 no-op 验证")

    default_callable = Seq2SeqCallable(
        model=seq2seq_model,
        tokenizer=seq2seq_tokenizer,
        device="cpu",
        target_label="positive",
        prompt_template="{text}",
    )
    _result(
        "默认模板下 _build_prompt(INPUT_TEXT) == INPUT_TEXT",
        default_callable._build_prompt(INPUT_TEXT) == INPUT_TEXT,
        f"'{default_callable._build_prompt(INPUT_TEXT)[:60]}'",
    )

    # ------------------------------------------------------------------
    # 5c  不同模板产生不同分数
    # ------------------------------------------------------------------
    _subheader("5c  不同模板产生不同预测分数")

    scores = {}
    for template_name, template in templates.items():
        c = Seq2SeqCallable(
            model=seq2seq_model,
            tokenizer=seq2seq_tokenizer,
            device="cpu",
            target_label="positive",
            prompt_template=template,
            normalize=True,
        )
        scores[template_name] = float(c.predict([INPUT_TEXT])[0])

    print(f"\n  {'模板名称':<20} {'分数':>14}")
    print("  " + "-" * 36)
    for name, sc in scores.items():
        print(f"  {name:<20} {sc:>14.6f}")

    unique_scores = len(set(round(v, 6) for v in scores.values()))
    _result(
        "所有模板产生不同的分数",
        unique_scores == len(scores),
        f"{unique_scores} 个唯一值，共 {len(scores)} 个模板",
    )


# ===========================================================================
# 测试 6 — 通过 TextImputer 的端到端集成
# ===========================================================================
# 预期行为
# --------
# TextImputer 将 Seq2SeqCallable 与每种 player strategy 和 perturbation
# strategy 连接。对每种组合验证：
#   • full_prediction() 返回有限浮点数
#   • value_function([coalition]) 返回形状为 (1,) 的有限 numpy 数组
#   • imputer 内部的 target_callable 是 Seq2SeqCallable 实例
#
# 测试的 player level：word、named_entity、chunk、sentence
#   （subword 跳过，因为 FLAN-T5 tokenizer 没有 [MASK] token）
#
# 测试的 perturbation strategy：pad、removal、neutral、wordnet_neutral
#   （mask 跳过，原因同上）
#   （mlm_infilling 单独在子测试 6b 中测试，因为需要下载额外模型）
# ===========================================================================

def test_6_end_to_end_integration(seq2seq_model, seq2seq_tokenizer) -> None:
    _header("测试 6 — 通过 TextImputer 的端到端集成")

    player_levels = [
        "word",
        "named_entity",
        "chunk",
        "sentence",
    ]

    perturbation_types = [
        "pad",
        "removal",
        "neutral",
        "wordnet_neutral",
    ]

    print(f"\n  输入文本   : {INPUT_TEXT}")
    print(f"  模型       : {SEQ2SEQ_MODEL_NAME}")
    print(f"  model_type : seq2seq\n")

    results_table = []

    for player_level in player_levels:
        for perturbation_type in perturbation_types:
            label = f"{player_level:15s} × {perturbation_type}"
            try:
                imputer = TextImputer(
                    model=seq2seq_model,
                    tokenizer=seq2seq_tokenizer,
                    text=INPUT_TEXT,
                    model_type="seq2seq",
                    target_label="positive",
                    prompt_template="sst2 sentence: {text}",
                    player_level=player_level,
                    perturbation_type=perturbation_type,
                    normalize_target_logprob=True,
                )

                # 验证 target_callable 类型
                is_seq2seq = isinstance(imputer.target_callable, Seq2SeqCallable)

                # 完整预测
                full_score = imputer.full_prediction()
                full_ok    = np.isfinite(full_score)

                # 联盟预测：仅保留第一个 player
                n         = imputer.n_players
                coalition = np.zeros(n, dtype=int)
                coalition[0] = 1

                coalition_scores = imputer.value_function([coalition])
                coalition_ok     = (
                    isinstance(coalition_scores, np.ndarray)
                    and coalition_scores.shape == (1,)
                    and np.isfinite(coalition_scores[0])
                )

                overall_ok = is_seq2seq and full_ok and coalition_ok
                status     = PASS if overall_ok else FAIL

                results_table.append((
                    label, status, n,
                    full_score, float(coalition_scores[0]),
                ))

            except Exception as exc:
                results_table.append((label, f"错误: {exc}", "-", "-", "-"))

    # 打印汇总表格
    print(f"  {'组合':<38} {'状态':<6} {'玩家数':>8} "
          f"{'完整分数':>14} {'联盟分数':>14}")
    print("  " + "-" * 84)

    all_passed = True
    for row in results_table:
        label, status, n, full_sc, coal_sc = row
        full_str = f"{full_sc:.4f}" if isinstance(full_sc, float) else str(full_sc)
        coal_str = f"{coal_sc:.4f}" if isinstance(coal_sc, float) else str(coal_sc)
        print(f"  {label:<38} {status:<6} {str(n):>8} "
              f"{full_str:>14} {coal_str:>14}")
        if status != PASS:
            all_passed = False

    _result("所有 player × perturbation 组合均通过", all_passed)

    # ------------------------------------------------------------------
    # 6b  MLM infilling 集成测试（单独子测试，需下载 BERT）
    # ------------------------------------------------------------------
    _subheader("6b  MLM infilling 与 word player 集成测试")

    try:
        imputer_mlm = TextImputer(
            model=seq2seq_model,
            tokenizer=seq2seq_tokenizer,
            text=INPUT_TEXT,
            model_type="seq2seq",
            target_label="positive",
            player_level="word",
            perturbation_type="mlm_infilling",
            mlm_num_samples=2,        # 小样本数，加快测试速度
            normalize_target_logprob=True,
        )

        full_score = imputer_mlm.full_prediction()
        n          = imputer_mlm.n_players
        coalition  = np.zeros(n, dtype=int)
        coalition[0] = 1

        val = imputer_mlm.value_function([coalition])

        print(f"  玩家数量       = {n}")
        print(f"  完整预测分数   = {full_score:.6f}")
        print(f"  联盟预测分数   = {val[0]:.6f}")

        _result(
            "MLM infilling 集成测试通过",
            np.isfinite(val[0]),
            f"分数={val[0]:.4f}",
        )

    except Exception as exc:
        _result("MLM infilling 集成测试", False, str(exc)[:100])


# ===========================================================================
# 程序入口
# ===========================================================================

if __name__ == "__main__":

    print("正在加载 FLAN-T5 tokenizer …")
    seq2seq_tokenizer = AutoTokenizer.from_pretrained(SEQ2SEQ_MODEL_NAME)

    print("正在加载 FLAN-T5 模型 …")
    seq2seq_model = AutoModelForSeq2SeqLM.from_pretrained(SEQ2SEQ_MODEL_NAME)
    seq2seq_model.eval()

    tests = [
        ("模型类型校验",        test_1_model_type_validation),
        ("单 token 目标",       test_2_single_token_target),
        ("多 token 目标",       test_3_multi_token_target),
        ("归一化",              test_4_normalization),
        ("Prompt 模板",         test_5_prompt_template),
        ("端到端集成",          test_6_end_to_end_integration),
    ]

    suite_passed = []
    suite_failed = []

    for test_name, test_fn in tests:
        try:
            test_fn(seq2seq_model, seq2seq_tokenizer)
            suite_passed.append(test_name)
        except Exception as exc:
            suite_failed.append((test_name, exc))
            print(f"\n  [崩溃]  {test_name}: {type(exc).__name__}: {exc}")

    # 最终汇总
    _header("测试套件汇总")
    print(f"  通过数量：{len(suite_passed)} / {len(tests)}\n")

    for name in suite_passed:
        print(f"  [通过]  {name}")

    if suite_failed:
        print()
        for name, exc in suite_failed:
            print(f"  [失败]  {name}  —  {type(exc).__name__}: {exc}")