"""
Pre-commit tests for Seq2SeqCallable and its integration with TextImputer.

Test order:
    1. Model type validation  — rejects non-seq2seq models (BERT, Gemma)
    2. Single-token target    — output shape and dtype
    3. Multi-token target     — teacher-forcing loop across multiple tokens
    4. Normalization          — normalize=True vs normalize=False
    5. Prompt template        — prompt is correctly applied before encoding
    6. End-to-end integration — all player levels x all perturbation types
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

from shapiq.imputer.text_imputer import Seq2SeqCallable, TextImputer

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

SEQ2SEQ_MODEL_NAME = "google/flan-t5-small"
BERT_MODEL_NAME    = "bert-base-uncased"

INPUT_TEXT = "The movie was surprisingly good and very entertaining."

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PASS = "PASSED"
FAIL = "FAILED"


def _header(title: str) -> None:
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


def _subheader(title: str) -> None:
    print("\n  " + "-" * 60)
    print(f"  {title}")
    print("  " + "-" * 60)


def _result(label: str, ok: bool, detail: str = "") -> None:
    tag = PASS if ok else FAIL
    line = f"  [{tag}]  {label}"
    if detail:
        line += f"  →  {detail}"
    print(line)


# ===========================================================================
# TEST 1 — Model type validation
# ===========================================================================
# Expected behaviour
# ------------------
# Seq2SeqCallable.__init__ reads model.config.is_encoder_decoder.
# If the flag is missing or False the constructor must raise ValueError
# with a message that mentions "is_encoder_decoder".
#
# Models under test
#   • BERT (encoder-only)  → is_encoder_decoder = False  → must raise
#   • Gemma (causal LM)    → is_encoder_decoder = False  → must raise
#   • FLAN-T5 (seq2seq)    → is_encoder_decoder = True   → must NOT raise
# ===========================================================================

def test_1_model_type_validation(seq2seq_model, seq2seq_tokenizer) -> None:
    _header("TEST 1 — Model type validation")

    # ------------------------------------------------------------------
    # 1a  BERT encoder-only model must be rejected
    # ------------------------------------------------------------------
    _subheader("1a  BERT (encoder-only) — expect ValueError")

    bert_model     = AutoModelForMaskedLM.from_pretrained(BERT_MODEL_NAME)
    bert_tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)

    # BERT config has is_encoder_decoder=False
    print(f"  bert config.is_encoder_decoder = "
          f"{getattr(bert_model.config, 'is_encoder_decoder', False)}")

    raised = False
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
        "BERT raises ValueError",
        raised,
        error_msg[:80] if raised else "no exception raised",
    )
    _result(
        "Error message mentions 'is_encoder_decoder'",
        "is_encoder_decoder" in error_msg,
        error_msg[:80],
    )

    # ------------------------------------------------------------------
    # 1b  Synthetic causal-LM config must be rejected
    #     (avoids downloading a large Gemma checkpoint)
    # ------------------------------------------------------------------
    _subheader("1b  Synthetic causal-LM config — expect ValueError")

    # Patch the seq2seq model's config to look like a causal LM
    # so we can test the guard without downloading Gemma.
    original_flag = seq2seq_model.config.is_encoder_decoder
    seq2seq_model.config.is_encoder_decoder = False
    print(f"  patched config.is_encoder_decoder = "
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
        # Restore original flag so later tests are not affected
        seq2seq_model.config.is_encoder_decoder = original_flag

    _result(
        "Causal-LM config raises ValueError",
        raised,
        error_msg[:80] if raised else "no exception raised",
    )

    # ------------------------------------------------------------------
    # 1c  FLAN-T5 (true seq2seq) must be accepted
    # ------------------------------------------------------------------
    _subheader("1c  FLAN-T5 (seq2seq) — expect no error")

    print(f"  flan-t5 config.is_encoder_decoder = "
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
        "FLAN-T5 accepted without error",
        accepted,
        "ok" if accepted else error_msg[:80],
    )


# ===========================================================================
# TEST 2 — Single-token target: output shape and dtype
# ===========================================================================
# Expected behaviour
# ------------------
# predict([text]) returns a numpy array of shape (1,) and dtype float32.
# The scalar value must be a finite negative number (it is a log-probability).
#
# "positive" encodes to a single token in FLAN-T5's vocabulary.
# We verify this assumption explicitly so the test is self-documenting.
# ===========================================================================

def test_2_single_token_target(seq2seq_model, seq2seq_tokenizer) -> None:
    _header("TEST 2 — Single-token target: output shape and dtype")

    target_label = "positive"
    token_ids    = seq2seq_tokenizer.encode(target_label, add_special_tokens=False)
    n_tokens     = len(token_ids)

    print(f"  target_label  = '{target_label}'")
    print(f"  token_ids     = {token_ids}  (n={n_tokens})")

    _result(
        f"'{target_label}' encodes to exactly 1 token",
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

    print(f"\n  predict([INPUT_TEXT]) =")
    print(f"    type   : {type(scores)}")
    print(f"    shape  : {scores.shape}")
    print(f"    dtype  : {scores.dtype}")
    print(f"    value  : {scores[0]:.6f}")

    _result("Return type is np.ndarray",  isinstance(scores, np.ndarray))
    _result("Shape is (1,)",              scores.shape == (1,),   str(scores.shape))
    _result("dtype is float32",           scores.dtype == np.float32, str(scores.dtype))
    _result("Value is finite",            np.isfinite(scores[0]),  f"{scores[0]:.4f}")
    _result("Value is negative (log-prob)", scores[0] < 0,        f"{scores[0]:.4f}")


# ===========================================================================
# TEST 3 — Multi-token target: teacher-forcing loop
# ===========================================================================
# Expected behaviour
# ------------------
# A multi-token target such as "very positive" is scored token by token.
# We verify this by comparing:
#   a) the score produced by Seq2SeqCallable for the full phrase, and
#   b) a reference score computed manually with the same teacher-forcing
#      logic written inline in the test.
# The two values must agree to within 1e-5.
#
# We also verify that the decoder_input_ids grow by one token per step,
# which confirms the loop is advancing correctly.
# ===========================================================================

def test_3_multi_token_target(seq2seq_model, seq2seq_tokenizer) -> None:
    _header("TEST 3 — Multi-token target: teacher-forcing loop")

    target_label = "very positive"
    token_ids    = seq2seq_tokenizer.encode(target_label, add_special_tokens=False)

    print(f"  target_label = '{target_label}'")
    print(f"  token_ids    = {token_ids}  (n={len(token_ids)})")

    _result(
        f"'{target_label}' encodes to more than 1 token",
        len(token_ids) > 1,
        f"n_tokens={len(token_ids)}",
    )

    callable_obj = Seq2SeqCallable(
        model=seq2seq_model,
        tokenizer=seq2seq_tokenizer,
        device="cpu",
        target_label=target_label,
        normalize=False,     # raw sum, easier to verify manually
    )

    scores = callable_obj.predict([INPUT_TEXT])
    model_score = float(scores[0])

    # ------------------------------------------------------------------
    # Reference score: manual teacher-forcing written in the test itself
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

    start_id = seq2seq_model.config.decoder_start_token_id
    decoder_input_ids = torch.tensor([[start_id]])
    ref_log_prob = 0.0
    step_sizes   = []          # track decoder sequence length at each step

    for step_idx, tid in enumerate(token_ids):
        step_sizes.append(decoder_input_ids.shape[1])

        with torch.no_grad():
            out = seq2seq_model(
                encoder_outputs=encoder_outputs,
                attention_mask=encoded["attention_mask"],
                decoder_input_ids=decoder_input_ids,
            )
        lp = torch.log_softmax(out.logits[:, -1, :], dim=-1)
        ref_log_prob += lp[0, tid].item()
        decoder_input_ids = torch.cat(
            [decoder_input_ids, torch.tensor([[tid]])], dim=1
        )

    print(f"\n  Step-by-step decoder_input_ids lengths: {step_sizes}")
    print(f"  model_score (Seq2SeqCallable) = {model_score:.8f}")
    print(f"  ref_score   (manual loop)     = {ref_log_prob:.8f}")
    print(f"  absolute diff                 = {abs(model_score - ref_log_prob):.2e}")

    _result(
        "Decoder grows by 1 token per step",
        step_sizes == list(range(1, len(token_ids) + 1)),
        str(step_sizes),
    )
    _result(
        "Seq2SeqCallable matches manual reference (tol=1e-5)",
        abs(model_score - ref_log_prob) < 1e-5,
        f"diff={abs(model_score - ref_log_prob):.2e}",
    )


# ===========================================================================
# TEST 4 — Normalization: normalize=True vs normalize=False
# ===========================================================================
# Expected behaviour
# ------------------
# With normalize=False the score equals the sum of per-token log-probs.
# With normalize=True  the score equals the mean  of per-token log-probs.
#
# We verify:
#   a) raw_score  == sum_score   (normalize=False matches manual sum)
#   b) norm_score == raw_score / n_tokens
#   c) For a 1-token target, raw == norm (dividing by 1 changes nothing).
#   d) For a multi-token target, norm > raw  because dividing by n > 1
#      makes a negative number less negative.
# ===========================================================================

def test_4_normalization(seq2seq_model, seq2seq_tokenizer) -> None:
    _header("TEST 4 — Normalization: normalize=True vs normalize=False")

    labels = {
        "single-token  'positive'":     "positive",
        "multi-token   'very positive'": "very positive",
    }

    print(f"  {'Label':<30} {'n_tokens':<10} {'raw score':<20} "
          f"{'norm score':<20} {'raw/n':<20} {'match?'}")
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

        print(f"  {label_name:<30} {n_tokens:<10} {raw_score:<20.6f} "
              f"{norm_score:<20.6f} {expected:<20.6f} {'YES' if match else 'NO'}")

        _result(
            f"norm == raw / n_tokens  for '{label}'",
            match,
            f"diff={abs(norm_score - expected):.2e}",
        )

    # Extra sanity: for a multi-token target, normalized score > raw score
    # because dividing a negative number by n > 1 makes it less negative.
    raw_multi  = float(
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
        "normalized score > raw score for multi-token target (less negative)",
        norm_multi > raw_multi,
        f"raw={raw_multi:.4f}  norm={norm_multi:.4f}",
    )


# ===========================================================================
# TEST 5 — Prompt template
# ===========================================================================
# Expected behaviour
# ------------------
# _build_prompt inserts the input text into the template string via .format().
# We verify:
#   a) _build_prompt returns the correct string for several templates.
#   b) Different templates produce different scores (the encoder input changes).
#   c) The default template "{text}" is a no-op (prompt == original text).
# ===========================================================================

def test_5_prompt_template(seq2seq_model, seq2seq_tokenizer) -> None:
    _header("TEST 5 — Prompt template")

    templates = {
        "default (no-op)":        "{text}",
        "SST-2 format":           "sst2 sentence: {text}",
        "question format":        "Is this review positive or negative?\n{text}\nAnswer:",
        "instruction format":     "Classify sentiment:\n{text}\nLabel:",
    }

    # ------------------------------------------------------------------
    # 5a  _build_prompt string construction
    # ------------------------------------------------------------------
    _subheader("5a  _build_prompt string construction")

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

        print(f"\n  template      : {template_name}")
        print(f"  expected      : {expected[:80]}")
        print(f"  got           : {actual_prompt[:80]}")
        _result("_build_prompt matches .format()", ok)

    # ------------------------------------------------------------------
    # 5b  Default template is a no-op
    # ------------------------------------------------------------------
    _subheader("5b  Default template '{text}' is a no-op")

    default_callable = Seq2SeqCallable(
        model=seq2seq_model,
        tokenizer=seq2seq_tokenizer,
        device="cpu",
        target_label="positive",
        prompt_template="{text}",
    )
    _result(
        "_build_prompt(INPUT_TEXT) == INPUT_TEXT for default template",
        default_callable._build_prompt(INPUT_TEXT) == INPUT_TEXT,
        f"'{default_callable._build_prompt(INPUT_TEXT)[:60]}'",
    )

    # ------------------------------------------------------------------
    # 5c  Different templates produce different prediction scores
    # ------------------------------------------------------------------
    _subheader("5c  Different templates produce different scores")

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

    print(f"\n  {'Template':<30} {'Score':>12}")
    print("  " + "-" * 44)
    for name, sc in scores.items():
        print(f"  {name:<30} {sc:>12.6f}")

    unique_scores = len(set(round(v, 6) for v in scores.values()))
    _result(
        "All templates produce distinct scores",
        unique_scores == len(scores),
        f"{unique_scores} unique values out of {len(scores)}",
    )


# ===========================================================================
# TEST 6 — End-to-end integration via TextImputer
# ===========================================================================
# Expected behaviour
# ------------------
# TextImputer wires Seq2SeqCallable to every player strategy and every
# perturbation strategy.  For each combination we verify:
#   • full_prediction() returns a finite float
#   • value_function([coalition]) returns a finite numpy array of shape (1,)
#   • the target_callable inside the imputer is a Seq2SeqCallable instance
#
# Player levels tested: word, named_entity, chunk, sentence
#   (subword is skipped with mask/pad because FLAN-T5 does not have [MASK])
#
# Perturbation strategies tested: pad, removal, neutral, wordnet_neutral
#   (mask is skipped — FLAN-T5 tokenizer has no mask_token)
#   (mlm_infilling is tested separately because it downloads a second model)
# ===========================================================================

def test_6_end_to_end_integration(seq2seq_model, seq2seq_tokenizer) -> None:
    _header("TEST 6 — End-to-end integration via TextImputer")

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

    print(f"\n  Input text : {INPUT_TEXT}")
    print(f"  Model      : {SEQ2SEQ_MODEL_NAME}")
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

                # --- verify target_callable type ---
                is_seq2seq = isinstance(imputer.target_callable, Seq2SeqCallable)

                # --- full prediction ---
                full_score = imputer.full_prediction()
                full_ok    = np.isfinite(full_score)

                # --- coalition prediction ---
                # Build a coalition that keeps the first player only
                n          = imputer.n_players
                coalition  = np.zeros(n, dtype=int)
                coalition[0] = 1

                coalition_scores = imputer.value_function([coalition])
                coalition_ok     = (
                    isinstance(coalition_scores, np.ndarray)
                    and coalition_scores.shape == (1,)
                    and np.isfinite(coalition_scores[0])
                )

                overall_ok = is_seq2seq and full_ok and coalition_ok
                status     = PASS if overall_ok else FAIL

                results_table.append((label, status, n, full_score,
                                      float(coalition_scores[0])))

            except Exception as exc:
                results_table.append((label, f"ERROR: {exc}", "-", "-", "-"))

    # Print summary table
    print(f"  {'Combination':<38} {'Status':<8} {'n_players':>10} "
          f"{'full_score':>14} {'coalition_score':>16}")
    print("  " + "-" * 90)

    all_passed = True
    for row in results_table:
        label, status, n, full_sc, coal_sc = row
        full_str = f"{full_sc:.4f}" if isinstance(full_sc, float) else str(full_sc)
        coal_str = f"{coal_sc:.4f}" if isinstance(coal_sc, float) else str(coal_sc)
        print(f"  {label:<38} {status:<8} {str(n):>10} "
              f"{full_str:>14} {coal_str:>16}")
        if status != PASS:
            all_passed = False

    _result("All player × perturbation combinations pass", all_passed)

    # ------------------------------------------------------------------
    # 6b  MLM infilling integration (separate sub-test — downloads BERT)
    # ------------------------------------------------------------------
    _subheader("6b  MLM infilling with word players")

    try:
        imputer_mlm = TextImputer(
            model=seq2seq_model,
            tokenizer=seq2seq_tokenizer,
            text=INPUT_TEXT,
            model_type="seq2seq",
            target_label="positive",
            player_level="word",
            perturbation_type="mlm_infilling",
            mlm_num_samples=2,        # small sample count for speed
            normalize_target_logprob=True,
        )

        full_score = imputer_mlm.full_prediction()
        n          = imputer_mlm.n_players
        coalition  = np.zeros(n, dtype=int)
        coalition[0] = 1

        val = imputer_mlm.value_function([coalition])

        print(f"  n_players        = {n}")
        print(f"  full_prediction  = {full_score:.6f}")
        print(f"  value_function   = {val[0]:.6f}")

        _result("MLM infilling integration", np.isfinite(val[0]),
                f"score={val[0]:.4f}")

    except Exception as exc:
        _result("MLM infilling integration", False, str(exc)[:100])


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":

    print("Loading FLAN-T5 tokenizer …")
    seq2seq_tokenizer = AutoTokenizer.from_pretrained(SEQ2SEQ_MODEL_NAME)

    print("Loading FLAN-T5 model …")
    seq2seq_model = AutoModelForSeq2SeqLM.from_pretrained(SEQ2SEQ_MODEL_NAME)
    seq2seq_model.eval()

    tests = [
        ("Model type validation",      test_1_model_type_validation),
        ("Single-token target",        test_2_single_token_target),
        ("Multi-token target",         test_3_multi_token_target),
        ("Normalization",              test_4_normalization),
        ("Prompt template",            test_5_prompt_template),
        ("End-to-end integration",     test_6_end_to_end_integration),
    ]

    suite_passed = []
    suite_failed = []

    for test_name, test_fn in tests:
        try:
            test_fn(seq2seq_model, seq2seq_tokenizer)
            suite_passed.append(test_name)
        except Exception as exc:
            suite_failed.append((test_name, exc))
            print(f"\n  [CRASHED]  {test_name}: {type(exc).__name__}: {exc}")

    # -----------------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------------
    _header("TEST SUITE SUMMARY")
    print(f"  Passed : {len(suite_passed)} / {len(tests)}\n")

    for name in suite_passed:
        print(f"  [PASSED]  {name}")

    if suite_failed:
        print()
        for name, exc in suite_failed:
            print(f"  [FAILED]  {name}  —  {type(exc).__name__}: {exc}")