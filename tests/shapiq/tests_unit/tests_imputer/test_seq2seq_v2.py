"""
Test suite for Seq2SeqCallable improvements.
Tests: prompt templates, normalization, batch processing, encoder reuse.
"""

import os
import time

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from shapiq.imputer.text_imputer import TextImputer
from shapiq.imputer.text_imputer import Seq2SeqCallable

MODEL_NAME = "google/flan-t5-small"

TEXT = "The movie was surprisingly good and very entertaining."

TEXTS = [
    "The movie was surprisingly good and very entertaining.",
    "This film was absolutely terrible and boring.",
    "The acting was decent but the plot was confusing.",
]

PERTURBATIONS = [
    "mask",
    "pad",
    "wordnet_neutral",
    "mlm_infilling",
]


def print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)



def print_subheader(title: str) -> None:
    print("\n" + "-" * 60)
    print(title)
    print("-" * 60)


# =============================================================================
# TEST 1: Prompt Template Comparison
# Verify the effect of different prompt templates on prediction scores.
# =============================================================================

def test_prompt_templates(model, tokenizer) -> None:
    print_header("TEST 1: Prompt Template Comparison")

    templates = {
        "No template (default)":
            "{text}",

        "SST2 format (seen during FLAN-T5 training)":
            "sst2 sentence: {text}",

        "Question-answering format":
            "Is the following review positive or negative?\n{text}\nAnswer:",

        "Sentiment analysis format":
            "Classify the sentiment of the following sentence as positive or negative.\n{text}\nSentiment:",
    }

    print(f"\nInput text  : {TEXT}")
    print(f"Target label: positive\n")

    results = {}

    for template_name, template in templates.items():
        callable_obj = Seq2SeqCallable(
            model=model,
            tokenizer=tokenizer,
            device="cpu",
            target_label="positive",
            prompt_template=template,
            normalize=True,
        )

        actual_prompt = callable_obj._build_prompt(TEXT)
        scores = callable_obj.predict([TEXT])
        score = scores[0]
        results[template_name] = score

        print(f"Template name : {template_name}")
        print(f"Actual prompt : {actual_prompt}")
        print(f"Score         : {score:.6f}")
        print()

    best = max(results, key=results.get)
    print(f"Best-scoring template: {best}  ({results[best]:.6f})")


# =============================================================================
# TEST 2: Normalization Effect
# Verify the impact of normalize=True/False on labels of different lengths.
# =============================================================================

def test_normalization(model, tokenizer) -> None:
    print_header("TEST 2: Normalization Effect")

    labels = {
        "yes":           "yes",
        "positive":      "positive",
        "very positive": "very positive",
    }

    print(f"Input text: {TEXT}\n")
    print(f"{'Label':<20} {'Token count':<14} {'Raw score':<22} {'Normalized score':<20}")
    print("-" * 76)

    for label_name, label in labels.items():
        token_ids = tokenizer.encode(label, add_special_tokens=False)
        n_tokens = len(token_ids)

        callable_no_norm = Seq2SeqCallable(
            model=model,
            tokenizer=tokenizer,
            device="cpu",
            target_label=label,
            normalize=False,
        )
        score_raw = callable_no_norm.predict([TEXT])[0]

        callable_norm = Seq2SeqCallable(
            model=model,
            tokenizer=tokenizer,
            device="cpu",
            target_label=label,
            normalize=True,
        )
        score_norm = callable_norm.predict([TEXT])[0]

        print(f"{label_name:<20} {n_tokens:<14} {score_raw:<22.6f} {score_norm:<20.6f}")

    print("\nNote:")
    print("  Raw score    : longer labels score lower due to accumulating log-probs across more tokens.")
    print("  Normalized   : scores are comparable across labels of different lengths.")


# =============================================================================
# TEST 3: Batch Processing Consistency and Speed
# Verify that batch results match single-item results, and measure speedup.
# =============================================================================

def test_batch_consistency(model, tokenizer) -> None:
    print_header("TEST 3: Batch Processing Consistency & Speed")

    callable_obj = Seq2SeqCallable(
        model=model,
        tokenizer=tokenizer,
        device="cpu",
        target_label="positive",
        normalize=True,
    )

    print(f"Number of test texts: {len(TEXTS)}\n")

    # Single-item processing
    t0 = time.time()
    scores_single = np.array([
        callable_obj.predict([text])[0]
        for text in TEXTS
    ])
    t_single = time.time() - t0

    # Batch processing
    t0 = time.time()
    scores_batch = callable_obj.predict(TEXTS)
    t_batch = time.time() - t0

    print(f"{'Text':<55} {'Single score':<16} {'Batch score':<16} {'Diff':<10}")
    print("-" * 97)

    all_close = True
    for text, s1, s2 in zip(TEXTS, scores_single, scores_batch):
        diff = abs(s1 - s2)
        if diff > 1e-4:
            all_close = False
        short_text = text[:50] + "..." if len(text) > 50 else text
        print(f"{short_text:<55} {s1:<16.6f} {s2:<16.6f} {diff:<10.2e}")

    print(f"\nSingle-item time : {t_single:.3f}s")
    print(f"Batch time       : {t_batch:.3f}s")
    print(f"Consistency      : {'PASSED' if all_close else 'FAILED — results differ'}")


# =============================================================================
# TEST 4: Encoder Reuse Verification
# Verify that the encoder is called only once per batch,
# not once per target token.
# =============================================================================

def test_encoder_reuse(model, tokenizer) -> None:
    print_header("TEST 4: Encoder Reuse Verification")

    multi_token_label = "very positive"
    token_ids = tokenizer.encode(multi_token_label, add_special_tokens=False)
    n_tokens = len(token_ids)

    print(f"Target label        : '{multi_token_label}'")
    print(f"Number of tokens    : {n_tokens}")
    print(f"Number of texts     : {len(TEXTS)}\n")

    encoder_call_count = {"count": 0}
    original_encoder_forward = model.encoder.forward

    def patched_encoder_forward(*args, **kwargs):
        encoder_call_count["count"] += 1
        return original_encoder_forward(*args, **kwargs)

    model.encoder.forward = patched_encoder_forward

    callable_obj = Seq2SeqCallable(
        model=model,
        tokenizer=tokenizer,
        device="cpu",
        target_label=multi_token_label,
        normalize=True,
    )

    encoder_call_count["count"] = 0
    _ = callable_obj.predict(TEXTS)
    actual_calls = encoder_call_count["count"]

    model.encoder.forward = original_encoder_forward

    expected_calls = 1
    naive_calls = n_tokens * len(TEXTS)

    print(f"Actual encoder calls   : {actual_calls}")
    print(f"Expected (with reuse)  : {expected_calls}")
    print(f"Naive impl would call  : {naive_calls}  (n_tokens x n_texts)")
    print(f"Calls saved            : {naive_calls - actual_calls}")
    print(f"Result                 : {'PASSED — encoder correctly reused' if actual_calls == expected_calls else 'FAILED — encoder not reused'}")


# =============================================================================
# TEST 5: TextImputer Integration
# Verify that Seq2SeqCallable works correctly inside the full shapiq pipeline,
# and that prompt_template and normalize are correctly passed through.
# =============================================================================

def test_imputer_integration(model, tokenizer) -> None:
    print_header("TEST 5: TextImputer Integration with Seq2SeqCallable")

    for perturbation_type in PERTURBATIONS:
        print_subheader(f"Perturbation: {perturbation_type}")

        try:
            imputer = TextImputer(
                model=model,
                tokenizer=tokenizer,
                text=TEXT,
                model_type="seq2seq",
                target_label="positive",
                prompt_template="sst2 sentence: {text}",
                player_level="word",
                perturbation_type=perturbation_type,
            )

            callable_type = type(imputer.target_callable).__name__
            print(f"target_callable type : {callable_type}")

            if hasattr(imputer.target_callable, "prompt_template"):
                print(f"prompt_template      : {imputer.target_callable.prompt_template}")

            if hasattr(imputer.target_callable, "normalize"):
                print(f"normalize            : {imputer.target_callable.normalize}")

            full_score = imputer.full_prediction()
            print(f"full_prediction score: {full_score:.6f}")

            n = imputer.n_players
            coalition = [False] * n
            coalition[4] = True

            perturbed = imputer.coalition_to_text(coalition)
            coalition_score = imputer.value_function([coalition])[0]

            print(f"Coalition text       : {perturbed}")
            print(f"Coalition score      : {coalition_score:.6f}")
            print(f"Status               : PASSED")

        except Exception as e:
            print(f"FAILED: {type(e).__name__}: {e}")


# =============================================================================
# TEST 6: Sentiment Direction Sanity Check
# Verify that the model assigns higher scores to positive texts
# and lower scores to negative texts.
# =============================================================================

def test_sentiment_direction(model, tokenizer) -> None:
    print_header("TEST 6: Sentiment Direction Sanity Check")

    positive_texts = [
        "This movie was absolutely wonderful and I loved every minute.",
        "An outstanding film with brilliant performances.",
    ]

    negative_texts = [
        "This movie was absolutely terrible and I hated every minute.",
        "A dreadful film with awful performances.",
    ]

    callable_pos = Seq2SeqCallable(
        model=model,
        tokenizer=tokenizer,
        device="mps",
        target_label="positive",
        prompt_template="sst2 sentence: {text}",
        normalize=True,
    )

    print(f"{'Text':<55} {'Score (positive)':<22} {'Expected'}")
    print("-" * 90)

    for text in positive_texts:
        score = callable_pos.predict([text])[0]
        short = text[:50] + "..." if len(text) > 50 else text
        print(f"{short:<55} {score:<22.6f} high")

    print()

    for text in negative_texts:
        score = callable_pos.predict([text])[0]
        short = text[:50] + "..." if len(text) > 50 else text
        print(f"{short:<55} {score:<22.6f} low")

    pos_scores = callable_pos.predict(positive_texts)
    neg_scores = callable_pos.predict(negative_texts)

    avg_pos = np.mean(pos_scores)
    avg_neg = np.mean(neg_scores)
    direction_correct = avg_pos > avg_neg

    print(f"\nAverage positive score : {avg_pos:.6f}")
    print(f"Average negative score : {avg_neg:.6f}")
    print(f"Direction check        : {'PASSED — positive > negative as expected' if direction_correct else 'FAILED — scores go in the wrong direction'}")


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Loading model...")
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model.eval()

    tests = [
        ("Prompt Templates",        test_prompt_templates),
        ("Normalization",           test_normalization),
        ("Batch Consistency",       test_batch_consistency),
        ("Encoder Reuse",           test_encoder_reuse),
        ("TextImputer Integration", test_imputer_integration),
        ("Sentiment Direction",     test_sentiment_direction),
    ]

    passed = []
    failed = []

    for test_name, test_fn in tests:
        try:
            test_fn(model, tokenizer)
            passed.append(test_name)
        except Exception as e:
            failed.append((test_name, e))
            print(f"\nCRASHED — {test_name}: {type(e).__name__}: {e}")

    print_header("TEST SUMMARY")
    print(f"Passed: {len(passed)} / {len(tests)}\n")

    for name in passed:
        print(f"  PASSED  {name}")

    if failed:
        print()
        for name, err in failed:
            print(f"  FAILED  {name}: {err}")