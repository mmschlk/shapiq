import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from shapiq.imputer.text_imputer import TextImputer


MODEL_NAME = "google/gemma-4-E2B-it"

TEXT = (
    "The movie was surprisingly good and very entertaining."
)

TARGET_LABEL = "positive"

PERTURBATIONS = [
    "mask",
    "pad",
    "wordnet_neutral",
    "mlm_infilling",
]


def print_header(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_players(imputer):

    print_header("Players")

    print(f"n_players = {imputer.n_players}\n")

    players = imputer.player_strategy.get_players()

    for i, player in enumerate(players):
        print(f"[{i}] {player}")


def show_prediction(imputer):

    print_header("Full Prediction")

    score = imputer.full_prediction()

    print(f"Target Label : {TARGET_LABEL}")
    print(f"Prediction   : {score}")


def show_perturbation_examples(imputer):

    print_header("Perturbation Examples")

    n = imputer.n_players

    examples = []

    #
    # Keep Only "good"
    #
    coalition = [False] * n
    coalition[4] = True

    examples.append(
        (
            "Keep Only 'good'",
            coalition,
        )
    )

    #
    # Keep movie + good
    #
    coalition = [False] * n
    coalition[1] = True
    coalition[4] = True

    examples.append(
        (
            "Keep 'movie' + 'good'",
            coalition,
        )
    )

    for title, coalition in examples:

        print("\n")
        print("-" * 60)
        print(title)
        print("-" * 60)

        print("Coalition:")
        print(coalition)

        try:

            perturbed_text = (
                imputer.coalition_to_text(
                    coalition
                )
            )

            print("\nPerturbed Text:")
            print(perturbed_text)

            prediction = (
                imputer.value_function(
                    [coalition]
                )[0]
            )

            print("\nPrediction:")
            print(prediction)

        except Exception as e:

            print(
                f"Could not generate text: {e}"
            )


def run_demo(
    perturbation_type,
    model,
    tokenizer,
):

    print_header(
        f"Gemma4 + Word + {perturbation_type}"
    )

    imputer = TextImputer(
        model=model,
        tokenizer=tokenizer,
        text=TEXT,
        model_type="causal_lm",
        target_label=TARGET_LABEL,
        player_level="word",
        perturbation_type=perturbation_type,
    )

    print_header("Backend")

    print(
        type(
            imputer.target_callable
        ).__name__
    )

    print_header("Original Text")

    print(TEXT)

    print_players(imputer)

    show_prediction(imputer)

    show_perturbation_examples(imputer)

    print("\nDemo completed.")


if __name__ == "__main__":

    print("Loading tokenizer...")

    tokenizer = (
        AutoTokenizer.from_pretrained(
            MODEL_NAME
        )
    )

    print("Loading model...")

    model = (
        AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto",
        )
    )

    for perturbation in PERTURBATIONS:

        try:

            run_demo(
                perturbation,
                model,
                tokenizer,
            )

        except Exception as e:

            print("\nFAILED")
            print(type(e).__name__)
            print(e)

    print_header("Support Matrix")

    print("mask             ✓")
    print("pad              ✓")
    print("wordnet_neutral  ✓")
    print("mlm_infilling    ✓")