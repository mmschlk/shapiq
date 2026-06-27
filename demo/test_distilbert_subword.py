import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

from shapiq.imputer.text_imputer import TextImputer


MODEL_NAME = (
    "distilbert-base-uncased-finetuned-sst-2-english"
)

TEXT = (
    "The unbelievably extraordinary performance shocked everyone."
)

TARGET_LABEL = "POSITIVE"


SUPPORTED_PERTURBATIONS = [
    "mask",
    "pad",
    "removal",
    "neutral",
    "wordnet_neutral",
]

UNSUPPORTED_PERTURBATIONS = [
    "mlm_infilling",
]


def print_header(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_players(imputer):

    print_header("Subword Players")

    print(f"n_players = {imputer.n_players}\n")

    players = (
        imputer.player_strategy.get_players()
    )

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
    # Example 1
    #
    examples.append(
        (
            "Keep Everything",
            [True] * n,
        )
    )

    #
    # Example 2
    #
    examples.append(
        (
            "Keep Nothing",
            [False] * n,
        )
    )

    #
    # Example 3
    # Keep middle subwords
    #
    coalition = [False] * n

    if n >= 4:
        coalition[1] = True
        coalition[2] = True

    examples.append(
        (
            "Keep First Important Subwords",
            coalition,
        )
    )

    #
    # Example 4
    # Keep performance
    #
    coalition = [False] * n

    middle = n // 2
    coalition[middle] = True

    examples.append(
        (
            "Keep Middle Token",
            coalition,
        )
    )

    #
    # Example 5
    # Keep content words
    #
    coalition = [False] * n

    for idx in range(1, n, 2):
        coalition[idx] = True

    examples.append(
        (
            "Keep Alternate Tokens",
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


def run_supported_demo(
    perturbation_type,
    model,
    tokenizer,
):

    print_header(
        f"Subword + {perturbation_type}"
    )

    imputer = TextImputer(
        model=model,
        tokenizer=tokenizer,
        text=TEXT,
        model_type="encoder_classifier",
        target_label=TARGET_LABEL,
        player_level="subword",
        perturbation_type=perturbation_type,
    )

    print_header("Original Text")
    print(TEXT)

    print_players(imputer)

    show_prediction(imputer)

    show_perturbation_examples(imputer)

    print("\nDemo completed.")


def run_expected_failure(
    perturbation_type,
    model,
    tokenizer,
):

    print_header(
        f"Subword + {perturbation_type}"
    )

    try:

        TextImputer(
            model=model,
            tokenizer=tokenizer,
            text=TEXT,
            model_type="encoder_classifier",
            target_label=TARGET_LABEL,
            player_level="subword",
            perturbation_type=perturbation_type,
        )

        print(
            "\nWARNING: Expected failure but "
            "construction succeeded."
        )

    except Exception as e:

        print_header(
            "Expected Failure"
        )

        print(type(e).__name__)
        print(e)


if __name__ == "__main__":

    tokenizer = (
        AutoTokenizer.from_pretrained(
            MODEL_NAME
        )
    )

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(MODEL_NAME)
    )

    for perturbation in (
        SUPPORTED_PERTURBATIONS
    ):

        run_supported_demo(
            perturbation,
            model,
            tokenizer,
        )

    for perturbation in (
        UNSUPPORTED_PERTURBATIONS
    ):

        run_expected_failure(
            perturbation,
            model,
            tokenizer,
        )

    print_header("Support Matrix")

    print("mask             ✓")
    print("pad              ✓")
    print("removal          ✓")
    print("neutral          ✓")
    print("wordnet_neutral  ✓")
    print("mlm_infilling    ✗")