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
    "Albert Einstein developed revolutionary theories in physics. "
    "He worked at Princeton University for many years. "
    "His discoveries influenced modern science."
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

    print_header("Sentence Players")

    print(f"n_players = {imputer.n_players}\n")

    players = (
        imputer.player_strategy.get_players()
    )

    for i, player in enumerate(players):
        print(f"[{i}] {player}")

    return players


def show_prediction(imputer):

    print_header("Full Prediction")

    score = imputer.full_prediction()

    print(f"Target Label : {TARGET_LABEL}")
    print(f"Prediction   : {score}")


def build_coalitions(players):

    n = len(players)

    examples = []

    #
    # Full coalition
    #
    examples.append(
        (
            "Keep Everything",
            [True] * n,
        )
    )

    #
    # Empty coalition
    #
    examples.append(
        (
            "Keep Nothing",
            [False] * n,
        )
    )

    #
    # First sentence
    #
    if n >= 1:

        coalition = [False] * n
        coalition[0] = True

        examples.append(
            (
                "Keep First Sentence",
                coalition,
            )
        )

    #
    # Second sentence
    #
    if n >= 2:

        coalition = [False] * n
        coalition[1] = True

        examples.append(
            (
                "Keep Second Sentence",
                coalition,
            )
        )

    #
    # Third sentence
    #
    if n >= 3:

        coalition = [False] * n
        coalition[2] = True

        examples.append(
            (
                "Keep Third Sentence",
                coalition,
            )
        )

    #
    # First + Third
    #
    if n >= 3:

        coalition = [False] * n
        coalition[0] = True
        coalition[2] = True

        examples.append(
            (
                "Keep First + Third",
                coalition,
            )
        )

    #
    # Remove Middle Sentence
    #
    if n >= 3:

        coalition = [True] * n
        coalition[1] = False

        examples.append(
            (
                "Remove Middle Sentence",
                coalition,
            )
        )

    return examples


def show_perturbation_examples(imputer):

    print_header("Perturbation Examples")

    players = (
        imputer.player_strategy.get_players()
    )

    examples = build_coalitions(players)

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
        f"Sentence + {perturbation_type}"
    )

    imputer = TextImputer(
        model=model,
        tokenizer=tokenizer,
        text=TEXT,
        model_type="encoder_classifier",
        target_label=TARGET_LABEL,
        player_level="sentence",
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
        f"Sentence + {perturbation_type}"
    )

    try:

        imputer = TextImputer(
            model=model,
            tokenizer=tokenizer,
            text=TEXT,
            model_type="encoder_classifier",
            target_label=TARGET_LABEL,
            player_level="sentence",
            perturbation_type=perturbation_type,
        )

        coalition = [False] * imputer.n_players

        imputer.coalition_to_text(
            coalition
        )

        print(
            "\nWARNING: Expected failure "
            "but execution succeeded."
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

        try:

            run_supported_demo(
                perturbation,
                model,
                tokenizer,
            )

        except Exception as e:

            print("\nFAILED")
            print(type(e).__name__)
            print(e)

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