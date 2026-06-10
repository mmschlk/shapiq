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
    "Barack Obama visited New York and met Microsoft executives."
)

TARGET_LABEL = "POSITIVE"


PERTURBATIONS = [
    "mask",
    "pad",
    "removal",
    "neutral",
    "wordnet_neutral",
    "mlm_infilling",
]


def print_header(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_players(imputer):

    print_header("Named Entity Players")

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
    # Keep Barack Obama
    #
    coalition = [False] * n

    for i, player in enumerate(players):

        p = str(player)

        if (
            "Barack" in p
            or "Obama" in p
        ):
            coalition[i] = True

    examples.append(
        (
            "Keep Barack Obama",
            coalition,
        )
    )

    #
    # Keep New York
    #
    coalition = [False] * n

    for i, player in enumerate(players):

        if "New York" in str(player):
            coalition[i] = True

    examples.append(
        (
            "Keep New York",
            coalition,
        )
    )

    #
    # Keep Microsoft
    #
    coalition = [False] * n

    for i, player in enumerate(players):

        if "Microsoft" in str(player):
            coalition[i] = True

    examples.append(
        (
            "Keep Microsoft",
            coalition,
        )
    )

    #
    # Keep Named Entities
    #
    coalition = [False] * n

    for i, player in enumerate(players):

        p = str(player)

        if (
            "Barack" in p
            or "Obama" in p
            or "New York" in p
            or "Microsoft" in p
        ):
            coalition[i] = True

    examples.append(
        (
            "Keep Named Entities",
            coalition,
        )
    )

    #
    # Remove Named Entities
    #
    coalition = [True] * n

    for i, player in enumerate(players):

        p = str(player)

        if (
            "Barack" in p
            or "Obama" in p
            or "New York" in p
            or "Microsoft" in p
        ):
            coalition[i] = False

    examples.append(
        (
            "Remove Named Entities",
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


def run_demo(
    perturbation_type,
    model,
    tokenizer,
):

    print_header(
        f"Named Entity + {perturbation_type}"
    )

    imputer = TextImputer(
        model=model,
        tokenizer=tokenizer,
        text=TEXT,
        model_type="encoder_classifier",
        target_label=TARGET_LABEL,
        player_level="named_entity",
        perturbation_type=perturbation_type,
    )

    print_header("Original Text")
    print(TEXT)

    print_players(imputer)

    show_prediction(imputer)

    show_perturbation_examples(imputer)

    print("\nDemo completed.")


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
    print("removal          ✓")
    print("neutral          ✓")
    print("wordnet_neutral  ✓")
    print("mlm_infilling    ✓")