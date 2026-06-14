from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

from shapiq.imputer.text_imputer import TextImputer


MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

TEXT = (
    "The movie was surprisingly good and very entertaining."
)

TARGET_LABEL = "POSITIVE"


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

    examples = [
        ("Keep Everything", [True] * n),
        ("Keep Nothing", [False] * n),
    ]

    if n >= 5:
        coalition = [False] * n
        coalition[4] = True

        examples.append(
            ("Keep Only 'good'", coalition)
        )

    for title, coalition in examples:

        print(f"\n{title}")
        print("-" * 40)

        print("Coalition:")
        print(coalition)

        try:

            perturbed_text = imputer.coalition_to_text(
                coalition
            )

            print("\nPerturbed Text:")
            print(perturbed_text)

            prediction = imputer.value_function(
                [coalition]
            )[0]

            print("\nPrediction:")
            print(prediction)

        except Exception as e:

            print(
                f"Could not visualize perturbation: {e}"
            )


def run_demo(
    perturbation_type,
    model,
    tokenizer,
):
    print_header(
        f"Perturbation Strategy: {perturbation_type}"
    )

    imputer = TextImputer(
        model=model,
        tokenizer=tokenizer,
        text=TEXT,
        model_type="encoder_classifier",
        target_label=TARGET_LABEL,
        player_level="word",
        perturbation_type=perturbation_type,
    )

    print_header("Original Text")
    print(TEXT)

    print_players(imputer)

    show_prediction(imputer)

    show_perturbation_examples(imputer)

    print("\nDemo completed.")


if __name__ == "__main__":

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    model = (
        AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME
        )
    )

    perturbations = [
        "mask",
        "pad",
        "removal",
        "neutral",
        "wordnet_neutral",
    ]

    for perturbation in perturbations:

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