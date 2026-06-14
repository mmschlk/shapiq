import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

from shapiq.imputer.text_imputer import TextImputer


MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

TEXT = (
    "The movie was surprisingly good and very entertaining for everyone."
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
    print_header("MLM Infilling Examples")

    players = imputer.player_strategy.get_players()

    print("\nPlayer Index Mapping")
    print("-" * 40)

    for i, player in enumerate(players):
        print(f"[{i}] {player}")

    n = imputer.n_players

    examples = []

    #
    # Example 1
    # Only sentiment word
    #
    coalition = [False] * n
    coalition[4] = True

    examples.append(
        (
            "Only Sentiment Word",
            coalition,
        )
    )

    #
    # Example 2
    # movie + good
    #
    coalition = [False] * n
    coalition[1] = True
    coalition[4] = True

    examples.append(
        (
            "Movie + Good",
            coalition,
        )
    )

    #
    # Example 3
    # sentence skeleton
    #
    coalition = [False] * n

    for idx in [0, 1, 2, 4, 5, 7, 9, 10]:
        if idx < n:
            coalition[idx] = True

    examples.append(
        (
            "Sentence Skeleton",
            coalition,
        )
    )

    #
    # Example 4
    # important content words
    #
    coalition = [False] * n

    for idx in [1, 4, 7, 9]:
        if idx < n:
            coalition[idx] = True

    examples.append(
        (
            "Important Keywords",
            coalition,
        )
    )

    #
    # Example 5
    # almost complete sentence
    #
    coalition = [False] * n

    for idx in [0, 1, 2, 3, 4, 5, 7, 8, 9, 10]:
        if idx < n:
            coalition[idx] = True

    examples.append(
        (
            "Near Complete Sentence",
            coalition,
        )
    )

    for title, coalition in examples:

        print("\n")
        print("=" * 80)
        print(title)
        print("=" * 80)

        print("\nCoalition:")
        print(coalition)

        try:

            infilled_text = imputer.coalition_to_text(
                coalition
            )

            print("\nInfilled Text:")
            print(infilled_text)

            prediction = imputer.value_function(
                [coalition]
            )[0]

            print("\nPrediction:")
            print(prediction)

        except Exception as e:

            print(
                f"Could not generate infilling: {e}"
            )


def run_demo():

    print_header(
        "DistilBERT + Word Strategy + MLM Infilling"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    model = (
        AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME
        )
    )

    imputer = TextImputer(
        model=model,
        tokenizer=tokenizer,
        text=TEXT,
        model_type="encoder_classifier",
        target_label=TARGET_LABEL,
        player_level="word",
        perturbation_type="mlm_infilling",
    )

    print_header("Original Text")
    print(TEXT)

    print_players(imputer)

    show_prediction(imputer)

    show_perturbation_examples(imputer)

    print("\nDemo completed.")


if __name__ == "__main__":
    run_demo()