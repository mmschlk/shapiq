from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

from shapiq.imputer.text_imputer import TextImputer


MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

TEXT = (
    "The movie was surprisingly good and very entertaining. "
    "I would definitely watch it again. "
    "The acting was excellent."
)

TARGET_LABEL = "POSITIVE"


def run_test(perturbation_type: str):
    print("=" * 80)
    print(f"Testing: Sentence + {perturbation_type}")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME
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

    print("\nPlayer Strategy")
    print("-" * 40)
    print(type(imputer.player_strategy))

    print("\nAttributes")
    print("-" * 40)
    print(vars(imputer.player_strategy))

    print("\nPlayers")
    print("-" * 40)
    print(f"n_players: {imputer.n_players}")

    if hasattr(imputer.player_strategy, "players"):
        for idx, player in enumerate(
            imputer.player_strategy.players
        ):
            print(f"{idx}: {repr(player)}")

    elif hasattr(imputer.player_strategy, "sentences"):
        for idx, player in enumerate(
            imputer.player_strategy.sentences
        ):
            print(f"{idx}: {repr(player)}")

    print("\nFull Prediction")
    print("-" * 40)

    full_pred = imputer.full_prediction()

    print(full_pred)

    print("\nValue Function")
    print("-" * 40)

    n = imputer.n_players

    coalitions = [
        [True] * n,
        [False] * n,
    ]

    if n >= 2:
        coalition = [False] * n
        coalition[0] = True
        coalition[1] = True
        coalitions.append(coalition)

    values = imputer.value_function(coalitions)

    print(values)

    print("\nSuccess")


def run_expected_failure():
    print("=" * 80)
    print("Testing: Sentence + MLMInfilling")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME
    )

    try:
        imputer = TextImputer(
            model=model,
            tokenizer=tokenizer,
            text=TEXT,
            model_type="encoder_classifier",
            target_label=TARGET_LABEL,
            player_level="sentence",
            perturbation_type="mlm_infilling",
        )

        n = imputer.n_players

        coalitions = [
            [True] * n,
            [False] * n,
        ]

        imputer.value_function(coalitions)

        print(
            "\nERROR: MLMInfilling unexpectedly succeeded "
            "for SentencePlayerStrategy"
        )

    except ValueError as e:
        print("\nExpected failure:")
        print(type(e).__name__)
        print(e)

    except Exception as e:
        print("\nUnexpected exception:")
        print(type(e).__name__)
        print(e)


if __name__ == "__main__":

    supported_perturbations = [
        "mask",
        "pad",
        "removal",
        "neutral",
        "wordnet_neutral",
    ]

    for perturbation in supported_perturbations:

        try:
            run_test(perturbation)

        except Exception as e:

            print("\nFAILED")
            print(type(e).__name__)
            print(e)

    print("\n")
    print("#" * 80)
    print("MLM VALIDATION")
    print("#" * 80)

    run_expected_failure()