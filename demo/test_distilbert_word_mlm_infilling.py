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


def main():
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
        player_level="word",
        perturbation_type="mlm_infilling",
    )

    print("=" * 80)
    print("Word + MLMInfilling")
    print("=" * 80)

    print("\nPlayers")
    print("-" * 40)

    print(f"n_players: {imputer.n_players}")

    try:
        print(imputer.players)
    except Exception:
        pass

    print("\nFull Prediction")
    print("-" * 40)

    full_pred = imputer.full_prediction()

    print(full_pred)

    print("\nValue Function")
    print("-" * 40)

    n = imputer.n_players

    coalitions = [
        [True] * n,      # full coalition
        [False] * n,     # empty coalition
    ]

    if n >= 2:
        coalition = [False] * n
        coalition[0] = True
        coalition[1] = True
        coalitions.append(coalition)

    values = imputer.value_function(coalitions)

    print(values)

    print("\nSuccess")


if __name__ == "__main__":
    main()