import inspect

from shapiq.imputer.text_imputer import (

    PadTokenPerturbation,

)

print(inspect.signature(PadTokenPerturbation.perturb))


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


def run_test(perturbation_type: str):
    print("=" * 80)
    print(f"Testing perturbation: {perturbation_type}")
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
        player_level="word",
        perturbation_type=perturbation_type,
    )

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


if __name__ == "__main__":

    perturbations = [
        "mask",
        "pad",
        "removal",
        "neutral",
        "wordnet_neutral",
    ]

    for perturbation in perturbations:
        try:
            run_test(perturbation)

        except Exception as e:
            print("\nFAILED")
            print(type(e).__name__)
            print(e)
            print()