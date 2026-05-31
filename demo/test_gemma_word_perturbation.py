import numpy as np

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
    "removal",
    "neutral",
    "wordnet_neutral",
]


print("loading tokenizer")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

print("loading model")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
)

print("\n")
print("=" * 80)
print("GEMMA WORD PERTURBATION TEST")
print("=" * 80)


for perturbation in PERTURBATIONS:

    print("\n")
    print("=" * 80)
    print(f"Testing: Word + {perturbation}")
    print("=" * 80)

    try:

        imputer = TextImputer(
            model=model,
            tokenizer=tokenizer,
            text=TEXT,
            model_type="causal_lm",
            target_label=TARGET_LABEL,
            player_level="word",
            perturbation_type=perturbation,
        )

        print("\nTarget Callable")
        print("----------------------------------------")
        print(type(imputer.target_callable))

        print("\nPlayers")
        print("----------------------------------------")
        print(f"n_players: {imputer.n_players}")

        print("\nFull Prediction")
        print("----------------------------------------")
        
        full_pred = imputer.full_prediction()

        print(full_pred)

        n = imputer.n_players

        coalitions = np.array(
            [
                np.ones(n, dtype=bool),
                np.zeros(n, dtype=bool),
                [i % 2 == 0 for i in range(n)],
            ]
        )

        print("\nCoalitions")
        print("----------------------------------------")
        print(coalitions)

        print("\nValue Function")
        print("----------------------------------------")

        values = imputer.value_function(
            coalitions
        )

        print(values)

        print("\nConsistency Check")
        print("----------------------------------------")

        print(
            "full_prediction == all_present:",
            np.isclose(
                full_pred,
                values[0],
                atol=1e-2,
            ),
        )

        print("\nSUCCESS")

    except Exception as e:

        print("\nFAILED")
        print(type(e).__name__)
        print(e)


print(tokenizer.mask_token)

print(tokenizer.mask_token_id)

print(tokenizer.pad_token)

print(tokenizer.pad_token_id)

print(tokenizer.eos_token)

print(tokenizer.eos_token_id)