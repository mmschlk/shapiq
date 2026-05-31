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


print("loading tokenizer")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

print("loading model")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
)

print("\n" + "=" * 80)
print("GEMMA MLM-INFILLING VALIDATION")
print("=" * 80)

try:

    imputer = TextImputer(
        model=model,
        tokenizer=tokenizer,
        text=TEXT,
        model_type="causal_lm",
        target_label=TARGET_LABEL,
        player_level="word",
        perturbation_type="mlm_infilling",
    )

    print("\nUNEXPECTED SUCCESS")
    print("----------------------------------------")

    print(type(imputer))

    print("\nFull Prediction")
    print("----------------------------------------")

    print(imputer.full_prediction())

except Exception as e:

    print("\nExpected failure:")
    print(type(e).__name__)
    print(e)


print(type(imputer.perturbation_strategy))
print(imputer.perturbation_strategy.__dict__)