import numpy as np

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from shapiq.imputer.text_imputer import TextImputer


MODEL_NAME = "google/gemma-4-E2B-it"

TEXT = (
    "The movie was surprisingly good and very entertaining. "
    "I would definitely watch it again. "
    "The acting was excellent."
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
print("GEMMA SENTENCE TEST")
print("=" * 80)

imputer = TextImputer(
    model=model,
    tokenizer=tokenizer,
    text=TEXT,
    model_type="causal_lm",
    target_label=TARGET_LABEL,
    player_level="sentence",
    perturbation_type="mask",
)

print("\nTarget Callable")
print("----------------------------------------")
print(type(imputer.target_callable))

print("\nPlayer Strategy")
print("----------------------------------------")
print(type(imputer.player_strategy))

print("\nPlayer Strategy Attributes")
print("----------------------------------------")
print(imputer.player_strategy.__dict__)

print("\nPlayers")
print("----------------------------------------")
print(f"n_players: {imputer.n_players}")

if hasattr(imputer.player_strategy, "sentences"):

    for i, sentence in enumerate(
        imputer.player_strategy.sentences
    ):
        print(f"{i}: {repr(sentence)}")

print("\nFull Prediction")
print("----------------------------------------")

full_pred = imputer.full_prediction()

print(full_pred)

n = imputer.n_players

coalitions = np.array(
    [
        np.ones(n, dtype=bool),
        np.zeros(n, dtype=bool),
        [True, False, True],
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

print("\nDetailed Results")
print("----------------------------------------")

print(f"All present : {values[0]}")
print(f"All missing : {values[1]}")
print(f"Sentence 1+3 : {values[2]}")

print("\nSUCCESS")