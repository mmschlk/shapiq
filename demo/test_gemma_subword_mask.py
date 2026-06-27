import numpy as np

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from shapiq.imputer.text_imputer import TextImputer


MODEL_NAME = "google/gemma-4-E2B-it"

TEXT = (

    "Antidisestablishmentarianism pseudopseudohypoparathyroidism."

)

TARGET_LABEL = "positive"


print("loading tokenizer")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

print(tokenizer.tokenize(TEXT))

print("loading model")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
)

print("\n" + "=" * 80)
print("GEMMA SUBWORD TEST")
print("=" * 80)

print("\nTokenizer Output")
print("-" * 40)

tokens = tokenizer.tokenize(TEXT)

print(tokens)

print("\nNumber of tokenizer tokens:")
print(len(tokens))

print("\n" + "=" * 80)
print("CREATE IMPUTER")
print("=" * 80)

imputer = TextImputer(
    model=model,
    tokenizer=tokenizer,
    text=TEXT,
    model_type="causal_lm",
    target_label=TARGET_LABEL,
    player_level="subword",
    perturbation_type="mask",
)

print("\nTarget Callable")
print("-" * 40)
print(type(imputer.target_callable))

print("\nPlayer Strategy")
print("-" * 40)
print(type(imputer.player_strategy))

print("\nPlayer Strategy Attributes")
print("-" * 40)
print(imputer.player_strategy.__dict__)

print("\nPlayers")
print("-" * 40)
print(f"n_players: {imputer.n_players}")

if hasattr(imputer.player_strategy, "tokens"):
    for i, token in enumerate(imputer.player_strategy.tokens):
        print(f"{i}: {repr(token)}")

print("\nFull Prediction")
print("-" * 40)

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
print("-" * 40)
print(coalitions)

print("\nValue Function")
print("-" * 40)

values = imputer.value_function(
    coalitions
)

print(values)

print("\nConsistency Check")
print("-" * 40)

print(
    "full_prediction == all_present:",
    np.isclose(
        full_pred,
        values[0],
        atol=1e-2,
    ),
)

print("\nSUCCESS")