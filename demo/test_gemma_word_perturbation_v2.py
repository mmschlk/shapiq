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
print("TOKENIZER SPECIAL TOKENS")
print("=" * 80)

print("mask_token:")
print(tokenizer.mask_token)

print("\nmask_token_id:")
print(tokenizer.mask_token_id)

print("\npad_token:")
print(tokenizer.pad_token)

print("\npad_token_id:")
print(tokenizer.pad_token_id)

print("\neos_token:")
print(tokenizer.eos_token)

print("\neos_token_id:")
print(tokenizer.eos_token_id)


print("\n" + "=" * 80)
print("CREATE IMPUTER")
print("=" * 80)

imputer = TextImputer(
    model=model,
    tokenizer=tokenizer,
    text=TEXT,
    model_type="causal_lm",
    target_label=TARGET_LABEL,
    player_level="word",
    perturbation_type="mask",
)

print("\nTarget Callable")
print(type(imputer.target_callable))

print("\nPerturbation Strategy")
print(type(imputer.perturbation_strategy))

print("\nPerturbation Attributes")
print(imputer.perturbation_strategy.__dict__)

print("\nPlayers")
print(imputer.n_players)


print("\n" + "=" * 80)
print("PLAYER INSPECTION")
print("=" * 80)

strategy = imputer.player_strategy

print(type(strategy))
print(strategy.__dict__)


print("\n" + "=" * 80)
print("MASKED TEXT EXAMPLES")
print("=" * 80)

players = strategy.get_players()

print("Original:")
print(players)

for idx in [0, 1, 2]:

    coalition = np.ones(len(players), dtype=bool)
    coalition[idx] = False

    masked_players = []

    for keep, player in zip(coalition, players):

        if keep:
            masked_players.append(player)
        else:
            masked_players.append(
                imputer.perturbation_strategy.perturb(
                    player
                )
            )

    print(f"\nMask player {idx}")
    print(masked_players)


print("\n" + "=" * 80)
print("FULL PREDICTION")
print("=" * 80)

print(imputer.full_prediction())