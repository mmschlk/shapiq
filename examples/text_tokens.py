"""End-to-end: explain a token scorer with mask-token masking.

Run with:
    uv run python examples/text_tokens.py

A toy sentiment scorer over token ids: every token contributes its lexicon
value, and the negator "not" flips the sentiment of the token after it.
The sequence "this movie is not good" is explained through

    TokenMasker + scorer -> ModelMaskedPredictor -> MaskedGame -> Explainer

with token positions as players and the mask token as the baseline —
exactly how a real language model would be explained, with the tokenizer's
mask id in place of ``MASK``. The punchline: because "not" mirrors the
sentiment of "good", the Shapley value of "good" is exactly zero — its
positive sense and the negation cancel, so order-1 attributions hide the
word doing the semantic work. The pairwise Shapley interactions (SII)
recover it: the ("not", "good") pair carries the whole flip.
"""

import jax.numpy as jnp
import numpy as np

from shapiq import SII, SV, ExactExplainer, MaskedGame, ModelMaskedPredictor, TokenMasker

if __name__ == "__main__":
    VOCABULARY = {"[MASK]": 0, "this": 1, "movie": 2, "is": 3, "not": 4, "good": 5}
    LEXICON = np.zeros(len(VOCABULARY))
    LEXICON[VOCABULARY["good"]] = 1.5
    LEXICON[VOCABULARY["not"]] = -0.2
    NOT_ID = VOCABULARY["not"]

    def score(token_ids: np.ndarray) -> np.ndarray:
        """Sum lexicon values; "not" flips the sentiment of the next token."""
        ids = np.asarray(token_ids)
        values = LEXICON[ids]
        flips = np.zeros_like(values)
        flips[..., 1:] = np.where(ids[..., :-1] == NOT_ID, -2.0 * values[..., 1:], 0.0)
        return (values + flips).sum(axis=-1)

    sentence = ["this", "movie", "is", "not", "good"]
    token_ids = np.asarray([VOCABULARY[word] for word in sentence])
    print(f"sentence: {' '.join(sentence)!r}  ->  score {score(token_ids):+.2f}")

    # --- token positions become players; absent tokens become [MASK] ---
    masker = TokenMasker(inputs=token_ids, baseline=VOCABULARY["[MASK]"])
    game = MaskedGame(masked_predictor=ModelMaskedPredictor(masker=masker, model=score))

    def tidy(value: object) -> float:
        scalar = float(jnp.asarray(value))
        return 0.0 if abs(scalar) < 1e-6 else scalar

    # --- Shapley values per token position ---
    shapley = ExactExplainer(game, SV()).estimate().view
    print("\nShapley values (baseline: the fully masked sequence)")
    for position, word in enumerate(sentence):
        print(f"  {word:>6}: {tidy(shapley((position,))):+.3f}")
    total = sum(tidy(shapley((position,))) for position in range(len(sentence)))
    print(f"  {'sum':>6}: {total:+.3f}")
    print("  ('good' nets to zero: its sense and the negation cancel at order 1)")

    # --- pairwise interactions recover the negation ---
    interactions = ExactExplainer(game, SII(order=2)).estimate().view
    not_position, good_position = sentence.index("not"), sentence.index("good")
    negation = tidy(interactions((not_position, good_position)))
    print("\npairwise Shapley interactions (order 2)")
    print(f"  ('not', 'good'):   {negation:+.3f}  <- the negation lives on the pair")
    print(f"  ('movie', 'good'): {tidy(interactions((1, good_position))):+.3f}")

    # the pair flips twice the lexicon value of "good"
    expected = -2.0 * LEXICON[VOCABULARY["good"]]
    print(f"  expected flip: {expected:+.3f} | gap: {abs(negation - expected):.2e}")
