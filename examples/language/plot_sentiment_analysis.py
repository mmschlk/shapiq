"""
Explaining Sentiment Analysis
==============================

This example shows how to explain a sentiment classifier using ``shapiq``.
Each token in the input text becomes a player in a cooperative game, and
Shapley values quantify each token's contribution to the predicted sentiment.

We use the :class:`~shapiq_games.benchmark.SentimentAnalysisLocalXAI` game
from the ``shapiq_games`` package, which wraps a pretrained DistilBERT model
fine-tuned on IMDb reviews.
"""

from __future__ import annotations

import os

# Prevent OpenMP/MKL thread conflicts with PyTorch backend
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import shapiq
from shapiq_games.benchmark import SentimentAnalysisLocalXAI

# %%
# Set Up the Sentiment Game
# --------------------------
# Each word in the input becomes a player. The game value for a coalition
# is the model's sentiment score when only those tokens are visible (absent
# tokens are replaced with ``[MASK]``). The score is normalized so that the
# empty coalition maps to 0.

game = SentimentAnalysisLocalXAI(
    input_text="I really loved this amazing film",
    mask_strategy="mask",
    normalize=True,
)
token_names = game.input_text.split()
print(f"Tokens (players): {token_names}")
print(f"Number of players: {game.n_players}")
print(f"Grand coalition (full-text sentiment): {game.grand_coalition_value:.3f}")

# %%
# Compute Shapley Values
# -----------------------

approx = shapiq.KernelSHAP(n=game.n_players, random_state=42)
sv = approx.approximate(budget=50, game=game)
print(sv)

# %%
# Force Plot
# ----------
# Shows how each token pushes the sentiment away from the baseline
# (all tokens masked).

sv.plot_force(feature_names=token_names)

# %%
# Second-Order Interactions
# --------------------------
# Pairwise interactions reveal which tokens amplify or dampen each
# other's effect on the sentiment.

approx_sii = shapiq.KernelSHAPIQ(
    n=game.n_players,
    index="k-SII",
    max_order=2,
    random_state=42,
)
sii = approx_sii.approximate(budget=50, game=game)

sii.plot_network(feature_names=token_names)

# %%
# Negative Sentiment Example
# ----------------------------
# Let's also explain a negative review.

game_neg = SentimentAnalysisLocalXAI(
    input_text="This movie was terrible and boring",
    mask_strategy="mask",
    normalize=True,
)
token_names_neg = game_neg.input_text.split()
print(f"Tokens: {token_names_neg}")
print(f"Sentiment: {game_neg.grand_coalition_value:.3f}")

sv_neg = shapiq.KernelSHAP(n=game_neg.n_players, random_state=42).approximate(
    budget=50,
    game=game_neg,
)
sv_neg.plot_force(feature_names=token_names_neg)

# %%
# Sentence Plot
# --------------
# The :func:`~shapiq.sentence_plot` colors each word by its first-order
# attribution, giving an inline text visualization.

sv.plot_sentence(words=token_names)

# %%

sv_neg.plot_sentence(words=token_names_neg)

# %%
# References
# ----------
# .. footbibliography::
