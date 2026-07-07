"""End-to-end: explain a torch model with baseline masking.

Run with:
    uv run python examples/torch_baseline.py

A small MLP is trained on synthetic tabular data whose label depends on an
interaction between the first two features. The trained model is explained
through the composed pipeline

    BaselineMasker + model -> ModelMaskedPredictor -> LinkFunction -> MaskedGame

first with a scalar link (the predicted probability of class 1), then with a
vector link (both class log-probabilities in one pass), each cross-checked
against the exact explainer on all ``2**n_players`` coalitions.
"""

from itertools import combinations

import jax.numpy as jnp
import torch
from jax import Array
from torch import nn

from shapiq import (
    FSII,
    SV,
    ExactExplainer,
    InsufficientSamplesError,
    MaskedGame,
    ModelMaskedPredictor,
    PermutationSampling,
    Regression,
)
from shapiq.games.torch import BaselineMasker, to_jax

if __name__ == "__main__":
    torch.manual_seed(0)
    N_PLAYERS = 8
    N_SAMPLES = 4000

    # --- synthetic tabular data: the label depends on x0 * x1 ---
    features = torch.randn(N_SAMPLES, N_PLAYERS)
    logits_true = 3.0 * features[:, 0] * features[:, 1] + 1.5 * features[:, 2] - features[:, 3]
    labels = (torch.sigmoid(logits_true) > torch.rand(N_SAMPLES)).long()

    model = nn.Sequential(
        nn.Linear(N_PLAYERS, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 2),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(300):
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(model(features), labels)
        loss.backward()
        optimizer.step()
    accuracy = float((model(features).argmax(-1) == labels).float().mean())
    print(f"trained MLP | final loss {float(loss):.3f} | train accuracy {accuracy:.3f}")

    # --- the pipeline: masker + model -> predictor -> link -> game ---
    instance = torch.tensor([1.4, 1.1, 0.8, -0.6, 0.2, -0.1, 0.4, -0.3])
    baseline = features.mean(dim=0)
    masker = BaselineMasker(inputs=instance, baseline=baseline)
    predictor = ModelMaskedPredictor(masker=masker, model=model)

    print("\n=== A: scalar link (predicted probability of class 1) ===")

    def probability_link(predictions: torch.Tensor) -> Array:
        return to_jax(torch.softmax(predictions, dim=-1)[..., 1])

    game = MaskedGame(masked_predictor=predictor, link_function=probability_link)
    exact = ExactExplainer(game, SV()).explain()
    exact_values = jnp.stack([exact((player,)) for player in range(N_PLAYERS)])
    print(f"exact SV ({2**N_PLAYERS} evaluations): {exact_values.round(3)}")

    approximator = PermutationSampling(game, SV(), random_state=0, track_history=True)
    payout = float(jnp.sum(exact_values))
    for budget in (9, 54, 700, 2000):
        approximator = approximator.sample(budget)
        estimate = jnp.stack(
            [approximator.explain()((player,)) for player in range(N_PLAYERS)],
        )
        print(
            f"after +{budget:>4} evals | stored: {approximator.state.n_samples:>4}"
            f" | pending: {approximator.sampler.n_pending_samples}"
            f" | max error: {jnp.max(jnp.abs(estimate - exact_values)):.4f}"
            f" | efficiency gap: {jnp.abs(jnp.sum(estimate) - payout):.2e}"
        )

    print("\n=== B: vector link (both class log-probabilities at once) ===")

    def log_probability_link(predictions: torch.Tensor) -> Array:
        return to_jax(torch.log_softmax(predictions, dim=-1))

    vector_game = MaskedGame(
        masked_predictor=predictor,
        link_function=log_probability_link,
        value_shape=(2,),
    )
    exact_fsii = ExactExplainer(vector_game, FSII(order=2)).explain()
    pairs = list(combinations(range(N_PLAYERS), 2))
    strengths = jnp.stack([exact_fsii(pair) for pair in pairs])  # (n_pairs, 2 classes)
    print("strongest exact pairwise interactions per class:")
    for class_index in (0, 1):
        top = int(jnp.argmax(jnp.abs(strengths[:, class_index])))
        print(f"  class {class_index}: {pairs[top]} with {float(strengths[top, class_index]):+.3f}")

    fsii = Regression(vector_game, FSII(order=2), random_state=0, deduplicate=True)
    print(f"min budget (identification): {fsii.min_budget}")
    for budget in (fsii.min_budget + 20, 60, 80):
        fsii = fsii.sample(budget)
        try:
            estimate = fsii.explain()
        except InsufficientSamplesError as error:
            print(f"after +{budget:>3} novel evals | {error}")
            continue
        errors = jnp.stack(
            [jnp.abs(estimate(pair) - exact_fsii(pair)) for pair in pairs],
        )
        interaction = estimate((0, 1))
        print(
            f"after +{budget:>3} novel evals | stored: {fsii.state.n_samples:>4}"
            f" | max pair error: {jnp.max(errors):.4f}"
            f" | (0, 1) per class: {interaction.round(3)}"
        )
    empty = exact_fsii.baseline
    totals = sum(
        (jnp.sum(exact_fsii.attributions_by_order[size], axis=-2) for size in (1, 2)),
        start=jnp.zeros(2),
    )
    grand = log_probability_link(model(instance))
    print(
        "per-class efficiency of the exact faithful fit "
        f"(sum of attributions vs v(N) - v(empty)): gap "
        f"{jnp.max(jnp.abs(totals - (grand - empty))):.2e}"
    )
