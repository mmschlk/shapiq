"""End-to-end: explain a torch image model over grid superpixels.

Run with:
    uv run python examples/image_superpixels.py

A synthetic image carries a bright blob in the top-left superpixel of a 3x3
grid. The image is explained through the chunked torch pipeline

    SuperpixelMasker + model -> ChunkedMaskedPredictor -> MaskedGame -> Explainer

first with a transparent brightness model (the Shapley values must locate
the blob), then with a small CNN, including a batch-size throughput scan of
the chunked evaluation and a sampled cross-check against the exact values.
Devices follow the model: masked chunks move to the model's parameter
device automatically, so `cnn.to("cuda")` is the only change a GPU needs.
"""

import time
from itertools import combinations

import jax.numpy as jnp
import torch
from torch import nn

from shapiq import FSII, SV, DenseCoalitionArray, ExactExplainer, MaskedGame, Regression
from shapiq.games.torch import ChunkedMaskedPredictor, SuperpixelMasker, grid_labels, to_jax


def chunked_image_game(masker, model, batch_size, link_function=None) -> MaskedGame:
    predictor = ChunkedMaskedPredictor(masker=masker, model=model, batch_size=batch_size)
    # without a link the dispatched to_values conversion turns predictions into values
    return MaskedGame(masked_predictor=predictor, link_function=link_function)

if __name__ == "__main__":
    torch.manual_seed(0)
    CHANNELS, HEIGHT, WIDTH = 3, 27, 27
    GRID = (3, 3)
    N_PLAYERS = GRID[0] * GRID[1]

    # --- a synthetic image: dim everywhere, bright blob in the top-left superpixel ---
    image = torch.full((CHANNELS, HEIGHT, WIDTH), 0.1)
    image[:, 1:8, 1:8] = 1.0
    labels = grid_labels(HEIGHT, WIDTH, grid=GRID)
    masker = SuperpixelMasker(inputs=image, baseline=0.0, labels=labels)

    print("=== A: transparent model (mean brightness) ===")

    def brightness(flat_images: torch.Tensor) -> torch.Tensor:
        return flat_images.mean(dim=(-3, -2, -1))

    game = chunked_image_game(masker, brightness, batch_size=128)
    explanation = ExactExplainer(game, SV()).explain()
    values = explanation.attributions_by_order[1].reshape(GRID)
    print("Shapley values per superpixel (row-major 3x3 grid):")
    for row in values:
        print("  " + " ".join(f"{float(cell):+0.4f}" for cell in row))
    share = float(values[0, 0] / jnp.sum(values))
    print(f"the blob superpixel holds {share:.0%} of the summed Shapley values")

    print("\n=== B: small CNN, chunked throughput scan ===")
    cnn = nn.Sequential(
        nn.Conv2d(CHANNELS, 8, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(3),
        nn.Conv2d(8, 8, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(8, 2),
    )

    # train briefly to detect whether the blob sits in the top-left superpixel
    def blob_batch(n_images: int) -> tuple[torch.Tensor, torch.Tensor]:
        images = torch.full((n_images, CHANNELS, HEIGHT, WIDTH), 0.1)
        images += 0.05 * torch.randn_like(images)
        corners = torch.randint(0, 4, (n_images,))
        for index, corner in enumerate(corners):
            top = 1 if corner < 2 else HEIGHT - 8
            left = 1 if corner % 2 == 0 else WIDTH - 8
            images[index, :, top : top + 7, left : left + 7] = 1.0
        return images, (corners == 0).long()

    optimizer = torch.optim.Adam(cnn.parameters(), lr=5e-3)
    for _ in range(120):
        train_images, train_classes = blob_batch(64)
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(cnn(train_images), train_classes)
        loss.backward()
        optimizer.step()
    print(f"trained blob-corner CNN | final loss {float(loss.detach()):.3f}")

    def class_one_probability(predictions: torch.Tensor) -> jnp.ndarray:
        return to_jax(torch.softmax(predictions, dim=-1)[..., 1])

    def cnn_game(batch_size: int) -> MaskedGame:
        return chunked_image_game(masker, cnn, batch_size, link_function=class_one_probability)

    start = time.perf_counter()
    exact = ExactExplainer(cnn_game(256), SV()).explain()
    print(f"exact SV over all {2**N_PLAYERS} coalitions: {time.perf_counter() - start:.2f}s")

    # time the game directly on a fixed coalition array so the numbers show
    # pure chunked evaluation, not explainer solve overhead
    scan_rows = torch.rand((2048, N_PLAYERS), generator=torch.Generator().manual_seed(1)) < 0.5
    scan_coalitions = DenseCoalitionArray(jnp.asarray(scan_rows.numpy()))
    for batch_size in (8, 64, 512):
        game = cnn_game(batch_size)
        game(scan_coalitions)  # warmup
        start = time.perf_counter()
        game(scan_coalitions)
        duration = time.perf_counter() - start
        rate = scan_rows.shape[0] / duration
        print(f"  batch_size {batch_size:>4}: {duration:.3f}s ({rate:,.0f} masked images/s)")

    print("\n=== C: sampled faithful interactions on the CNN ===")
    exact_fsii = ExactExplainer(cnn_game(256), FSII(order=2)).explain()
    approximator = Regression(cnn_game(256), FSII(order=2), random_state=0, deduplicate=True)
    approximator = approximator.sample(approximator.min_budget + 120)
    estimate = approximator.explain()
    pairs = list(combinations(range(N_PLAYERS), 2))
    errors = jnp.stack([jnp.abs(estimate(pair) - exact_fsii(pair)) for pair in pairs])
    top = int(jnp.argmax(jnp.abs(jnp.stack([exact_fsii(pair) for pair in pairs]))))
    print(
        f"after {approximator.state.n_samples} stored evaluations | "
        f"max pair error {float(jnp.max(errors)):.4f} | "
        f"strongest exact pair {pairs[top]} at {float(exact_fsii(pairs[top])):+0.4f}"
    )
