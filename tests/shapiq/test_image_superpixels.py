"""Tests for superpixel masking and chunked torch prediction."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from shapiq import (  # noqa: E402  # noqa: E402
    FSII,
    SV,
    BaselineMasker,
    DenseCoalitionArray,
    ExactExplainer,
    MaskedGame,
    ModelMaskedPredictor,
    Regression,
    SuperpixelMasker,
    grid_labels,
)
from shapiq.games.torch import ChunkedMaskedPredictor, to_jax  # noqa: E402

HEIGHT = WIDTH = 6
CHANNELS = 3
N_PLAYERS = 9


def image(seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(CHANNELS, HEIGHT, WIDTH, generator=generator)


def gray_masker(inputs):
    return SuperpixelMasker(
        inputs=inputs,
        baseline=torch.tensor(0.0),
        labels=grid_labels(HEIGHT, WIDTH),
    )


def chunked_game(masker, model, link_function=to_jax, batch_size=64, value_shape=(), device=None):
    predictor = ChunkedMaskedPredictor(
        masker=masker,
        model=model,
        batch_size=batch_size,
        device=device,
    )
    return MaskedGame(
        masked_predictor=predictor,
        link_function=link_function,
        value_shape=value_shape,
    )


def tiny_cnn(out_features=2):
    torch.manual_seed(1)
    return torch.nn.Sequential(
        torch.nn.Conv2d(CHANNELS, 4, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
        torch.nn.Linear(4, out_features),
    )


def coalitions(rows):
    return DenseCoalitionArray(jnp.asarray(rows, dtype=bool))


def random_coalitions(n_rows, seed=3):
    generator = torch.Generator().manual_seed(seed)
    rows = torch.rand((n_rows, N_PLAYERS), generator=generator) < 0.5
    return DenseCoalitionArray(jnp.asarray(rows.numpy()))


def test_grid_labels_partition_divisible_images_into_blocks():
    labels = grid_labels(6, 6, grid=(3, 3))
    for row_band in range(3):
        for column_band in range(3):
            block = labels[2 * row_band : 2 * row_band + 2, 2 * column_band : 2 * column_band + 2]
            assert np.all(block == row_band * 3 + column_band)


def test_grid_labels_bucket_non_divisible_axes():
    labels = grid_labels(7, 5, grid=(3, 3))
    assert np.array_equal(labels[:, 0] // 3 * 3, labels[:, 0])  # first column holds row bands
    assert labels[0, 0] == 0
    assert labels[6, 4] == 8
    row_band_sizes = [int((labels[:, 0] == band * 3).sum()) for band in range(3)]
    column_band_sizes = [int((labels[0, :] == band).sum()) for band in range(3)]
    assert row_band_sizes == [3, 2, 2]
    assert column_band_sizes == [2, 2, 1]


def test_grid_labels_validate_the_fit():
    with pytest.raises(ValueError, match="does not fit"):
        grid_labels(2, 2, grid=(3, 3))


def test_masker_keeps_present_superpixels_and_replaces_absent():
    inputs = image()
    masked = gray_masker(inputs)(coalitions([[player == 4 for player in range(N_PLAYERS)]]))
    assert masked.shape == (1, CHANNELS, HEIGHT, WIDTH)
    center = masked[0, :, 2:4, 2:4]
    assert torch.equal(center, inputs[:, 2:4, 2:4])
    without_center = masked[0].clone()
    without_center[:, 2:4, 2:4] = 0.0
    assert torch.equal(without_center, torch.zeros_like(without_center))


def test_masker_broadcasts_baseline_forms():
    inputs = image()
    empty = coalitions([[False] * N_PLAYERS])
    per_channel = torch.tensor([1.0, 2.0, 3.0]).reshape(CHANNELS, 1, 1)
    masked = SuperpixelMasker(
        inputs=inputs,
        baseline=per_channel,
        labels=grid_labels(HEIGHT, WIDTH),
    )(empty)
    assert torch.equal(masked[0], per_channel.expand(CHANNELS, HEIGHT, WIDTH))
    baseline_image = image(seed=7)
    masked = SuperpixelMasker(
        inputs=inputs,
        baseline=baseline_image,
        labels=grid_labels(HEIGHT, WIDTH),
    )(empty)
    assert torch.equal(masked[0], baseline_image)


def test_masker_accepts_float_baselines():
    masked = SuperpixelMasker(inputs=image(), baseline=0.5, labels=grid_labels(HEIGHT, WIDTH))(
        coalitions([[False] * N_PLAYERS]),
    )
    assert torch.equal(masked[0], torch.full((CHANNELS, HEIGHT, WIDTH), 0.5))


def test_masker_explains_image_batches():
    batch = torch.stack([image(), 2 * image()])
    masker = SuperpixelMasker(
        inputs=batch,
        baseline=torch.tensor(0.0),
        labels=grid_labels(HEIGHT, WIDTH),
    )
    assert masker.target_shape == (2,)
    masked = masker(coalitions([[True] * N_PLAYERS]))
    assert masked.shape == (2, 1, CHANNELS, HEIGHT, WIDTH)
    assert torch.equal(masked[:, 0], batch)


def test_masker_validates_metadata():
    inputs = image()
    labels = grid_labels(HEIGHT, WIDTH)
    with pytest.raises(ValueError, match="channel-first"):
        SuperpixelMasker(inputs=torch.zeros(4, 4), baseline=0.0, labels=labels)
    with pytest.raises(ValueError, match="one superpixel per pixel"):
        SuperpixelMasker(inputs=inputs, baseline=0.0, labels=labels[:, :3])
    with pytest.raises(ValueError, match="channels-last"):
        SuperpixelMasker(inputs=inputs.permute(1, 2, 0), baseline=0.0, labels=labels)
    with pytest.raises(ValueError, match="integer superpixel ids"):
        SuperpixelMasker(inputs=inputs, baseline=0.0, labels=labels.astype(float))
    with pytest.raises(ValueError, match="integer superpixel ids"):
        SuperpixelMasker(inputs=inputs, baseline=0.0, labels=labels - 1)
    with pytest.raises(ValueError, match="no gaps"):
        SuperpixelMasker(inputs=inputs, baseline=0.0, labels=labels * 2)
    with pytest.raises(ValueError, match="does not broadcast"):
        SuperpixelMasker(inputs=inputs, baseline=torch.zeros(2), labels=labels)


class _CountingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.flat_batch_sizes = []

    def forward(self, flat):
        self.flat_batch_sizes.append(int(flat.shape[0]))
        return flat.mean(dim=(-3, -2, -1))


def test_predictor_streams_coalitions_in_batch_size_chunks():
    model = _CountingModel()
    game = chunked_game(gray_masker(image()), model, batch_size=4)
    values = game(random_coalitions(10))
    assert values.shape == (10,)
    assert model.flat_batch_sizes == [4, 4, 2]


def test_chunked_values_match_the_single_batch():
    def make(batch_size):
        return chunked_game(
            gray_masker(image()),
            tiny_cnn(1),
            link_function=lambda predictions: to_jax(predictions[..., 0]),
            batch_size=batch_size,
        )

    rows = random_coalitions(21)
    assert jnp.allclose(make(4)(rows), make(1000)(rows), atol=1e-6)


def test_chunked_predictor_matches_the_one_shot_composition():
    inputs = image()
    cnn = tiny_cnn(2)

    def flat_model(masked):
        flat = masked.reshape(-1, *masked.shape[-3:])
        return cnn(flat).reshape(*masked.shape[:-3], 2)

    composed = MaskedGame(
        masked_predictor=ModelMaskedPredictor(masker=gray_masker(inputs), model=flat_model),
        link_function=to_jax,
        value_shape=(2,),
    )
    chunked = chunked_game(gray_masker(inputs), cnn, batch_size=5, value_shape=(2,))
    rows = random_coalitions(17)
    with torch.no_grad():
        expected = composed(rows)
    assert jnp.allclose(chunked(rows), expected, atol=1e-6)


def test_predictor_explains_target_batches_within_batch_size():
    batch = torch.stack([image(), image(seed=5)])
    masker = SuperpixelMasker(
        inputs=batch,
        baseline=torch.tensor(0.0),
        labels=grid_labels(HEIGHT, WIDTH),
    )
    model = _CountingModel()
    game = chunked_game(masker, model, batch_size=8)
    values = game(random_coalitions(11))
    assert values.shape == (2, 11)
    # two target images divide the coalition samples per chunk: 4 + 4 + 3
    assert model.flat_batch_sizes == [8, 8, 6]


def test_predictor_evaluates_scalar_and_empty_coalition_arrays():
    model = _CountingModel()
    game = chunked_game(gray_masker(image()), model, batch_size=4)
    single = game(DenseCoalitionArray(jnp.zeros(N_PLAYERS, dtype=bool)))
    assert single.shape == (1,)
    empty = game(coalitions(jnp.zeros((0, N_PLAYERS), dtype=bool)))
    assert empty.shape == (0,)
    assert model.flat_batch_sizes == [1, 0]  # models must accept empty batches


def test_predictor_validates_the_evaluation_policy():
    with pytest.raises(ValueError, match="batch_size"):
        ChunkedMaskedPredictor(masker=gray_masker(image()), model=_CountingModel(), batch_size=0)
    with pytest.raises(ValueError, match="instance_axes"):
        ChunkedMaskedPredictor(
            masker=gray_masker(image()),
            model=_CountingModel(),
            instance_axes=0,
        )


def test_tabular_models_chunk_with_one_instance_axis():
    inputs = torch.tensor([1.0, -2.0, 3.0, 0.5])
    weight = torch.tensor([[1.0, -0.5], [0.0, 2.0], [-1.0, 1.0], [0.5, 0.5]])
    masker = BaselineMasker(inputs=inputs, baseline=torch.zeros(4))
    one_shot = MaskedGame(
        masked_predictor=ModelMaskedPredictor(masker=masker, model=lambda x: x @ weight),
        link_function=to_jax,
        value_shape=(2,),
    )
    chunked = MaskedGame(
        masked_predictor=ChunkedMaskedPredictor(
            masker=masker,
            model=lambda x: x @ weight,
            batch_size=3,
            instance_axes=1,
        ),
        link_function=to_jax,
        value_shape=(2,),
    )
    generator = torch.Generator().manual_seed(2)
    rows = DenseCoalitionArray(
        jnp.asarray((torch.rand((9, 4), generator=generator) < 0.5).numpy()),
    )
    assert jnp.allclose(chunked(rows), one_shot(rows), atol=1e-6)


def test_explicit_devices_override_the_model_inference():
    game = chunked_game(gray_masker(image()), _CountingModel(), batch_size=4, device="cpu")
    values = game(random_coalitions(6))
    assert values.shape == (6,)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="needs an mps device to test cross-device streaming",
)
def test_masked_chunks_follow_the_models_device():
    cpu_model = tiny_cnn(1)
    mps_model = tiny_cnn(1).to("mps")

    def link(predictions):
        return to_jax(predictions[..., 0])

    rows = random_coalitions(13)
    on_cpu = chunked_game(gray_masker(image()), cpu_model, link_function=link, batch_size=4)
    on_mps = chunked_game(gray_masker(image()), mps_model, link_function=link, batch_size=4)
    assert jnp.allclose(on_mps(rows), on_cpu(rows), atol=1e-4)


def test_exact_shapley_values_are_efficient_on_the_image_game():
    game = chunked_game(
        gray_masker(image()),
        tiny_cnn(1),
        link_function=lambda predictions: to_jax(predictions[..., 0]),
        batch_size=128,
    )
    explanation = ExactExplainer(game, SV()).estimate().view
    attributions = jnp.stack([explanation((player,)) for player in range(N_PLAYERS)])
    ends = game(coalitions([[False] * N_PLAYERS, [True] * N_PLAYERS]))
    assert jnp.allclose(jnp.sum(attributions), ends[1] - ends[0], atol=1e-4)


def test_sampled_faithful_interactions_run_on_the_image_game():
    game = chunked_game(gray_masker(image()), tiny_cnn(2), batch_size=64, value_shape=(2,))
    approximator = Regression(game, FSII(order=2), random_state=0, deduplicate=True)
    explanation = approximator.estimate(approximator.min_budget + 40)
    assert explanation.index == FSII(order=2)
    assert explanation[(0, 1)].shape == (2,)
