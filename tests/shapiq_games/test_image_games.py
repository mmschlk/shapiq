"""The image classification game: regions, removal, the ResNet crop, and the inspection helpers."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

import shapiq
from shapiq_games import ImageClassifier
from shapiq_games.vision import grid_regions
from tests.shapiq_games.helpers import is_installed, mean_brightness_classifier

_IMAGE = np.random.default_rng(0).integers(0, 255, (40, 60, 3), dtype=np.uint8)
_GRID = grid_regions(40, 60, rows=2, cols=3)


class RecordingClassifier:
    """A brightness classifier that remembers the images it was shown."""

    def __init__(self) -> None:
        self.seen: list[np.ndarray] = []

    def __call__(self, images: np.ndarray) -> np.ndarray:
        self.seen.extend(np.asarray(images))
        return mean_brightness_classifier(images)


def test_grid_regions_number_cells_row_by_row() -> None:
    regions = grid_regions(5, 7, rows=2, cols=3)
    assert regions.shape == (5, 7)
    assert set(np.unique(regions)) == set(range(6))
    assert regions[0, 0] == 0
    assert regions[0, -1] == 2
    assert regions[-1, 0] == 3


def test_custom_regions_are_the_players() -> None:
    game = ImageClassifier(_IMAGE, mean_brightness_classifier, regions=_GRID, normalize=False)
    assert game.n_players == 6
    np.testing.assert_array_equal(game.regions, _GRID)
    with pytest.raises(ValueError, match="shape"):
        ImageClassifier(_IMAGE, mean_brightness_classifier, regions=_GRID[:10])
    with pytest.raises(ValueError, match="every label"):
        ImageClassifier(_IMAGE, mean_brightness_classifier, regions=_GRID * 2)


def test_masked_image_is_what_the_classifier_sees() -> None:
    classifier = RecordingClassifier()
    game = ImageClassifier(_IMAGE, classifier, regions=_GRID, normalize=False)
    coalition = np.array([1, 0, 1, 0, 0, 1], dtype=bool)
    classifier.seen.clear()
    game(coalition.reshape(1, -1))
    np.testing.assert_array_equal(classifier.seen[-1], game.masked_image(coalition))
    kept = coalition[_GRID]
    np.testing.assert_array_equal(game.masked_image(coalition)[kept], _IMAGE[kept])
    np.testing.assert_array_equal(game.masked_image(np.ones(6)), _IMAGE)


@pytest.mark.parametrize(
    ("fill", "expected"),
    [
        ("black", lambda image: np.zeros_like(image)),
        ("gray", lambda image: np.full_like(image, 128)),
        (
            "mean",
            lambda image: np.broadcast_to(image.reshape(-1, 3).mean(axis=0).round(), image.shape),
        ),
        ("image", lambda image: 255 - image),
    ],
)
def test_fill_decides_what_replaces_removed_regions(fill: str, expected: object) -> None:
    fill_value = 255 - _IMAGE if fill == "image" else fill
    game = ImageClassifier(_IMAGE, mean_brightness_classifier, regions=_GRID, fill=fill_value)
    np.testing.assert_array_equal(game.masked_image(np.zeros(6)), expected(_IMAGE))  # type: ignore[operator]


def test_blur_fill_and_invalid_fills() -> None:
    blurred = ImageClassifier(_IMAGE, mean_brightness_classifier, regions=_GRID, fill="blur")
    empty = blurred.masked_image(np.zeros(6)).astype(float)
    assert empty.std() < _IMAGE.astype(float).std()  # blurring removes the noise
    with pytest.raises(ValueError, match="fill must be"):
        ImageClassifier(_IMAGE, mean_brightness_classifier, regions=_GRID, fill="noise")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="same shape|image's shape"):
        ImageClassifier(_IMAGE, mean_brightness_classifier, regions=_GRID, fill=_IMAGE[:10])


def test_attribution_map_and_player_images() -> None:
    game = ImageClassifier(_IMAGE, mean_brightness_classifier, regions=_GRID)
    values = np.arange(6, dtype=float)
    heatmap = game.attribution_map(values)
    assert heatmap.shape == (40, 60)
    np.testing.assert_array_equal(heatmap, values[_GRID])
    exact = shapiq.ExactComputer(game)("SV", 1)
    np.testing.assert_allclose(game.attribution_map(exact), exact.get_n_order_values(1)[_GRID])
    with pytest.raises(ValueError, match="one per player"):
        game.attribution_map(values[:3])

    patches = game.player_images()
    assert len(patches) == 6
    assert all(isinstance(patch, Image.Image) for patch in patches)
    assert patches[0].size == (20, 20)  # (width, height) of the top-left grid cell


def test_float_images_in_the_unit_interval_are_scaled() -> None:
    game = ImageClassifier(_IMAGE / 255.0, mean_brightness_classifier, regions=_GRID)
    np.testing.assert_array_equal(game.image, _IMAGE)


def test_empty_coalitions_take_the_stored_empty_value() -> None:
    """v(empty) is exactly the normalization value, without evaluating the model again."""
    classifier = RecordingClassifier()
    game = ImageClassifier(_IMAGE, classifier, regions=_GRID)
    classifier.seen.clear()
    coalitions = np.zeros((3, 6), dtype=bool)
    coalitions[1, 2] = True
    values = game(coalitions)
    assert values[0] == values[2] == 0.0
    assert len(classifier.seen) == 1  # only the non-empty coalition


def test_torch_batches_are_padded_to_one_size() -> None:
    from shapiq_games.vision._batching import pad_batch

    rows = np.arange(6).reshape(3, 2)
    np.testing.assert_array_equal(pad_batch(rows, 5), [[0, 1], [2, 3], [4, 5], [4, 5], [4, 5]])
    assert pad_batch(rows, 3) is rows


def test_vision_transformer_regions_follow_the_patch_grid(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict] = []

    class FakeViT:
        def __init__(self, image: np.ndarray, n_players: int, **kwargs: object) -> None:
            calls.append(kwargs)
            self.image, self.n_players = image, n_players
            side = round(n_players**0.5)
            self.regions = grid_regions(*image.shape[:2], side, side)
            self.categories = [f"class {i}" for i in range(7)] + ["tabby cat"]

        def __call__(self, coalitions: np.ndarray) -> np.ndarray:  # class 7: the visible share
            probabilities = np.zeros((coalitions.shape[0], 8))
            probabilities[:, 7] = coalitions.mean(axis=1)
            return probabilities

        def classify(self, images: np.ndarray) -> np.ndarray:  # class 7: the share of black pixels
            black = (images == 0).all(axis=-1).mean(axis=(1, 2))
            return np.tile(black[:, None], (1, 8))

    import shapiq_games.vision.image_classifier as module

    monkeypatch.setattr(module, "ViTTokenModel", FakeViT)
    game = ImageClassifier(_IMAGE, "vit_9_patches")
    assert calls[-1]["mask_strategy"] == "mask"  # the vision transformers mask by default
    assert game.mask_strategy == "mask"
    assert game.class_name == "tabby cat"
    np.testing.assert_array_equal(game.regions, grid_regions(40, 60, 3, 3))
    shown = game.masked_image(np.eye(9, dtype=bool)[4])  # only the center patch
    assert np.all(shown[~(game.regions == 4)] == 128)
    np.testing.assert_array_equal(shown[game.regions == 4], _IMAGE[game.regions == 4])
    ImageClassifier(_IMAGE, "vit_9_patches", mask_strategy="remove")
    assert calls[-1]["mask_strategy"] == "remove"

    # image space: the model classifies the filled image
    filled = ImageClassifier(_IMAGE, "vit_9_patches", fill="black", normalize=False)
    assert filled.mask_strategy is None
    one = np.eye(9, dtype=bool)[4]
    assert filled(one[None])[0] == pytest.approx((filled.regions != 4).mean())  # the black share
    assert np.all(filled.masked_image(one)[filled.regions != 4] == 0)
    with pytest.raises(ValueError, match="not both"):
        ImageClassifier(_IMAGE, "vit_9_patches", fill="black", mask_strategy="mask")
    with pytest.raises(ValueError, match="its patches"):
        ImageClassifier(_IMAGE, "vit_9_patches", regions=grid_regions(40, 60, 3, 3))
    with pytest.raises(ValueError, match="mask_strategy applies"):
        ImageClassifier(_IMAGE, lambda images: images.mean(axis=(1, 2)), mask_strategy="mask")
    with pytest.raises(ValueError, match="mask_strategy must be"):  # before any model is loaded
        ImageClassifier(_IMAGE, "vit_9_patches", mask_strategy="drop")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="fill must be"):
        ImageClassifier(_IMAGE, "vit_9_patches", fill="white")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="out of range"):
        ImageClassifier(_IMAGE, "vit_9_patches", class_index=8)
    assert game.class_index == 7
    assert game.original_model_output == pytest.approx(1.0)  # the full image
    assert game.empty_value == 0.0


@pytest.mark.skipif(
    not (is_installed("torch") and is_installed("torchvision")),
    reason="torch and torchvision are not installed",
)
@pytest.mark.filterwarnings("error::UserWarning")  # e.g. torch on the read-only prepared image
def test_resnet_players_live_on_the_crop_the_model_sees(monkeypatch: pytest.MonkeyPatch) -> None:
    """ResNet-18 sees a 224 x 224 center crop; the game's image and regions are that crop."""
    import torchvision.models

    untrained = torchvision.models.resnet18
    monkeypatch.setattr(torchvision.models, "resnet18", lambda **_: untrained(weights=None))
    image = np.random.default_rng(1).integers(0, 255, (300, 500, 3), dtype=np.uint8)
    game = ImageClassifier(image, "resnet_18", regions=grid_regions(224, 224, 2, 2), fill="gray")
    assert game.image.shape == (224, 224, 3)
    assert game.n_players == 4
    assert game.class_name is not None
    values = game(np.eye(4, dtype=bool))
    assert values.shape == (4,)
    assert np.all(np.isfinite(values))
    # padded forward passes: a value does not depend on the batch it is evaluated in
    coalitions = np.random.default_rng(0).random((5, 4)) < 0.5
    batched = game(coalitions)
    np.testing.assert_array_equal(batched, [game(row[None])[0] for row in coalitions])


@pytest.mark.skipif(not is_installed("torch"), reason="torch is not installed")
def test_token_masking_keeps_every_token_and_its_position() -> None:
    """Absent players' tokens are masked in place; any coalition layout gives the same outputs."""
    import torch

    from shapiq_games.vision._token_removal import TokenMasker, token_remover

    # four tokens carrying their player's number, masked tokens carrying 100 (their position)
    patch_tokens = torch.tensor([[0.0], [1.0], [2.0], [2.0]])
    masked_tokens = torch.full((4, 1), 100.0)
    cls_token = torch.tensor([[[10.0]]])

    def encode(tokens: torch.Tensor) -> torch.Tensor:  # (sum of tokens, sequence length)
        return torch.cat([tokens.sum(dim=1), torch.full_like(tokens[:, 0], tokens.shape[1])], 1)

    masker = TokenMasker(
        torch, cls_token, patch_tokens, masked_tokens, np.array([0, 1, 2, 2]), encode, 3
    )
    coalitions = np.array([[0, 0, 0], [1, 0, 1], [0, 1, 0], [1, 1, 1], [0, 0, 1]], dtype=bool)
    outputs = masker(coalitions)
    np.testing.assert_array_equal(outputs[:, 0], [410, 114, 311, 15, 214])
    np.testing.assert_array_equal(outputs[:, 1], [5] * 5)  # the sequence never shrinks
    np.testing.assert_array_equal(masker(coalitions[::-1])[::-1], outputs)  # negative strides

    embeddings = torch.cat([cls_token, patch_tokens[None]], dim=1)
    masked = torch.cat([cls_token, masked_tokens[None]], dim=1)
    players = np.array([0, 1, 2, 2])
    dropper = token_remover("remove", torch, embeddings, masked, players, encode, 3)
    assert dropper(coalitions[:1])[0, 1] == 1  # only the class token is left
    assert isinstance(
        token_remover("mask", torch, embeddings, masked, players, encode, 3), TokenMasker
    )
    with pytest.raises(ValueError, match="mask_strategy"):
        token_remover("blank", torch, embeddings, masked, players, encode, 3)  # type: ignore[arg-type]


def test_token_players_are_rectangular_blocks_of_the_token_grid() -> None:
    from shapiq_games.vision._regions import pixel_regions, token_players

    players = token_players(16, 5, 4)  # DINOv2's 16 x 16 tokens in a 5 x 4 grid
    assert players.max() == 19
    np.testing.assert_array_equal(np.bincount(players.ravel()), [16] * 4 + [12] * 16)
    regions = pixel_regions(token_players(2, 2, 2), 3)
    np.testing.assert_array_equal(regions[:, 0], [0, 0, 0, 2, 2, 2])
    assert regions.shape == (6, 6)


@pytest.mark.skipif(not is_installed("torch"), reason="torch is not installed")
def test_token_dropping_encodes_only_the_present_tokens() -> None:
    """Absent players' tokens are dropped; the output of a coalition does not depend on its batch."""
    import torch

    from shapiq_games.vision._token_removal import TokenDropper

    # four tokens, each carrying its player's number; the class token carries 10
    patch_tokens = torch.tensor([[0.0], [1.0], [2.0], [2.0]])
    cls_token = torch.tensor([[[10.0]]])

    def encode(tokens: torch.Tensor) -> torch.Tensor:  # (sum of tokens, sequence length)
        return torch.cat([tokens.sum(dim=1), torch.full_like(tokens[:, 0], tokens.shape[1])], 1)

    dropper = TokenDropper(torch, cls_token, patch_tokens, np.array([0, 1, 2, 2]), encode, 4)
    coalitions = np.array([[0, 0, 0], [1, 0, 1], [0, 1, 0], [1, 1, 1], [0, 0, 1]], dtype=bool)
    outputs = dropper(coalitions)
    np.testing.assert_array_equal(outputs[:, 0], [10, 14, 11, 15, 14])
    np.testing.assert_array_equal(outputs[:, 1], [1, 4, 2, 5, 3])
    np.testing.assert_array_equal(dropper(coalitions[::-1])[::-1], outputs)


@pytest.mark.skipif(not is_installed("torch"), reason="torch is not installed")
def test_token_removal_without_a_class_token() -> None:
    """SigLIP has no class token: dropping every token leaves an empty sequence."""
    import torch

    from shapiq_games.vision._token_removal import token_remover

    embeddings = torch.tensor([[[0.0], [1.0], [2.0], [2.0]]])  # four patch tokens, no prefix
    masked = torch.full((1, 4, 1), 100.0)

    def encode_sum_and_length(tokens: torch.Tensor) -> torch.Tensor:  # works on empty sequences
        length = torch.full((tokens.shape[0], 1), float(tokens.shape[1]))
        return torch.cat([tokens.sum(dim=1), length], 1)

    players = np.array([0, 1, 2, 2])
    coalitions = np.array([[0, 0, 0], [1, 0, 1], [1, 1, 1]], dtype=bool)
    dropper = token_remover(
        "remove", torch, embeddings, masked, players, encode_sum_and_length, 2, n_prefix=0
    )
    np.testing.assert_array_equal(dropper(coalitions), [[0, 0], [4, 3], [5, 4]])
    masker = token_remover(
        "mask", torch, embeddings, masked, players, encode_sum_and_length, 2, n_prefix=0
    )
    np.testing.assert_array_equal(masker(coalitions), [[400, 4], [104, 4], [5, 4]])


def test_dinov2_models_explain_the_crop_they_see(monkeypatch: pytest.MonkeyPatch) -> None:
    from shapiq_games.vision._regions import pixel_regions, token_players

    calls: list[dict] = []

    class FakeDinoV2:
        def __init__(self, image: np.ndarray, n_players: int, **kwargs: object) -> None:
            calls.append({"n_players": n_players, **kwargs})
            self.image = np.full((224, 224, 3), 7, dtype=np.uint8)
            self.regions = pixel_regions(token_players(16, 5, 4), 14)
            self.n_players = n_players
            self.categories = ["other"] * 217 + ["English springer"]

        def __call__(self, coalitions: np.ndarray) -> np.ndarray:  # class 217: the visible share
            probabilities = np.zeros((coalitions.shape[0], 218))
            probabilities[:, 217] = coalitions.mean(axis=1)
            return probabilities

    import shapiq_games.vision.image_classifier as module

    monkeypatch.setattr(module, "DinoV2TokenModel", FakeDinoV2)
    game = ImageClassifier(_IMAGE, "dinov2_20_patches", revision="v1", batch_size=8)
    assert calls[0]["n_players"] == 20
    assert calls[0]["revision"] == "v1"
    assert game.mask_strategy == "remove"  # DINOv2 drops tokens, as in the paper
    assert game.n_players == 20
    assert game.image.shape == (224, 224, 3)  # the crop, not the original image
    assert game.class_name == "English springer"
    shown = game.masked_image(np.eye(20, dtype=bool)[0])
    assert np.all(shown[game.regions != 0] == 128)
    assert game(game.grand_coalition)[0] == pytest.approx(1.0)
    ImageClassifier(_IMAGE, "dinov2_20_patches", mask_strategy="remove")
    for removal in ({"mask_strategy": "mask"}, {"fill": "black"}):
        with pytest.raises(ValueError, match="only by dropping"):
            ImageClassifier(_IMAGE, "dinov2_20_patches", **removal)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="mask_strategy must be"):  # no longer stored silently
        ImageClassifier(_IMAGE, "dinov2_20_patches", mask_strategy="drop")  # type: ignore[arg-type]


class FakeClip:
    """CLIP on a 4 x 4 token grid: embeddings in 2-d, the first axis grows with the visible area."""

    def __init__(self, image: np.ndarray, grid: tuple[int, int], **_: object) -> None:
        from shapiq_games.vision._regions import pixel_regions, token_players

        self.image = np.zeros((8, 8, 3), dtype=np.uint8)
        self.regions = pixel_regions(token_players(4, *grid), 2)

    def image_embeddings(self, coalitions: np.ndarray) -> np.ndarray:
        share = np.asarray(coalitions, dtype=float).mean(axis=1)
        return np.stack([share, 1.0 - share], axis=1)

    def text_embeddings(self, texts: list[str]) -> np.ndarray:
        return np.array([[1.0, 0.0] if "dog" in text else [0.0, 1.0] for text in texts])

    def filled_embeddings(
        self, coalitions: np.ndarray, fill: np.ndarray
    ) -> np.ndarray:  # the share of pixels that are not filled
        share = np.asarray(coalitions, dtype=bool)[:, self.regions].mean(axis=(1, 2))
        return np.stack([share, 1.0 - share], axis=1)


def test_image_text_similarity_matches_a_text_or_the_zero_shot_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import shapiq_games.vision.image_text as module
    from shapiq_games import ImageTextSimilarity

    monkeypatch.setattr(module, "ImageTextTokenModel", FakeClip)
    monkeypatch.setattr(module, "imagenet_class_names", lambda: ["cat", "dog"])
    game = ImageTextSimilarity(_IMAGE, grid=(2, 2), normalize=False)
    assert (game.label, game.text) == ("dog", "a photo of a dog.")  # the full image is all "dog"
    assert game.n_players == 4
    assert game.original_model_output == pytest.approx(1.0)
    half = np.array([[True, True, False, False]])
    assert game(half)[0] == pytest.approx(0.5)  # the visible share
    assert game(game.empty_coalition)[0] == 0.0

    given = ImageTextSimilarity(_IMAGE, "a cat", grid=(2, 2))
    assert given.label is None
    assert given.text == "a cat"
    assert given(given.empty_coalition)[0] == 0.0  # centered at the empty image's similarity
    assert given(half)[0] == pytest.approx(-0.5)
    assert np.all(given.masked_image([1, 0, 0, 0])[given.regions != 0] == 128)
    assert given.attribution_map(np.arange(4.0)).shape == (8, 8)
    assert given.mask_strategy == "remove"

    filled = ImageTextSimilarity(_IMAGE, "a dog", grid=(2, 2), fill="gray", normalize=False)
    assert filled.mask_strategy is None
    assert filled(half)[0] == pytest.approx(0.5)  # the share of pixels CLIP still sees
    assert np.all(filled.masked_image([1, 0, 0, 0])[filled.regions != 0] == 128)
    with pytest.raises(ValueError, match="not both"):
        ImageTextSimilarity(_IMAGE, "a dog", grid=(2, 2), fill="gray", mask_strategy="mask")
