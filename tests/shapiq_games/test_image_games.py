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
        ("mean", lambda image: np.broadcast_to(image.reshape(-1, 3).mean(axis=0).round(), image.shape)),
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


def test_vision_transformer_regions_follow_the_patch_grid(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeViT:
        def __init__(self, image: np.ndarray, n_players: int, **_: object) -> None:
            self.n_players = n_players
            self.class_index = 7
            self.class_name = "tabby cat"
            self.model_commit = "abc"

        def __call__(self, coalitions: np.ndarray) -> np.ndarray:
            return coalitions.mean(axis=1)

    import shapiq_games.vision.image_classifier as module

    monkeypatch.setattr(module, "ViTPatchModel", FakeViT)
    game = ImageClassifier(_IMAGE, "vit_9_patches")
    assert game.class_name == "tabby cat"
    np.testing.assert_array_equal(game.regions, grid_regions(40, 60, 3, 3))
    shown = game.masked_image(np.eye(9, dtype=bool)[4])  # only the center patch
    assert np.all(shown[~(game.regions == 4)] == 128)
    np.testing.assert_array_equal(shown[game.regions == 4], _IMAGE[game.regions == 4])
    with pytest.raises(ValueError, match="regions and fill"):
        ImageClassifier(_IMAGE, "vit_9_patches", fill="black")


@pytest.mark.skipif(
    not (is_installed("torch") and is_installed("torchvision")),
    reason="torch and torchvision are not installed",
)
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
