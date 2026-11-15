"""Play FIxLIP's pointing game on four photos of the demo, for the page's figure on it.

The pointing game is the paper's sanity check with a designed ground truth. Four photos are
combined into one image, two by two, and the text names what each of them shows. An interaction
between a name and a patch is correct if it is positive and the patch lies in the photo of the
name, or if it is negative and the patch lies in another photo. The pointing game recognition is
the share of the absolute interactions between names and patches that are correct.

The image and the text are explained as the photos of the demo are (explain.py): SigLIP 2, one
player per patch and per token, and ProxySHAP on 10,000 masks. The script writes the combined
image to ``../assets/pointing.jpg`` and the interactions between names and patches, with their
recognition, to ``../assets/figures-data.js`` under ``pointing``.

Usage::

    python pointing.py

The game takes about 10 minutes on 4 CPU cores.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import numpy as np
import torch
from explain import Network, SimilarityGame, caption_tokens, interactions, load_photo, r2
from pack import read_figures, write_figures
from PIL import Image

from shapiq_games.vision import _image_text_model as image_text

HERE = Path(__file__).resolve().parent
MODEL = "siglip2"
# the photos, row by row, and the name of what each of them shows
PHOTOS = {
    "elephant": "elephant",
    "giraffe-tree": "giraffe",
    "pineapple": "pineapple",
    "clock-tower": "clock",
}
SIDE = 448  # the pixels of one photo in the combined image
N_TEST = 512  # as explain.py


def combine(photos: list[dict[str, Any]], results: Path) -> Image.Image:
    """Return the four photos as one image, two by two."""
    image = Image.new("RGB", (2 * SIDE, 2 * SIDE))
    for k, photo in enumerate(photos):
        part = load_photo(photo, results).resize((SIDE, SIDE), Image.Resampling.LANCZOS)
        image.paste(part, (k % 2 * SIDE, k // 2 * SIDE))
    return image


def recognition(cross: np.ndarray, grid: int) -> float:
    """Return the pointing game recognition of the interactions of names (rows) and patches."""
    rows, columns = np.divmod(np.arange(grid * grid), grid)
    photo = 2 * (rows // (grid // 2)) + columns // (grid // 2)  # the photo that a patch lies in
    inside = photo[None, :] == np.arange(len(cross))[:, None]
    correct = np.where(inside, np.maximum(cross, 0), np.maximum(-cross, 0))
    return float(correct.sum() / np.abs(cross).sum())


def play(
    image: Image.Image, names: list[str], model: dict[str, Any], budget: int
) -> dict[str, Any]:
    """Explain the image and the text of the names, and score the explanation."""
    name = f"{model['id']}_demo"
    image_text.IMAGE_TEXT_MODEL_IDS[name] = model["checkpoint"]
    grid = model["grid"]
    embedder = image_text.ImageTextTokenModel(
        np.asarray(image), (grid, grid), model=name, mask_strategy="mask", batch_size=32
    )
    text = " ".join(names)
    tokens = [t["text"] for t in caption_tokens(Network(embedder), text)]
    if tokens != names:
        message = f"every name has to be one token of the model, but the tokens are {tokens}"
        raise ValueError(message)
    n_patches = grid * grid
    game = SimilarityGame(embedder, text, n_patches, len(tokens))
    first, pairs = interactions(game, budget)
    n = game.n_players
    test = np.random.default_rng(1).random((N_TEST, n)) < 0.5
    x = test.astype(float)
    y_hat = x @ first + 0.5 * np.einsum("ai,ij,aj->a", x, pairs, x)
    cross = pairs[n_patches:, :n_patches]
    return {
        "names": names,
        "similarity": round(float(game.similarity(np.ones((1, n), dtype=bool))[0]), 4),
        "r2": round(r2(game.similarity(test) - game.empty_value, y_hat), 3),
        "recognition": round(recognition(cross, grid), 3),
        "main": [round(float(value), 6) for value in first[n_patches:]],
        "interactions": [[round(float(value), 6) for value in row] for row in cross],
    }


def main() -> None:
    """Play the game and write it to ``figures-data.js``."""
    torch.set_num_threads(4)
    config = tomllib.loads((HERE / "config.toml").read_text(encoding="utf-8"))
    model = next(m for m in config["models"] if m["id"] == MODEL)
    results = HERE / "results"
    results.mkdir(exist_ok=True)
    photos = [next(p for p in config["photos"] if p["id"] == photo) for photo in PHOTOS]
    image = combine(photos, results)
    assets = HERE.parent / "assets"
    image.resize((SIDE, SIDE), Image.Resampling.LANCZOS).save(
        assets / "pointing.jpg", quality=86, optimize=True, progressive=True
    )
    game = play(image, list(PHOTOS.values()), model, config["budget"])
    print(f"recognition {game['recognition']}, R^2 {game['r2']}", flush=True)
    figures = read_figures()
    figures["pointing"] = {"model": MODEL, "grid": model["grid"], "photos": list(PHOTOS), **game}
    write_figures(figures)


if __name__ == "__main__":
    main()
