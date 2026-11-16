"""Evaluate SigLIP 2 on the masks that the figures of the demo's page show.

The figures of the page (``../assets/figures.js``) explain the demo step by step on the photo
chosen in the demo. Most of them only read the demo's data. Some show what the model itself
predicts for a masked photo and caption, and these predictions are computed here, once for every
photo. They are written to ``../assets/figures-data.js`` under ``images``.

- ``presets``: the similarity of masks chosen by hand (the whole pair, the empty pair, one
  modality, and the two objects of config.toml with the words that name them).
- ``together``: the similarity with and without a patch and a word, on average over random masks
  of the other players. The four averages give the pair's Banzhaf interaction. The first pair is
  the strongest interaction of the explanation between a patch and a word. The second pair is the
  same patch with the word of the largest main effect.
- ``holdout``: the similarity of 512 random masks for each of p = 0.3, 0.5 and 0.7, with the R^2
  of the demo's explanation and of its first-order conversion. For p = 0.5, it also holds what
  both predict for every mask.
- ``examples``: one random mask for each p.
- ``curves`` and ``subsets``: the similarity of the subsets that the explanation values the most
  and the least, for every size, which gives FIxLIP's insertion and deletion curves. The search
  for the subsets is the demo's greedy search for cliques.
- ``cliques``: the similarity of the demo's cliques, shown alone.

Usage::

    python evaluate.py [photo ...]

Without arguments, all photos of config.toml are evaluated. The model takes about 4 minutes for
each photo on 4 CPU cores.
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
import torch
from explain import Network, SimilarityGame, caption_tokens, greedy_cliques, load_photo, r2
from pack import read_data, read_figures, write_figures
from scipy.stats import spearmanr

from shapiq_games.vision import _image_text_model as image_text

HERE = Path(__file__).resolve().parent
MODEL = "siglip2"
N_TEST = 512  # as explain.py
N_CONTEXTS = 256  # the random masks that an interaction is averaged over
INSERTED, DELETED = 10, 5  # the sizes of the two subsets that the page shows


def explanation(data: dict[str, Any], photo: str) -> tuple[np.ndarray, np.ndarray]:
    """Return the main effects and the matrix of interactions of the demo's explanation."""
    e = data["explanations"][f"{photo}/{MODEL}"]
    first = np.array(e["first"])
    n = len(first)
    pairs = np.zeros((n, n))
    pairs[np.triu_indices(n, 1)] = e["pairs"]
    return first, pairs + pairs.T


def predict(masks: np.ndarray, first: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """Return the explanation's value of every mask, without its constant."""
    x = masks.astype(float)
    return x @ first + 0.5 * np.einsum("ai,ij,aj->a", x, pairs, x)


def as_masks(subsets: list[list[int]], n: int) -> np.ndarray:
    """Return the subsets of players as rows of a boolean matrix."""
    masks = np.zeros((len(subsets), n), dtype=bool)
    for row, subset in zip(masks, subsets, strict=True):
        row[subset] = True
    return masks


def evaluate(photo: dict[str, Any], model: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    """Return the predictions of the model for the masks of one photo's figures."""
    results = HERE / "results"
    results.mkdir(exist_ok=True)
    name = f"{model['id']}_demo"
    image_text.IMAGE_TEXT_MODEL_IDS[name] = model["checkpoint"]
    grid = model["grid"]
    embedder = image_text.ImageTextTokenModel(
        np.asarray(load_photo(photo, results)),
        (grid, grid),
        model=name,
        mask_strategy="mask",
        batch_size=32,
    )
    tokens = [t["text"] for t in caption_tokens(Network(embedder), photo["caption"])]
    n_patches, n = grid * grid, grid * grid + len(tokens)
    game = SimilarityGame(embedder, photo["caption"], n_patches, len(tokens))
    first, pairs = explanation(data, photo["id"])
    out: dict[str, Any] = {"empty": game.empty_value}

    # the two objects, as players, and the word of the largest main effect
    objects = [
        {
            "name": thing["name"],
            "patches": thing["patches"],
            "words": [n_patches + tokens.index(word) for word in thing["words"]],
        }
        for thing in photo["objects"]
    ]
    out["objects"] = objects
    word = n_patches + int(np.argmax(first[n_patches:]))
    out["word"] = word

    # the masks chosen by hand
    image, text = list(range(n_patches)), list(range(n_patches, n))
    one, two = objects
    presets = {
        "full": image + text,
        "empty": [],
        "image": image,
        "text": text,
        "word": [word],
        "no-one": [i for i in image if i not in one["patches"]] + text,
        "no-two": [i for i in image if i not in two["patches"]] + text,
    }
    values = game.similarity(as_masks(list(presets.values()), n))
    out["presets"] = {key: float(value) for key, value in zip(presets, values, strict=True)}

    # a patch and a word, together and alone, on average over random masks of the other players
    cross = pairs[:n_patches, n_patches:]
    patch, token = np.unravel_index(int(np.argmax(cross)), cross.shape)
    strongest = (int(patch), n_patches + int(token))
    other = n_patches + int(np.argsort(-first[n_patches:])[1]) if word == strongest[1] else word
    rng = np.random.default_rng(1)
    out["together"] = []
    for i, j in (strongest, (strongest[0], other)):
        contexts = rng.random((N_CONTEXTS, n)) < 0.5
        means = {}
        for label, (with_i, with_j) in {
            "neither": (False, False),
            "patch": (True, False),
            "word": (False, True),
            "both": (True, True),
        }.items():
            masks = contexts.copy()
            masks[:, i], masks[:, j] = with_i, with_j
            means[label] = float(game.similarity(masks).mean())
        out["together"].append({"players": [i, j], "interaction": pairs[i, j]} | means)

    # random masks, with every player present with probability p
    out["holdout"] = {}
    order1 = first + 0.5 * pairs.sum(axis=1)  # the first-order conversion (Theorem 1), p = 1/2
    for p in (0.3, 0.5, 0.7):
        masks = np.random.default_rng(1).random((N_TEST, n)) < p
        value = game.similarity(masks)
        predictions = {"order1": masks @ order1, "order2": predict(masks, first, pairs)}
        out["holdout"][str(p)] = {
            "kept": "".join(str(int(kept)) for kept in masks[:, word]),  # is the word kept
            "value": value.tolist(),
            "r2": {key: r2(value, y_hat) for key, y_hat in predictions.items()},
            "rho": {key: float(spearmanr(value, y_hat)[0]) for key, y_hat in predictions.items()},
        }
        if p == 0.5:  # with the constant that fits these masks the best, as the demo's R^2
            out["holdout"][str(p)] |= {
                key: (y_hat + np.mean(value - y_hat)).tolist() for key, y_hat in predictions.items()
            }
    # one of the masks of each p, to show
    out["examples"] = {}
    for p in (0.3, 0.5, 0.7):
        mask = np.random.default_rng(7).random((1, n)) < p
        out["examples"][str(p)] = {
            "players": np.flatnonzero(mask[0]).tolist(),
            "value": float(game.similarity(mask)[0]),
        }

    # the subsets that the explanation values the most and the least, for every size
    ranking = np.argsort(-order1)
    highest, lowest = greedy_cliques(first, pairs, n), greedy_cliques(-first, -pairs, n)
    subsets = {
        "order2-max": [highest[k][0] for k in range(1, n + 1)],
        "order2-min": [lowest[k][0] for k in range(1, n + 1)],
        "order1-max": [sorted(ranking[:k].tolist()) for k in range(1, n + 1)],
        "order1-min": [sorted(ranking[n - k :].tolist()) for k in range(1, n + 1)],
    }
    out["curves"] = {
        key: game.similarity(as_masks(value, n)).tolist() for key, value in subsets.items()
    }
    out["subsets"] = {
        "inserted": subsets["order2-max"][INSERTED - 1],
        "deleted": subsets["order2-min"][n - DELETED - 1],  # what is left without the deleted
    }

    # the demo's cliques, shown alone
    cliques = data["explanations"][f"{photo['id']}/{MODEL}"]["cliques"]
    values = game.similarity(as_masks([members for members, _ in cliques.values()], n))
    out["cliques"] = {size: float(value) for size, value in zip(cliques, values, strict=True)}
    return out


def rounded(value: object) -> object:
    """Return the value with every float in it rounded to four decimals."""
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, dict):
        return {key: rounded(item) for key, item in value.items()}
    if isinstance(value, list):
        return [rounded(item) for item in value]
    return value


def main() -> None:
    """Evaluate the photos and write them to ``figures-data.js``."""
    torch.set_num_threads(4)
    config = tomllib.loads((HERE / "config.toml").read_text(encoding="utf-8"))
    model = next(m for m in config["models"] if m["id"] == MODEL)
    chosen = sys.argv[1:] or [photo["id"] for photo in config["photos"]]
    data = read_data()
    for photo in config["photos"]:
        if photo["id"] not in chosen:
            continue
        result = rounded(evaluate(photo, model, data))
        figures = read_figures()
        figures["model"] = MODEL
        figures["images"][photo["id"]] = result
        # in the demo's order
        order = [p["id"] for p in config["photos"] if p["id"] in figures["images"]]
        figures["images"] = {key: figures["images"][key] for key in order}
        write_figures(figures)
        print(f"{photo['id']}: evaluated", flush=True)


if __name__ == "__main__":
    main()
