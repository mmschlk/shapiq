"""Evaluate SigLIP 2 on the masks that the figures of the demo's page show: ``figures.json``.

The figures of the page (figures.py) explain the demo step by step on its first photo, the baby
elephant and the egret. Most of them only read the demo's data. Some show what the model itself
predicts for a masked photo and caption, and these predictions are computed here, once.

- ``presets``: the similarity of masks chosen by hand (the whole pair, the empty pair, one
  modality, the elephant or the egret with the words that name them).
- ``together``: the similarity with and without a patch and a word, on average over random masks
  of the other players. The four averages give the pair's Banzhaf interaction.
- ``holdout``: the similarity of 512 random masks for each of p = 0.3, 0.5 and 0.7, next to what
  the demo's explanation and its first-order conversion predict for them. The page quotes the
  R^2 of all three and draws those of p = 0.5.
- ``curves`` and ``subsets``: the similarity of the subsets that the explanation values the most
  and the least, for every size, which gives FIxLIP's insertion and deletion curves. The search
  for the subsets is the demo's greedy search for cliques.
- ``cliques``: the similarity of the demo's cliques, shown alone.
- ``cross``: the similarity of every combination of 6 masked photos and 6 masked captions.

Usage::

    python evaluate.py

The model takes about 4 minutes on 4 CPU cores. ``figures.json`` is part of the repository, so
figures.py runs without the model.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
import torch
from explain import Network, SimilarityGame, caption_tokens, greedy_cliques, load_photo
from pack import read_data

from shapiq_games.vision import _image_text_model as image_text

HERE = Path(__file__).resolve().parent
PHOTO, MODEL = "elephant", "siglip2"
N_TEST = 512  # as explain.py
N_CONTEXTS = 256  # the random masks that an interaction is averaged over
# the patches of the 8 x 8 grid that show the elephant and the egret, chosen by eye
ELEPHANT = [13, 14, 15, 20, 21, 22, 23, 27, 28, 29, 30, 31, 35, 36, 37, 38, 39, 43, 45, 46, 47]
ELEPHANT += [53, 54, 55]
EGRET = [40, 41, 48]
# the pairs of a patch and a token of ``together``
PAIRS = {"egret": (40, "bird"), "trunk": (35, "elephant"), "mismatch": (40, "elephant")}


def explanation(data: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Return the main effects and the matrix of interactions of the demo's explanation."""
    e = data["explanations"][f"{PHOTO}/{MODEL}"]
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


def main() -> None:
    """Evaluate the masks and write ``figures.json``."""
    torch.set_num_threads(4)
    config = tomllib.loads((HERE / "config.toml").read_text(encoding="utf-8"))
    photo = next(p for p in config["photos"] if p["id"] == PHOTO)
    model = next(m for m in config["models"] if m["id"] == MODEL)
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
    first, pairs = explanation(read_data())
    token = {text: n_patches + k for k, text in enumerate(tokens)}
    rng = np.random.default_rng(1)
    out: dict[str, Any] = {"photo": PHOTO, "model": MODEL, "tokens": tokens, "grid": grid}
    out["empty"] = game.empty_value
    out["elephant"], out["egret"] = ELEPHANT, EGRET

    # the masks chosen by hand
    image, text = list(range(n_patches)), list(range(n_patches, n))
    words = {"elephant": [token["elephant"]], "bird": [token["white"], token["bird"]]}
    presets = {
        "full": image + text,
        "empty": [],
        "image": image,
        "text": text,
        "elephant-caption": ELEPHANT + text,
        "elephant-elephant": ELEPHANT + words["elephant"],
        "elephant-bird": ELEPHANT + words["bird"],
        "egret-elephant": EGRET + words["elephant"],
        "egret-bird": EGRET + words["bird"],
        "no-elephant": [i for i in image if i not in ELEPHANT] + text,
        "no-egret": [i for i in image if i not in EGRET] + text,
    }
    values = game.similarity(as_masks(list(presets.values()), n))
    out["presets"] = {
        key: {"players": players, "value": float(value)}
        for (key, players), value in zip(presets.items(), values, strict=True)
    }

    # a patch and a word, together and alone, on average over random masks of the other players
    out["together"] = {}
    for key, (patch, word) in PAIRS.items():
        i, j = patch, token[word]
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
        out["together"][key] = {"players": [i, j], "interaction": pairs[i, j]} | means

    # random masks, with every player present with probability p
    out["holdout"] = {}
    order1 = first + 0.5 * pairs.sum(axis=1)  # the first-order conversion (Theorem 1), p = 1/2
    for p in (0.3, 0.5, 0.7):
        masks = np.random.default_rng(1).random((N_TEST, n)) < p
        out["holdout"][str(p)] = {
            "elephant": masks[:, token["elephant"]].astype(int).tolist(),  # is the word kept
            "value": game.similarity(masks).tolist(),
            "order2": predict(masks, first, pairs).tolist(),
            "order1": (masks @ order1).tolist(),
        }
    # four of the masks of each p, to show
    out["examples"] = {}
    for p in (0.3, 0.5, 0.7):
        masks = np.random.default_rng(7).random((4, n)) < p
        out["examples"][str(p)] = [
            {"players": np.flatnonzero(mask).tolist(), "value": float(value)}
            for mask, value in zip(masks, game.similarity(masks), strict=True)
        ]

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
    out["subsets"] = {key: subsets[key] for key in ("order2-max", "order2-min")}

    # the demo's cliques, shown alone
    cliques = read_data()["explanations"][f"{PHOTO}/{MODEL}"]["cliques"]
    values = game.similarity(as_masks([members for members, _ in cliques.values()], n))
    out["cliques"] = {size: float(value) for size, value in zip(cliques, values, strict=True)}

    # every combination of 6 masked photos and 6 masked captions
    photos = np.random.default_rng(3).random((6, n_patches)) < 0.5
    captions = np.random.default_rng(4).random((6, len(tokens))) < 0.5
    masks = np.array([np.concatenate([a, b]) for a in photos for b in captions])
    out["cross"] = {
        "photos": [np.flatnonzero(a).tolist() for a in photos],
        "captions": [np.flatnonzero(b).tolist() for b in captions],
        "values": game.similarity(masks).reshape(6, 6).tolist(),
    }

    def rounded(value: object) -> object:
        if isinstance(value, float):
            return round(value, 5)
        if isinstance(value, dict):
            return {key: rounded(item) for key, item in value.items()}
        if isinstance(value, list):
            return [rounded(item) for item in value]
        return value

    path = HERE / "figures.json"
    path.write_text(json.dumps(rounded(out), separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"{path.name}: {path.stat().st_size / 1024:.0f} KiB")


if __name__ == "__main__":
    main()
