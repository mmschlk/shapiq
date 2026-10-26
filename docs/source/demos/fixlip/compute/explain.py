"""Explain the FIxLIP demo's photos and captions with ProxySHAP: one JSON file per photo and model.

The players of a photo and its caption are the image patches (one per patch: 8 x 8 for SigLIP 2 at
256 pixels, 7 x 7 for CLIP at 224 pixels, both ViT-B/32 as in FIxLIP's figures) and the tokens of
the caption. The value of a coalition is the cosine similarity of the image with only the
coalition's patches (the others masked in token space, as the embedding of a patch at the
normalization mean) and the caption with only the coalition's tokens (the others masked out of
attention, as in FIxLIP).

ProxySHAP fits an XGBoost proxy, with the HPO-informed settings of the ProxySHAP paper, to 10,000
coalitions, every player present with probability 1/2, and reads FBII (FIxLIP with p = 1/2) up to
order 2 off the proxy's trees exactly, without the residual adjustment. The explanation is scored by
its R^2 on 512 fresh coalitions, with its constant refit: ProxySHAP returns v(empty) as the order-0
term, and the demo shows only the first- and second-order terms. Finally, FIxLIP's greedy search
finds the cliques, the players that together raise the similarity the most.

Usage::

    python explain.py                                # every photo and model of config.toml
    python explain.py --photos elephant --models clip

The COCO photos are downloaded to ``results/coco/``. The explanations go to
``results/<photo>_<model>.json``, next to the cropped photos (``results/<photo>.jpg``); explanations
that exist are not computed again. One photo and model take about 5 to 10 minutes on 4 CPU cores.
"""

from __future__ import annotations

import argparse
import json
import math
import time
import tomllib
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from xgboost import XGBRegressor

import shapiq
from shapiq_games.vision import _image_text_model as image_text

HERE = Path(__file__).resolve().parent
COCO = "https://s3.amazonaws.com/images.cocodataset.org/train2017/{:012d}.jpg"
# the HPO-informed settings of ProxySHAP's XGBoost proxy
PROXY = {"n_estimators": 2000, "max_depth": 3, "learning_rate": 0.05, "reg_lambda": 5.0}
N_TEST = 512  # the fresh coalitions the explanation is scored on
MAX_CLIQUE = 12
WORD = "\N{LOWER ONE EIGHTH BLOCK}"  # SentencePiece's mark of a token that starts a word (SigLIP)


def load_photo(photo: dict[str, Any], results: Path) -> Image.Image:
    """Return the square crop of a COCO photo that the models see, and save it for the demo."""
    path = results / "coco" / f"{photo['coco']}.jpg"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(COCO.format(photo["coco"]), path)  # noqa: S310
    image = Image.open(path).convert("RGB").crop(tuple(photo["crop"]))
    image.resize((448, 448), Image.Resampling.LANCZOS).save(
        results / f"{photo['id']}.jpg", quality=86, optimize=True, progressive=True
    )
    return image


class Network:
    """The tokenizer and the model of a ``shapiq_games`` image-text model, for the caption."""

    def __init__(self, model: image_text.ImageTextTokenModel) -> None:
        """Take the tokenizer and the model that ``shapiq_games`` loaded."""
        self.tokenizer = model._tokenizer  # noqa: SLF001
        self.model = model._model  # noqa: SLF001
        self.siglip = model._siglip  # noqa: SLF001


def caption_tokens(network: Network, caption: str) -> list[dict[str, Any]]:
    """Return the text of every token of the caption and whether a space precedes it."""
    tokenizer = network.tokenizer
    if network.siglip:
        ids = tokenizer([caption], padding="max_length", max_length=64)["input_ids"][0]
        n = sum(i != tokenizer.pad_token_id for i in ids) - 1  # the tokens before </s>
        pieces = tokenizer.convert_ids_to_tokens(ids[:n])
        return [{"text": p.replace(WORD, ""), "space": p.startswith(WORD)} for p in pieces]
    pieces = tokenizer.convert_ids_to_tokens(tokenizer([caption])["input_ids"][0][1:-1])
    tokens, space = [], False
    for piece in pieces:
        tokens.append({"text": piece.replace("</w>", ""), "space": space})
        space = piece.endswith("</w>")
    return tokens


def text_embeddings(network: Network, caption: str, coalitions: np.ndarray) -> np.ndarray:
    """Embed the caption once per coalition of its tokens, the absent tokens masked out of attention."""
    tokenizer, net, siglip = network.tokenizer, network.model, network.siglip
    if siglip:  # SigLIP was trained on captions padded to 64 tokens
        ids = tokenizer([caption], padding="max_length", max_length=64, return_tensors="pt")
        ids = ids["input_ids"]
        first, n = 0, int((ids != tokenizer.pad_token_id).sum()) - 1  # the tokens before </s>
    else:
        ids = tokenizer([caption], return_tensors="pt")["input_ids"]
        first, n = 1, ids.shape[1] - 2  # the tokens between <|startoftext|> and <|endoftext|>
    outputs = []
    for start in range(0, len(coalitions), 256):
        chunk = coalitions[start : start + 256]
        mask = np.ones((len(chunk), ids.shape[1]), dtype=np.int64)
        mask[:, first : first + n] = chunk
        with torch.no_grad():
            features = net.text_model(
                input_ids=ids.repeat(len(chunk), 1), attention_mask=torch.as_tensor(mask)
            ).pooler_output
            if not siglip:
                features = net.text_projection(features)
        outputs.append((features / features.norm(dim=-1, keepdim=True)).numpy())
    return np.concatenate(outputs).astype(float)


class SimilarityGame(shapiq.Game):
    """The cosine similarity of a photo's present patches and a caption's present tokens."""

    def __init__(
        self, model: image_text.ImageTextTokenModel, caption: str, n_patches: int, n_tokens: int
    ) -> None:
        """Embed the empty coalition, the game's normalization value."""
        self.model, self.caption, self.n_patches = model, caption, n_patches
        self.network = Network(model)
        self.text_cache: dict[bytes, np.ndarray] = {}  # the caption's coalitions repeat
        self.empty_value = float(self.similarity(np.zeros((1, n_patches + n_tokens), bool))[0])
        super().__init__(
            n_players=n_patches + n_tokens, normalize=True, normalization_value=self.empty_value
        )

    def similarity(self, coalitions: np.ndarray) -> np.ndarray:
        """Return the cosine similarity of every coalition's image and caption."""
        coalitions = np.asarray(coalitions, dtype=bool)
        image = self.model.image_embeddings(coalitions[:, : self.n_patches])
        tokens = coalitions[:, self.n_patches :]
        missing = {row.tobytes(): row for row in tokens if row.tobytes() not in self.text_cache}
        if missing:
            rows = np.array(list(missing.values()))
            embeddings = text_embeddings(self.network, self.caption, rows)
            for row, embedding in zip(rows, embeddings, strict=True):
                self.text_cache[row.tobytes()] = embedding
        text = np.array([self.text_cache[row.tobytes()] for row in tokens])
        return np.sum(image * text, axis=1)

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        """Return the similarity of the coalitions (normalized by shapiq.Game)."""
        return self.similarity(coalitions)


def greedy_cliques(
    first: np.ndarray, pairs: np.ndarray, max_size: int
) -> dict[int, tuple[list[int], float]]:
    """Return the cliques of FIxLIP's greedy search, the best of each size and their values.

    From every player, the search adds the player that raises the clique's value the most, one at a
    time; a clique's value is the sum of its players' main effects and their interactions.
    """
    n = len(first)
    best: dict[int, tuple[list[int], float]] = {}
    for start in range(n):
        clique, value = [start], float(first[start])
        gains = first + pairs[:, start]
        free = np.ones(n, dtype=bool)
        free[start] = False
        for size in range(1, max_size + 1):
            if size not in best or value > best[size][1]:
                best[size] = (sorted(clique), value)
            if size == max_size:
                break
            nxt = int(np.argmax(np.where(free, gains, -np.inf)))
            value += float(gains[nxt])
            clique.append(nxt)
            free[nxt] = False
            gains = gains + pairs[:, nxt]
    return best


def r2(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Return the R^2 of a prediction, with the best constant added to it."""
    y_hat = y_hat + np.mean(y - y_hat)
    return float(1 - np.sum((y - y_hat) ** 2) / np.sum((y - y.mean()) ** 2))


def explain(photo: dict[str, Any], model: dict[str, Any], budget: int, results: Path) -> None:
    """Explain a photo and its caption for a model, and write the explanation to ``results``."""
    # shapiq_games names its models: CLIP ViT-B/32 is one of them, SigLIP 2 ViT-B/32 is not; a name
    # starting with "siglip" is run as SigLIP
    name = f"{model['id']}_demo"
    image_text.IMAGE_TEXT_MODEL_IDS[name] = model["checkpoint"]
    grid = (model["grid"], model["grid"])
    embedder = image_text.ImageTextTokenModel(
        np.asarray(load_photo(photo, results)),
        grid,
        model=name,
        mask_strategy="mask",
        batch_size=32,
    )
    tokens = caption_tokens(Network(embedder), photo["caption"])
    game = SimilarityGame(embedder, photo["caption"], grid[0] * grid[1], len(tokens))
    n = game.n_players

    start = time.time()
    approximator = shapiq.approximator.ProxySHAP(
        n=n,
        index="FBII",
        max_order=2,
        proxy_model=XGBRegressor(**PROXY, random_state=0, n_jobs=4),
        adjustment=False,
        # coalitions of players present with probability 1/2, where FBII puts its weight
        sampling_weights=np.array([math.comb(n, k) for k in range(n + 1)], dtype=float),
        random_state=0,
    )
    values = approximator.approximate(budget=budget, game=game)
    seconds = time.time() - start
    first = np.array([values[(i,)] for i in range(n)])
    pairs = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            pairs[i, j] = pairs[j, i] = values[(i, j)]
    cliques = greedy_cliques(first, pairs, MAX_CLIQUE)
    similarity = float(game.similarity(np.ones((1, n), dtype=bool))[0])

    test = np.random.default_rng(1).random((N_TEST, n)) < 0.5
    y = game.similarity(test) - game.empty_value
    x = test.astype(float)
    y_hat = x @ first + 0.5 * np.einsum("ai,ij,aj->a", x, pairs, x)
    result = {
        "photo": photo["id"],
        "model": model["id"],
        "checkpoint": model["checkpoint"],
        "caption": photo["caption"],
        "budget": budget,
        "seconds": round(seconds),
        "grid": model["grid"],
        "tokens": tokens,
        "similarity": similarity,
        "empty": game.empty_value,
        "r2": r2(y, y_hat),
        "first": first.tolist(),
        "pairs": pairs.tolist(),
        "cliques": {str(size): clique for size, clique in cliques.items()},
    }
    (results / f"{photo['id']}_{model['id']}.json").write_text(json.dumps(result))
    print(
        f"{photo['id']}, {model['label']}: {n} players, similarity {similarity:.3f}, "
        f"R^2 {result['r2']:.3f}, {seconds:.0f} s; clique of 6: {cliques[6][0]}",
        flush=True,
    )


def main() -> None:
    """Explain the photos and models of config.toml that are not explained yet."""
    config = tomllib.loads((HERE / "config.toml").read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--photos", nargs="+", default=[p["id"] for p in config["photos"]])
    parser.add_argument("--models", nargs="+", default=[m["id"] for m in config["models"]])
    parser.add_argument("--results", type=Path, default=HERE / "results")
    args = parser.parse_args()
    torch.set_num_threads(4)
    args.results.mkdir(parents=True, exist_ok=True)
    for model in (m for m in config["models"] if m["id"] in args.models):
        for photo in (p for p in config["photos"] if p["id"] in args.photos):
            if not (args.results / f"{photo['id']}_{model['id']}.json").exists():
                explain(photo, model, config["budget"], args.results)


if __name__ == "__main__":
    main()
