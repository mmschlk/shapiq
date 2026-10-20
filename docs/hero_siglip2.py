"""Draw the hero image of the README and the docs: SigLIP 2 explained as in FIxLIP.

The image explains the similarity of a photo and its caption in SigLIP 2 with weighted Banzhaf
interactions between image regions and words, as in FIxLIP (Baniecki et al., NeurIPS 2025). The
players are the 7 x 7 regions of the image and the 7 words of the text. The value of a coalition
is the cosine similarity of the image with only the coalition's regions (the others masked in
token space, as in ``shapiq_games.ImageTextSimilarity(mask_strategy="mask")``) and the text with
only the coalition's words (the others masked out of attention, as in FIxLIP).

The explanation is the faithful Banzhaf interaction index (``"FBII"``, FIxLIP with ``p = 0.5``)
up to order 2: the least-squares fit of a 2-additive game on coalitions sampled uniformly. As in
FIxLIP, every sampled image coalition is paired with every text coalition, so 8,192 image and 128
text forward passes give about a million values of the game. The figure shows the strongest
interactions between the words and eight image regions.

The photo is "Drinking Giraffe" by Jean & Nathalie (CC BY 2.0,
https://www.flickr.com/photos/60374292@N00/6721303357), image 304605 of MS COCO, cropped to a
square.

Run it from the root of the repository, with the extras of the games (torch, transformers,
sentencepiece, and protobuf) installed::

    uv run python docs/hero_siglip2.py --save hero.json  # about 20 minutes on a CPU
    uv run python docs/hero_siglip2.py --explanation hero.json  # draw it again
"""

# ruff: noqa: T201

from __future__ import annotations

import argparse
import hashlib
import io
import itertools
import json
import time
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import transformers
from matplotlib.patches import Rectangle
from PIL import Image, ImageColor

from shapiq.plot._config import BLUE, RED
from shapiq_games.vision._image_text_model import IMAGE_TEXT_MODEL_IDS, ImageTextTokenModel

if TYPE_CHECKING:
    from matplotlib.text import Text

PHOTO_URL = "http://images.cocodataset.org/train2017/000000304605.jpg"
PHOTO_SHA256 = "c959b7a12365d66b8274a3471e61bb3b699f16090b1284fbc5a75b07d09479d1"
PHOTO_CROP = (0, 100, 426, 526)  # left, top, right, bottom: the giraffe and its reflection
TEXT = "a giraffe drinking water from a river"
MODEL = "siglip2_vit_b16"
GRID = (7, 7)
OUTPUT = Path("docs/source/_static/images/hero_siglip2.png")

# the regions in the figure (numbered row by row), at their (x, y) in the figure (in inches)
REGIONS = {
    4: (11.0, 8.25),  # the giraffe's back
    11: (13.55, 8.25),  # its body
    10: (16.1, 8.25),  # the base of its neck
    16: (18.65, 8.25),  # its neck
    23: (13.6, 1.75),  # its head at the water
    30: (16.0, 1.75),  # the reflection of its head
    21: (20.4, 1.75),  # the bank
    34: (22.8, 1.75),  # the bank and the water
}
FUNCTION_WORDS = (0, 4, 5)  # "a", "from", "a": drawn muted
MIN_INTERACTION = 1.4e-3  # the weakest interaction drawn

INK, MUTED, BORDER = "#1f2328", "#6e7781", "#d0d7de"


def load_photo() -> Image.Image:
    """Download the photo, check it, and crop it to a square."""
    with urllib.request.urlopen(PHOTO_URL) as response:  # noqa: S310
        data = response.read()
    if hashlib.sha256(data).hexdigest() != PHOTO_SHA256:
        msg = f"The photo at {PHOTO_URL} changed."
        raise ValueError(msg)
    return Image.open(io.BytesIO(data)).convert("RGB").crop(PHOTO_CROP)


def text_embeddings(text: str, coalitions: np.ndarray) -> np.ndarray:
    """Embed the text once per coalition of its words, with the absent words masked out.

    The words are the tokens before ``<eos>`` (each word of the text is one token of SigLIP 2's
    tokenizer); ``<eos>`` and the padding stay, as in FIxLIP.

    Args:
        text: The text.
        coalitions: The boolean coalitions of the words, of shape ``(n_coalitions, n_words)``.

    Returns:
        The unit-length embeddings, of shape ``(n_coalitions, dim)``.
    """
    model_id = IMAGE_TEXT_MODEL_IDS[MODEL]
    tokenizer = transformers.SiglipProcessor.from_pretrained(model_id).tokenizer
    model = transformers.SiglipModel.from_pretrained(model_id).eval()
    ids = tokenizer([text], padding="max_length", max_length=64, return_tensors="pt")["input_ids"]
    n_words = int((ids != tokenizer.pad_token_id).sum()) - 1
    if n_words != coalitions.shape[1]:
        msg = f"Expected one token per word, got {tokenizer.convert_ids_to_tokens(ids[0])}."
        raise ValueError(msg)
    mask = np.ones((coalitions.shape[0], ids.shape[1]), dtype=np.int64)
    mask[:, :n_words] = coalitions
    with torch.no_grad():
        embeddings = model.text_model(
            input_ids=ids.repeat(coalitions.shape[0], 1), attention_mask=torch.as_tensor(mask)
        ).pooler_output
    embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
    return embeddings.numpy().astype(float)


def design_matrix(coalitions: np.ndarray, pairs: list[tuple[int, int]]) -> np.ndarray:
    """Return the features of a 2-additive game: ``1``, ``x_i``, and ``x_i * x_j``."""
    x = coalitions.astype(float)
    first, second = np.array(pairs).T
    return np.hstack([np.ones((x.shape[0], 1)), x, x[:, first] * x[:, second]])


def explain(photo: Image.Image, n_image_coalitions: int, n_test: int) -> dict:
    """Compute the FBII values up to order 2 of the image regions and words.

    Args:
        photo: The square photo.
        n_image_coalitions: The number of image coalitions to fit the explanation on.
        n_test: The number of further image coalitions to check its faithfulness on.

    Returns:
        The similarity, the faithfulness, and the interactions of the explanation.
    """
    model = ImageTextTokenModel(
        np.asarray(photo), GRID, model=MODEL, mask_strategy="mask", batch_size=32
    )
    n_regions, n_words = GRID[0] * GRID[1], len(TEXT.split())
    # every coalition of the words, and coalitions of the regions sampled uniformly
    text_coalitions = np.array(list(itertools.product([False, True], repeat=n_words)))
    rng = np.random.default_rng(0)
    image_coalitions = rng.random((n_image_coalitions + n_test, n_regions)) < 0.5
    image_coalitions[0], image_coalitions[1] = True, False
    start = time.time()
    values = model.image_embeddings(image_coalitions) @ text_embeddings(TEXT, text_coalitions).T
    print(f"{values.size:,} values of the game in {time.time() - start:.0f} s")
    similarity, empty = values[0, -1], values[1, 0]

    def pair_up(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        coalitions = np.hstack(
            [
                np.repeat(image_coalitions[rows], len(text_coalitions), axis=0),
                np.tile(text_coalitions, (len(rows), 1)),
            ]
        )
        return coalitions, values[rows].reshape(-1) - empty

    # least squares, with the normal equations summed over blocks of the paired coalitions
    pairs = list(itertools.combinations(range(n_regions + n_words), 2))
    n_features = 1 + n_regions + n_words + len(pairs)
    gram, moment = np.zeros((n_features, n_features)), np.zeros(n_features)
    for begin in range(0, n_image_coalitions, 64):
        coalitions, y = pair_up(np.arange(begin, min(begin + 64, n_image_coalitions)))
        x = design_matrix(coalitions, pairs)
        gram += x.T @ x
        moment += x.T @ y
    coefficients = np.linalg.solve(gram, moment)

    coalitions, y = pair_up(np.arange(n_image_coalitions, n_image_coalitions + n_test))
    residuals = y - design_matrix(coalitions, pairs) @ coefficients
    r2 = 1 - np.sum(residuals**2) / np.sum((y - y.mean()) ** 2)
    print(f"cosine similarity {similarity:.4f}, R^2 on held-out coalitions {r2:.4f}")
    return {
        "model": IMAGE_TEXT_MODEL_IDS[MODEL],
        "text": TEXT,
        "grid": GRID,
        "similarity": float(similarity),
        "heldout_r2": float(r2),
        "first_order": coefficients[1 : 1 + n_regions + n_words].tolist(),
        "pairs": pairs,
        "second_order": coefficients[1 + n_regions + n_words :].tolist(),
    }


def region_box(region: int, size: int) -> tuple[int, int, int, int]:
    """Return the pixel box (left, top, right, bottom) of a region of a square photo."""
    row, column = divmod(region, GRID[1])
    rows = np.linspace(0, size, GRID[0] + 1).round().astype(int)
    columns = np.linspace(0, size, GRID[1] + 1).round().astype(int)
    return columns[column], rows[row], columns[column + 1], rows[row + 1]


def chip(text: Text, *, pad: float = 0.35, rounding: float = 0.3) -> None:
    """Draw a white box with rounded corners around a text."""
    text.set_bbox(
        {
            "boxstyle": f"round,pad={pad},rounding_size={rounding}",
            "facecolor": "white",
            "edgecolor": BORDER,
            "linewidth": 2.0,
        }
    )


def draw(explanation: dict, photo: Image.Image, path: Path) -> None:
    """Draw the photo and its text, and the network of the strongest interactions."""
    mpl.rcParams["font.family"] = "Inter"
    pixels = np.asarray(photo)
    size, n_regions = pixels.shape[0], GRID[0] * GRID[1]
    words = explanation["text"].split()
    fig = plt.figure(figsize=(24, 10), dpi=100)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set(xlim=(0, 24), ylim=(0, 10), aspect="equal")
    ax.axis("off")

    # the input: the photo with the regions of the network bright, and the text
    faded = pixels.astype(float)
    veil = np.ones((size, size), dtype=bool)
    for region in REGIONS:
        left, top, right, bottom = region_box(region, size)
        veil[top:bottom, left:right] = False
    faded[veil] = 0.45 * faded[veil] + 0.55 * 255
    x0, y0, side = 0.7, 2.25, 7.4
    ax.imshow(faded.astype(np.uint8), extent=(x0, x0 + side, y0, y0 + side), zorder=2)
    for k in range(1, GRID[0]):
        ax.plot([x0, x0 + side], [y0 + side * k / GRID[0]] * 2, color="white", lw=1.5, zorder=3)
        ax.plot([x0 + side * k / GRID[1]] * 2, [y0, y0 + side], color="white", lw=1.5, zorder=3)
    ax.add_patch(Rectangle((x0, y0), side, side, fill=False, edgecolor=INK, lw=2.5, zorder=4))
    caption = ax.text(
        4.4,
        1.2,
        explanation["text"],
        ha="center",
        va="center",
        fontsize=25,
        family="JetBrains Mono",
        color=INK,
    )
    chip(caption, pad=0.55, rounding=0.25)
    ax.text(
        4.4,
        0.4,
        f"SigLIP 2 cosine similarity: {explanation['similarity']:.2f}",
        ha="center",
        va="center",
        fontsize=23,
        color=MUTED,
    )

    # the players: the words in a row between x = 10 and x = 23.6, the regions above and below
    texts = [
        ax.text(
            0, 5.0, word, ha="center", va="center", fontsize=28, family="JetBrains Mono", zorder=7
        )
        for word in words
    ]
    renderer = fig.canvas.get_renderer()
    widths = [text.get_window_extent(renderer).width / fig.dpi for text in texts]
    pad = 2 * 0.35 * 28 / 72  # the padding of a chip, in inches
    gap = (23.6 - 10.0 - sum(widths) - pad * len(words)) / (len(words) - 1)
    position = dict(REGIONS)
    x = 10.0
    for k, (text, width) in enumerate(zip(texts, widths, strict=True)):
        text.set_x(x + (pad + width) / 2)
        text.set_color(MUTED if k in FUNCTION_WORDS else INK)
        chip(text)
        position[n_regions + k] = (x + (pad + width) / 2, 5.0)
        x += width + pad + gap

    # the interactions between a word and a region, their width proportional to their value
    edges = [
        (pair, value)
        for pair, value in zip(
            map(tuple, explanation["pairs"]), explanation["second_order"], strict=True
        )
        if all(player in position for player in pair)
        and (pair[0] < n_regions) != (pair[1] < n_regions)
        and abs(value) >= MIN_INTERACTION
    ]
    largest = max(abs(value) for _, value in edges)
    for (i, j), value in sorted(edges, key=lambda edge: abs(edge[1])):
        strength = abs(value) / largest
        ax.plot(
            *zip(position[i], position[j], strict=True),
            color=RED.hex if value > 0 else BLUE.hex,
            lw=3 + 21 * strength,
            alpha=0.25 + 0.75 * strength,
            solid_capstyle="round",
            zorder=1,
        )
    for region, (x, y) in REGIONS.items():
        left, top, right, bottom = region_box(region, size)
        ax.imshow(
            pixels[top:bottom, left:right],
            extent=(x - 0.825, x + 0.825, y - 0.825, y + 0.825),
            zorder=5,
        )
        ax.add_patch(
            Rectangle(
                (x - 0.825, y - 0.825), 1.65, 1.65, fill=False, edgecolor=INK, lw=3.0, zorder=6
            )
        )

    for k, (color, label) in enumerate(((RED, "raises similarity"), (BLUE, "lowers similarity"))):
        ax.plot([20.45, 21.05], [8.75 - 0.5 * k] * 2, color=color.hex, lw=9, solid_capstyle="round")
        ax.text(21.4, 8.75 - 0.5 * k, label, ha="left", va="center", fontsize=20, color=MUTED)

    buffer = io.BytesIO()
    fig.savefig(buffer, dpi=fig.dpi, facecolor="white")
    plt.close(fig)
    save_with_palette(Image.open(buffer).convert("RGB"), path)


def save_with_palette(image: Image.Image, path: Path) -> None:
    """Save the image with a palette of 256 colors, a third of the size without a visible change.

    The photo would take the whole palette, so the colors of the drawing are reserved: the red
    and blue of the edges at every opacity, and the colors of the text and the borders.
    """
    white = np.full(3, 255.0)
    drawing = [
        alpha * np.array(ImageColor.getrgb(color.hex)) + (1 - alpha) * white
        for color in (RED, BLUE)
        for alpha in np.linspace(0.25, 1.0, 16)
    ]
    drawing += [np.array(ImageColor.getrgb(color)) for color in (INK, MUTED, BORDER)]
    drawing.append(white)
    n_photo = 256 - len(drawing)
    palette = image.quantize(colors=n_photo).getpalette()[: 3 * n_photo]
    palette += [round(float(channel)) for color in drawing for channel in color]
    reference = Image.new("P", (1, 1))
    reference.putpalette(palette)
    quantized = image.quantize(palette=reference, dither=Image.Dither.FLOYDSTEINBERG)
    # PIL looks colors up at a lower precision and can pick a close color instead (the white of
    # the background becomes 253, 254, 254): snap the colors next to the drawing's onto them
    entries = np.array(quantized.getpalette(), dtype=float).reshape(-1, 3)
    for color in drawing:
        entries[np.abs(entries - color).max(axis=1) <= 3] = color
    quantized.putpalette(entries.round().astype(int).flatten().tolist())
    quantized.save(path, optimize=True)


def main() -> None:
    """Compute or load the explanation, and draw the hero image."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--explanation", type=Path, help="a saved explanation to draw")
    parser.add_argument("--save", type=Path, help="where to save the computed explanation")
    parser.add_argument("--n-image-coalitions", type=int, default=8192)
    parser.add_argument("--n-test", type=int, default=512)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    photo = load_photo()
    if args.explanation is not None:
        explanation = json.loads(args.explanation.read_text())
    else:
        explanation = explain(photo, args.n_image_coalitions, args.n_test)
        if args.save is not None:
            args.save.write_text(json.dumps(explanation))
    draw(explanation, photo, args.output)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
