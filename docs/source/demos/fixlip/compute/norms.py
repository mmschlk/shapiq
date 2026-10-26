"""The scale of the FIxLIP demo's similarities: how similar COCO photos are to captions.

200 random photos of COCO's val2017 split, each with its first caption, lowercased as the demo's
captions: for each model, the cosine similarities of the photos with their own captions (matching
pairs) and with the other photos' captions (unrelated pairs). The demo places its similarities on
these two distributions, as 21 quantiles each, written to ``results/norms.json``.

Usage::

    python norms.py                                   # downloads COCO's annotations (241 MB)
    python norms.py --captions path/to/captions_val2017.json

The captions are COCO's annotations (``annotations/captions_val2017.json`` of
``annotations_trainval2017.zip``), downloaded to ``results/coco/`` if no file is given, and the 200
photos are downloaded to ``results/coco-val2017/``. It takes a few minutes on a CPU.
"""

from __future__ import annotations

import argparse
import json
import random
import tomllib
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

HERE = Path(__file__).resolve().parent
COCO = "https://s3.amazonaws.com/images.cocodataset.org"
N_PHOTOS = 200
BATCH = 25


def coco_captions(path: Path | None, results: Path) -> dict[int, list[str]]:
    """Return the captions of every photo of COCO's val2017 split."""
    if path is None:
        archive = results / "coco" / "annotations_trainval2017.zip"
        if not archive.exists():
            archive.parent.mkdir(parents=True, exist_ok=True)
            url = f"{COCO}/annotations/annotations_trainval2017.zip"
            urllib.request.urlretrieve(url, archive)  # noqa: S310
        with zipfile.ZipFile(archive) as files:
            data = json.loads(files.read("annotations/captions_val2017.json"))
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
    captions: dict[int, list[str]] = {}
    for annotation in data["annotations"]:
        captions.setdefault(annotation["image_id"], []).append(annotation["caption"].strip())
    return captions


def sample(captions: dict[int, list[str]], photos: Path) -> tuple[list[Image.Image], list[str]]:
    """Return 200 random photos (by id) and their first captions, lowercased, as the demo's."""
    random.seed(0)
    ids = sorted(random.sample(sorted(captions), N_PHOTOS))
    photos.mkdir(parents=True, exist_ok=True)

    def download(photo: int) -> Path:
        path = photos / f"{photo}.jpg"
        if not path.exists():
            urllib.request.urlretrieve(f"{COCO}/val2017/{photo:012d}.jpg", path)  # noqa: S310
        return path

    with ThreadPoolExecutor(8) as pool:
        paths = list(pool.map(download, ids))
    images = [Image.open(path).convert("RGB") for path in paths]
    return images, [captions[photo][0].lower() for photo in ids]


def quantiles(images: list[Image.Image], texts: list[str], model: dict[str, Any]) -> dict:
    """Return the quantiles of the similarities of matching and of unrelated pairs for a model."""
    network = AutoModel.from_pretrained(model["checkpoint"]).eval()
    processor = AutoProcessor.from_pretrained(model["checkpoint"])
    siglip = "siglip" in model["checkpoint"]
    # SigLIP was trained on captions padded to 64 tokens
    padding = {"padding": "max_length", "max_length": 64} if siglip else {"padding": True}
    image_embeddings, text_embeddings = [], []
    with torch.no_grad():
        for start in range(0, len(texts), BATCH):
            inputs = processor(
                images=images[start : start + BATCH],
                text=texts[start : start + BATCH],
                return_tensors="pt",
                **padding,
            )
            outputs = network(**inputs)
            image_embeddings.append(outputs.image_embeds)
            text_embeddings.append(outputs.text_embeds)
    image = torch.cat(image_embeddings)
    text = torch.cat(text_embeddings)
    image = image / image.norm(dim=-1, keepdim=True)
    text = text / text.norm(dim=-1, keepdim=True)
    similarity = (image @ text.T).numpy()
    matching = np.diag(similarity)
    unrelated = similarity[~np.eye(len(texts), dtype=bool)]
    steps = np.linspace(0, 1, 21)
    print(
        f"{model['label']}: median similarity {np.median(matching):.3f} of matching pairs, "
        f"{np.median(unrelated):.3f} of unrelated pairs",
        flush=True,
    )
    return {
        "matching": np.round(np.quantile(matching, steps), 4).tolist(),
        "unrelated": np.round(np.quantile(unrelated, steps), 4).tolist(),
        "pairs": len(texts),
    }


def main() -> None:
    """Write the quantiles of every model of config.toml to ``results/norms.json``."""
    config = tomllib.loads((HERE / "config.toml").read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--captions", type=Path, help="COCO's captions_val2017.json")
    parser.add_argument("--results", type=Path, default=HERE / "results")
    args = parser.parse_args()
    torch.set_num_threads(4)
    images, texts = sample(
        coco_captions(args.captions, args.results), args.results / "coco-val2017"
    )
    norms = {model["id"]: quantiles(images, texts, model) for model in config["models"]}
    (args.results / "norms.json").write_text(json.dumps(norms))


if __name__ == "__main__":
    main()
