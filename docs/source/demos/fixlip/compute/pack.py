"""Write the FIxLIP demo's data, ``../assets/data.js``, and copy its photos next to it.

Reads the explanations of explain.py and the scale of norms.py (in ``results/``) and the texts of
config.toml, keeps what the demo shows, rounded, and writes them as a script that sets
``window.SHAPIQ_FIXLIP`` (a script loads from disk too, where the browser would not fetch a JSON
file). A photo and model without an explanation is left out; the demo disables it.

Usage::

    python pack.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import tomllib
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
CLIQUES = range(2, 11)  # the sizes of the cliques the demo offers
PREFIX = "window.SHAPIQ_FIXLIP = "
FIGURES_PREFIX = "window.SHAPIQ_FIXLIP_FIGURES = "


def read_data(assets: Path = HERE.parent / "assets") -> dict[str, Any]:
    """Return the demo's data, as ``main`` wrote it to ``data.js``."""
    text = (assets / "data.js").read_text(encoding="utf-8")
    return json.loads(text[text.index(PREFIX) + len(PREFIX) :].rstrip().removesuffix(";"))


def read_figures(assets: Path = HERE.parent / "assets") -> dict[str, Any]:
    """Return the data of the page's figures, ``figures-data.js``, or an empty one."""
    path = assets / "figures-data.js"
    if not path.exists():
        return {"images": {}, "paper": {}}
    text = path.read_text(encoding="utf-8")
    start = text.index(FIGURES_PREFIX) + len(FIGURES_PREFIX)
    return json.loads(text[start:].rstrip().removesuffix(";"))


def write_figures(data: dict[str, Any], assets: Path = HERE.parent / "assets") -> None:
    """Write the data of the page's figures, which evaluate.py and paper.py each fill a part of."""
    text = json.dumps(data, separators=(",", ":"), ensure_ascii=False)
    (assets / "figures-data.js").write_text(
        "// The data of the figures of the FIxLIP demo's page (see figures.js), written by\n"
        "// ../compute/evaluate.py (images) and ../compute/paper.py (paper).\n"
        f"{FIGURES_PREFIX}{text};\n",
        encoding="utf-8",
    )
    print(f"figures-data.js: {len(text) / 1024:.0f} KiB")


def compact(result: dict[str, Any]) -> dict[str, Any]:
    """Return what the demo shows of an explanation: the interactions as an upper triangle."""
    n = len(result["first"])
    pairs = result["pairs"]
    return {
        "grid": result["grid"],  # grid x grid patches
        "tokens": [[token["text"], token["space"]] for token in result["tokens"]],
        "similarity": round(result["similarity"], 4),
        "r2": round(result["r2"], 3),
        "first": [round(value, 6) for value in result["first"]],
        "pairs": [round(pairs[i][j], 6) for i in range(n) for j in range(i + 1, n)],
        "cliques": {
            size: [members, round(value, 6)]
            for size, (members, value) in result["cliques"].items()
            if int(size) in CLIQUES
        },
    }


def main() -> None:
    """Write ``../assets/data.js`` from the results and config.toml."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results", type=Path, default=HERE / "results")
    parser.add_argument("--assets", type=Path, default=HERE.parent / "assets")
    args = parser.parse_args()
    config = tomllib.loads((HERE / "config.toml").read_text(encoding="utf-8"))
    norms = json.loads((args.results / "norms.json").read_text(encoding="utf-8"))

    photos, explanations = [], {}
    for photo in config["photos"]:
        photos.append({key: photo[key] for key in ("id", "excerpt", "title", "alt", "credit")})
        shutil.copyfile(args.results / f"{photo['id']}.jpg", args.assets / f"{photo['id']}.jpg")
        for model in config["models"]:
            path = args.results / f"{photo['id']}_{model['id']}.json"
            if not path.exists():
                print(f"{photo['id']}, {model['label']}: not explained, left out")
                continue
            result = json.loads(path.read_text(encoding="utf-8"))
            explanations[f"{photo['id']}/{model['id']}"] = compact(result)
    models = [
        {key: model[key] for key in ("id", "label", "detail")} | {"norms": norms[model["id"]]}
        for model in config["models"]
    ]
    data = {"method": config["method"], "images": photos, "models": models}
    data["explanations"] = explanations
    text = json.dumps(data, separators=(",", ":"), ensure_ascii=False)
    (args.assets / "data.js").write_text(
        "// The FIxLIP demo's explanations (see demo.js), written by ../compute/pack.py.\n"
        f"{PREFIX}{text};\n",
        encoding="utf-8",
    )
    print(f"{len(explanations)} explanations, {len(text) / 1024:.0f} KiB")


if __name__ == "__main__":
    main()
