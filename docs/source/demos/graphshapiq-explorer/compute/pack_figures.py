"""Write the data of the page's figures, ``../assets/figures-data.js``, from ``parts/*.json``.

Every figure script of this folder writes its own part, ``parts/<name>.json`` (e.g. ``sparsity.py``
writes ``parts/sparsity.json``), so that the scripts never write one file at the same time. This
script merges the parts into one script that sets a global, as the explorer's ``data.js`` (a script
loads from disk too, where the browser would not fetch a JSON file)::

    window.SHAPIQ_GRAPHSHAPIQ_EXPLORER_FIGURES = {"<part>": <its JSON>, ...};

The parts are sorted by name; figures.js reads a figure's part as ``c.part`` (the part named as
the figure) or ``c.extra["<part>"]``. Floats are written as they are in the parts, so a script
rounds its values itself (four significant digits are plenty for a figure).

Usage::

    python pack_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARTS = HERE / "parts"
TARGET = HERE.parent / "assets" / "figures-data.js"
PREFIX = "window.SHAPIQ_GRAPHSHAPIQ_EXPLORER_FIGURES = "


def main() -> None:
    """Merge every part into ``../assets/figures-data.js``."""
    parts = {
        path.stem: json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(PARTS.glob("*.json"))
    }
    text = json.dumps(parts, separators=(",", ":"), ensure_ascii=False)
    TARGET.write_text(
        "// The data of the figures of the GraphSHAP-IQ explorer's page (see figures.js), written by\n"
        "// ../compute/pack_figures.py from ../compute/parts/*.json.\n"
        f"{PREFIX}{text};\n",
        encoding="utf-8",
    )
    sizes = ", ".join(
        f"{name} {len(json.dumps(part, separators=(',', ':'))) / 1024:.0f} KiB"
        for name, part in parts.items()
    )
    print(f"figures-data.js: {len(text) / 1024:.0f} KiB ({sizes or 'no parts'})")


if __name__ == "__main__":
    main()
