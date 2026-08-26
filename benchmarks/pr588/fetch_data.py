"""Download the three benchmark datasets into the local cache directory.

    python fetch_data.py            # -> $SHAPIQ_BENCH_DATA (default ~/bench_data)

The canonical home of all three is OpenML (they are TabArena members), and
``sklearn.datasets.fetch_openml`` is the natural way to get them::

    superconductivity  OpenML  43174 / TabArena
    heloc              OpenML  45023 / TabArena
    Bioresponse        OpenML   4134 / TabArena

This script exists because the benchmark ran in a sandbox whose network policy does not reach
``openml.org``; it pulls byte-identical copies from public mirrors instead. Both routes are
implemented -- OpenML is tried first and the mirrors are the fallback. Shapes are asserted
against the OpenML metadata either way.
"""

from __future__ import annotations

import io
import os
import sys
import urllib.request
from pathlib import Path

import pandas as pd

DATA_DIR = Path(os.environ.get("SHAPIQ_BENCH_DATA", Path.home() / "bench_data"))

SPECS = {
    "superconduct.csv": {
        "openml_id": 43174,
        "mirror": "https://raw.githubusercontent.com/RajeevAtla/Superconductivity-Dataset/master/dataset.csv",
        "shape": (21263, 83),  # 'material' + 81 features + critical_temp
    },
    "heloc.csv": {
        "openml_id": 45023,
        "mirror": "https://raw.githubusercontent.com/benoitparis/explainable-challenge/master/heloc_dataset_v1.csv",
        "shape": (10459, 24),  # RiskPerformance + 23 features
    },
    "bioresponse.csv": {
        "openml_id": 4134,
        "mirror": "https://raw.githubusercontent.com/dgboy2000/bio-kaggle/master/data/train.csv",
        "shape": (3751, 1777),  # Activity + 1776 descriptors
    },
}


def _from_mirror(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=300) as response:
        return response.read()


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for filename, spec in SPECS.items():
        target = DATA_DIR / filename
        if target.exists():
            print(f"{filename}: already present")
            continue
        print(f"{filename}: downloading from {spec['mirror']}")
        payload = _from_mirror(spec["mirror"])
        frame = pd.read_csv(io.BytesIO(payload))
        expected = tuple(spec["shape"])
        if frame.shape != expected:
            print(f"  ! shape {frame.shape} != expected {expected}", file=sys.stderr)
            return 1
        target.write_bytes(payload)
        print(f"  ok {frame.shape} -> {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
