"""Datasets for the games in :mod:`shapiq_games`.

No data ships with the package. Real-world data is downloaded on first use from pinned sources,
verified by checksum, and cached locally (see :func:`get_data_dir`). Synthetic data is generated
from a seed.

Examples:
    >>> from shapiq_games.datasets import list_datasets, load_dataset
    >>> dataset = load_dataset("california_housing")
    >>> dataset.task, dataset.n_features
    ('regression', 8)
    >>> split = dataset.split(test_size=0.2, random_state=42)
    >>> xor = load_dataset("xor", n_samples=500, random_state=0)
"""

from . import _synthetic, _tabarena, _tabular  # noqa: F401  (registers the datasets)
from ._cache import get_data_dir
from ._images import list_example_images, load_example_image
from ._registry import (
    Dataset,
    DatasetSpec,
    DatasetSplit,
    get_dataset_spec,
    list_datasets,
    load_dataset,
)
from ._synthetic import load_curthvds_synthetic
from ._tabarena import TABARENA_DATASETS

__all__ = [
    "Dataset",
    "DatasetSpec",
    "DatasetSplit",
    "TABARENA_DATASETS",
    "get_data_dir",
    "get_dataset_spec",
    "list_datasets",
    "list_example_images",
    "load_curthvds_synthetic",
    "load_dataset",
    "load_example_image",
]
