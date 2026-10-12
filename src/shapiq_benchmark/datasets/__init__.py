"""Datasets of the benchmark setups (:mod:`shapiq_benchmark.setups`).

No data ships with the package. Real-world data is downloaded on first use from pinned sources,
verified by checksum, and cached locally (see :func:`get_data_dir`). Synthetic data is generated
from a seed. The image games use Imagenette, a ten-class subset of ImageNet
(:func:`load_imagenette`).

Examples:
    >>> from shapiq_benchmark.datasets import list_datasets, load_dataset
    >>> dataset = load_dataset("breast_cancer")
    >>> dataset.task, dataset.n_features
    ('classification', 30)
    >>> split = dataset.split(test_size=0.2, random_state=42)
    >>> xor = load_dataset("xor", n_samples=500, random_state=0)
"""

from . import _synthetic, _tabarena, _tabular  # noqa: F401  (registers the datasets)
from ._cache import get_data_dir
from ._imagenette import (
    IMAGENETTE_CLASSES,
    ImageDataset,
    ImagenetteSize,
    ImagenetteSplit,
    load_imagenette,
)
from ._registry import (
    Dataset,
    DatasetKind,
    DatasetSpec,
    DatasetSplit,
    get_dataset_spec,
    list_datasets,
    load_dataset,
    register_dataset,
)
from ._synthetic import CausalSetting, load_curthvds_synthetic
from ._tabarena import TABARENA_DATASETS

__all__ = [
    "CausalSetting",
    "Dataset",
    "DatasetKind",
    "DatasetSpec",
    "DatasetSplit",
    "IMAGENETTE_CLASSES",
    "ImageDataset",
    "ImagenetteSize",
    "ImagenetteSplit",
    "TABARENA_DATASETS",
    "get_data_dir",
    "get_dataset_spec",
    "list_datasets",
    "load_curthvds_synthetic",
    "load_dataset",
    "load_imagenette",
    "register_dataset",
]
