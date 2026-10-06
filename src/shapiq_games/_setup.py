"""Shared plumbing for the ``from_config`` constructors of the games."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .datasets import DatasetSplit, load_dataset
from .models import fit_model

__all__ = ["ConfiguredSetup", "configure"]


@dataclass(frozen=True)
class ConfiguredSetup:
    """A dataset split and (optionally) a model fitted on its training part."""

    split: DatasetSplit
    model: Any
    config: dict[str, Any]


def configure(
    *,
    dataset: str,
    model: str | None,
    random_state: int,
    test_size: float = 0.2,
    preset: str | None = None,
    model_params: dict[str, Any] | None = None,
    dataset_params: dict[str, Any] | None = None,
) -> ConfiguredSetup:
    """Load a dataset, split it, and fit a model on the training part, all seeded.

    Args:
        dataset: The dataset name (see :func:`shapiq_games.datasets.list_datasets`).
        model: The model name (see :data:`shapiq_games.models.MODEL_NAMES`), or ``None`` to skip
            fitting a model.
        random_state: The seed of the split and the model.
        test_size: The fraction of the data used as test set.
        preset: The hyperparameter preset of the model (``"tuned"`` or ``None``).
        model_params: Hyperparameters of the model.
        dataset_params: Parameters of synthetic datasets.

    Returns:
        The split, the fitted model, and the JSON-serializable configuration describing them.
    """
    model_params = dict(model_params or {})
    dataset_params = dict(dataset_params or {})
    split = load_dataset(dataset, **dataset_params).split(
        test_size=test_size,
        random_state=random_state,
    )
    fitted = None
    if model is not None:
        fitted = fit_model(model, split, random_state=random_state, preset=preset, **model_params)
    config = {
        "dataset": dataset,
        "dataset_params": dataset_params,
        "model": model,
        "model_params": model_params,
        "preset": preset,
        "random_state": random_state,
        "test_size": test_size,
    }
    return ConfiguredSetup(split=split, model=fitted, config=config)
