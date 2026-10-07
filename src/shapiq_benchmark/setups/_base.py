"""The base classes of the setups and their registry."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from shapiq_benchmark.datasets import Dataset, DatasetSplit, load_dataset
from shapiq_benchmark.models import build_model, fit_model

if TYPE_CHECKING:
    from collections.abc import Mapping

    from shapiq import Game

__all__ = ["SETUPS", "ModelSetup", "Setup", "TabularSetup", "runtime_field", "setup_from_dict"]

SETUPS: dict[str, type[Setup]] = {}
"""Every setup class by its name, the ``"setup"`` entry of :meth:`Setup.to_dict`."""

_IN_KEY = "in_key"


def runtime_field(default: Any) -> Any:  # noqa: ANN401
    """Declare a field that does not change the game, such as a device or a batch size.

    Runtime fields are part of :meth:`Setup.to_dict` but not of :attr:`Setup.key`, so changing them
    reuses cached ground truth.
    """
    return field(default=default, metadata={_IN_KEY: False})


def _to_builtin(value: object) -> object:
    """Convert numpy scalars and arrays to Python objects for JSON."""
    if isinstance(value, np.generic | np.ndarray):
        return value.tolist()
    msg = f"Object of type {type(value).__name__} is not JSON serializable"
    raise TypeError(msg)


def _jsonable(value: object) -> Any:  # noqa: ANN401
    return json.loads(json.dumps(value, sort_keys=True, default=_to_builtin))


@dataclass(frozen=True, kw_only=True)
class Setup(ABC):
    """A named, typed recipe that builds a game of :mod:`shapiq_games` for a benchmark.

    A setup holds only plain values (names, numbers, seeds), so it can be stored, compared, and
    hashed. :meth:`build` turns it into the game; :attr:`key` identifies it in the ground-truth
    cache of :meth:`shapiq_benchmark.Benchmark.from_setup`. Every concrete setup registers itself
    under its ``name`` (see :data:`SETUPS` and :func:`setup_from_dict`).

    Subclasses declare their fields as a frozen, keyword-only dataclass and pass ``name=`` to the
    class statement. They bump :attr:`version` when :meth:`build` changes the game it builds for
    the same fields.
    """

    name: ClassVar[str]
    """The registered name of the setup."""

    version: ClassVar[int] = 1
    """The version of :meth:`build`, part of :attr:`key`."""

    def __init_subclass__(cls, *, name: str | None = None, **kwargs: Any) -> None:
        """Register a concrete setup under ``name``."""
        super().__init_subclass__(**kwargs)
        if name is None:
            return
        if name in SETUPS:
            msg = f"A setup named {name!r} is already registered."
            raise ValueError(msg)
        cls.name = name
        SETUPS[name] = cls

    def __post_init__(self) -> None:
        """Reject fields that cannot be stored (fails here rather than at the first cache write)."""
        try:
            self.to_dict()
        except TypeError as error:
            msg = f"The fields of {type(self).__name__} must be JSON-serializable: {error}"
            raise TypeError(msg) from error

    @abstractmethod
    def build(self) -> Game:
        """Build the game."""

    def to_dict(self) -> dict[str, Any]:
        """Return the setup as a JSON-serializable dictionary (the inverse of :func:`setup_from_dict`)."""
        values = {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}
        return {"setup": self.name, **_jsonable(values)}

    @property
    def key(self) -> str:
        """A stable identifier of the game this setup builds, used as its cache key.

        It hashes the setup's name, :attr:`version`, and fields (runtime fields excluded), so it is
        the same across processes and machines.
        """
        params = {
            f.name: getattr(self, f.name)
            for f in dataclasses.fields(self)
            if f.metadata.get(_IN_KEY, True)
        }
        payload = {"setup": self.name, "version": self.version, "params": _jsonable(params)}
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8"))
        return digest.hexdigest()[:16]


def setup_from_dict(data: Mapping[str, Any]) -> Setup:
    """Build a setup from its dictionary form, e.g. one stored with :meth:`Setup.to_dict`.

    Args:
        data: The setup name under ``"setup"`` and its fields.

    Returns:
        The setup.

    Raises:
        ValueError: If the setup name is unknown.

    Examples:
        >>> setup = setup_from_dict({"setup": "knn", "dataset": "xor", "n_train": 8})
        >>> type(setup).__name__
        'KNNSetup'
        >>> setup_from_dict(setup.to_dict()) == setup
        True
    """
    params = dict(data)
    name = params.pop("setup", None)
    if name not in SETUPS:
        msg = f"Unknown setup {name!r}. Available: {', '.join(sorted(SETUPS))}."
        raise ValueError(msg)
    return SETUPS[name](**params)


@dataclass(frozen=True, kw_only=True)
class TabularSetup(Setup):
    """A setup on a registered tabular dataset (see :func:`shapiq_benchmark.datasets.load_dataset`).

    Attributes:
        dataset: The dataset name.
        random_state: The seed of the split and of everything else the setup draws.
        test_size: The fraction of the data used as test set. Defaults to ``0.2``.
        dataset_params: Parameters of synthetic datasets (e.g. ``{"n_samples": 300}``).
    """

    dataset: str
    random_state: int = 42
    test_size: float = 0.2
    dataset_params: dict[str, Any] = field(default_factory=dict)

    def load(self) -> Dataset:
        """Load the dataset."""
        return load_dataset(self.dataset, **self.dataset_params)

    def load_split(self) -> DatasetSplit:
        """Load the dataset and split it (seeded, stratified for classification)."""
        return self.load().split(test_size=self.test_size, random_state=self.random_state)

    def sample_rows(self, n_rows: int, n: int | None) -> np.ndarray:
        """Return ``n`` sorted row indices out of ``n_rows``, drawn with :attr:`random_state`.

        All rows are returned if ``n`` is ``None`` or at least ``n_rows``.
        """
        if n is None or n >= n_rows:
            return np.arange(n_rows)
        rng = np.random.default_rng(self.random_state)
        return np.sort(rng.choice(n_rows, size=n, replace=False))


@dataclass(frozen=True, kw_only=True)
class ModelSetup(TabularSetup):
    """A tabular setup with a model of the registry (see :mod:`shapiq_benchmark.models`).

    Attributes:
        model: The model name (see :data:`shapiq_benchmark.models.MODEL_NAMES`).
        preset: ``"tuned"`` for the tuned hyperparameters of the model on the dataset, or
            ``None`` for the defaults of the registry.
        model_params: Hyperparameters of the model, overriding the preset.
    """

    model: str = "random_forest"
    preset: str | None = None
    model_params: dict[str, Any] = field(default_factory=dict)

    def estimator(self, task: str) -> Any:  # noqa: ANN401
        """Return the unfitted, seeded model for a task."""
        return build_model(
            self.model,
            task,  # type: ignore[arg-type]
            random_state=self.random_state,
            preset=self.preset,
            dataset=self.dataset,
            **self.model_params,
        )

    def fit(self, split: DatasetSplit) -> Any:  # noqa: ANN401
        """Return the seeded model fitted on the training part of ``split``."""
        return fit_model(
            self.model,
            split,
            random_state=self.random_state,
            preset=self.preset,
            **self.model_params,
        )
