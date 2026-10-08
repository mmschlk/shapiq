"""The base classes of the setups and their registry."""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
import types
import typing
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Literal, NoReturn

import numpy as np
from sklearn.model_selection import train_test_split

from shapiq_benchmark.datasets import Dataset, DatasetSplit, get_dataset_spec, load_dataset
from shapiq_benchmark.models import ModelName, Preset, build_model, tuned_params
from shapiq_games.typing import Task  # noqa: TC001  (resolved by the field checks)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from shapiq import Game
    from shapiq.typing import IntVector

__all__ = [
    "RECIPE_VERSION",
    "SETUPS",
    "DatasetSetup",
    "ModelSetup",
    "Setup",
    "TabularSetup",
    "runtime_field",
    "setup_from_dict",
]

SETUPS: dict[str, type[Setup]] = {}
"""Every setup class by its name, the ``"setup"`` entry of :meth:`Setup.to_dict`."""

RECIPE_VERSION = 1
"""The version of the code every setup shares, part of every :attr:`Setup.key`.

Bump it when a change to the shared recipe changes the games the setups build for the same
fields: the datasets and their preprocessing, row sampling and splits, or the defaults of the
model registry. Cached ground truth is then not reused."""

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
        return np.asarray(value).tolist()
    msg = f"Object of type {type(value).__name__} is not JSON serializable"
    raise TypeError(msg)


def _jsonable(value: object) -> Any:  # noqa: ANN401
    return json.loads(json.dumps(value, sort_keys=True, default=_to_builtin))


class _FrozenDict(dict):
    """A read-only dict, so that a setup cannot change after its key was taken."""

    def __hash__(self) -> int:
        return hash(json.dumps(self, sort_keys=True))

    def __reduce__(self) -> tuple[type, tuple[dict]]:
        return type(self), (dict(self),)

    def _read_only(self, *_: object, **__: object) -> NoReturn:
        msg = "The fields of a setup are read-only; create a new setup instead."
        raise TypeError(msg)

    __setitem__ = __delitem__ = __ior__ = _read_only
    clear = pop = popitem = setdefault = update = _read_only


def _check_keys(value: object, name: str) -> None:
    """Reject dict keys that JSON would turn into strings (``{0: 1}`` would read back as ``{"0": 1}``)."""
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                msg = f"The dict keys of {name} must be strings, got {key!r}."
                raise TypeError(msg)
            _check_keys(item, name)
    elif isinstance(value, list | tuple):
        for item in value:
            _check_keys(item, name)


def _frozen(value: object) -> Any:  # noqa: ANN401
    """Return a JSON value with read-only dicts and tuples instead of lists."""
    if isinstance(value, dict):
        return _FrozenDict({key: _frozen(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_frozen(item) for item in value)
    return value


def _literal_options(hint: object) -> tuple[object, ...]:
    """Return the values of the ``Literal`` types in an annotation."""
    if isinstance(hint, typing.TypeAliasType):
        return _literal_options(hint.__value__)
    if typing.get_origin(hint) is Literal:
        return typing.get_args(hint)
    if typing.get_origin(hint) in (typing.Union, types.UnionType, tuple):
        return tuple(o for arg in typing.get_args(hint) for o in _literal_options(arg))
    return ()


def _matches(value: object, hint: object) -> bool:
    """Return whether a field value fits its annotation, for the annotations setups use."""
    if isinstance(hint, typing.TypeAliasType):
        return _matches(value, hint.__value__)
    origin, args = typing.get_origin(hint), typing.get_args(hint)
    if hint is type(None):
        return value is None
    if origin is Literal:
        return any(value == arg and type(value) is type(arg) for arg in args)
    if origin in (typing.Union, types.UnionType):
        return any(_matches(value, arg) for arg in args)
    if hint is float:
        return isinstance(value, int | float) and not isinstance(value, bool)
    if hint is int:
        return isinstance(value, int) and not isinstance(value, bool)
    if hint in (str, bool) or origin is dict:
        return isinstance(value, origin or hint)  # type: ignore[arg-type]
    if origin is tuple:
        if not isinstance(value, tuple):
            return False
        if len(args) == 2 and args[1] is Ellipsis:
            return all(_matches(item, args[0]) for item in value)
        return len(value) == len(args) and all(map(_matches, value, args))
    return True  # Any


def _takes_int(hint: object) -> bool:
    """Return whether an annotation accepts ints (else an int in a float field is stored as float)."""
    if isinstance(hint, typing.TypeAliasType):
        return _takes_int(hint.__value__)
    if typing.get_origin(hint) in (typing.Union, types.UnionType):
        return any(_takes_int(arg) for arg in typing.get_args(hint))
    return hint is int or hint is Any


@functools.cache
def _field_hints(cls: type) -> dict[str, Any]:
    return typing.get_type_hints(cls)


@dataclass(frozen=True, kw_only=True)
class Setup(ABC):
    """A named, typed recipe that builds a game of :mod:`shapiq_games` for a benchmark.

    A setup holds only plain values (names, numbers, seeds), so it can be stored, compared, and
    hashed. :meth:`build` turns it into the game; :attr:`key` identifies it in the ground-truth
    cache of :meth:`shapiq_benchmark.Benchmark.from_setup`. Every concrete setup registers itself
    under its ``name`` (see :data:`SETUPS` and :func:`setup_from_dict`).

    The fields are checked when the setup is created: against their annotations (including the
    choices of ``Literal`` fields), and for the dataset and model names. They are stored in their
    JSON form, read-only: dicts with string keys and tuples instead of lists, so a setup read back
    from :meth:`to_dict` equals the original.

    Subclasses declare their fields as a frozen, keyword-only dataclass and pass ``name=`` to the
    class statement; a subclass without a name of its own cannot be created, as it would share its
    parent's cache. They bump :attr:`version` when :meth:`build` changes the game it builds for
    the same fields; changes to the code all setups share bump :data:`RECIPE_VERSION`.
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
        """Store the fields in their read-only JSON form and check them against their annotations.

        Raises:
            TypeError: If the setup is not registered, or a field cannot be stored or has the
                wrong type.
            ValueError: If a field is not one of the choices of its ``Literal`` annotation.
        """
        cls = type(self)
        if SETUPS.get(getattr(cls, "name", "")) is not cls:
            msg = (
                f"{cls.__name__} is not a registered setup: give its class statement a name of "
                f"its own, e.g. `class {cls.__name__}(..., name='my_setup')`."
            )
            raise TypeError(msg)
        hints = _field_hints(cls)
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            _check_keys(value, f"{cls.__name__}.{f.name}")
            try:
                value = _frozen(_jsonable(value))
            except TypeError as error:
                msg = f"The fields of {cls.__name__} must be JSON-serializable: {error}"
                raise TypeError(msg) from error
            hint = hints[f.name]
            if type(value) is int and not _takes_int(hint):  # 0 and 0.0 are one game
                value = float(value)
            object.__setattr__(self, f.name, value)
            if not _matches(value, hint):
                options = _literal_options(hint)
                if options and isinstance(value, str | tuple):
                    msg = f"{cls.__name__}.{f.name} must be one of {options}, got {value!r}."
                    raise ValueError(msg)
                msg = f"{cls.__name__}.{f.name} must be of type {hint}, got {value!r}."
                raise TypeError(msg)

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

        It is a digest of the setup's name, :attr:`version`, :data:`RECIPE_VERSION`, and fields
        (runtime fields excluded), so it is the same across processes and machines.
        """
        params = {
            f.name: getattr(self, f.name)
            for f in dataclasses.fields(self)
            if f.metadata.get(_IN_KEY, True)
        }
        payload = {
            "setup": self.name,
            "version": self.version,
            "recipe": RECIPE_VERSION,
            "params": _jsonable(params),
        }
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
class DatasetSetup(Setup):
    """A setup on a registered dataset (see :func:`shapiq_benchmark.datasets.load_dataset`).

    Attributes:
        dataset: The dataset name.
        random_state: The seed of everything the setup draws.
        dataset_params: Parameters of synthetic datasets (e.g. ``{"n_samples": 300}``).
    """

    tasks: ClassVar[tuple[Task, ...]] = ("classification", "regression")
    """The tasks of the datasets the setup accepts."""

    dataset: str
    random_state: int = 42
    dataset_params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Check that the dataset is registered and of a task the setup accepts."""
        super().__post_init__()
        task = get_dataset_spec(self.dataset).task
        if task not in self.tasks:
            msg = (
                f"{type(self).__name__} needs a {' or '.join(self.tasks)} dataset, got "
                f"'{self.dataset}' ({task})."
            )
            raise ValueError(msg)

    def load(self) -> Dataset:
        """Load the dataset."""
        return load_dataset(self.dataset, **self.dataset_params)

    def sample_rows(self, n_rows: int, n: int | None) -> np.ndarray:
        """Return ``n`` sorted row indices out of ``n_rows``, drawn with :attr:`random_state`.

        All rows are returned if ``n`` is ``None`` or at least ``n_rows``.
        """
        if n is None or n >= n_rows:
            return np.arange(n_rows)
        rng = np.random.default_rng(self.random_state)
        return np.sort(rng.choice(n_rows, size=n, replace=False))

    def stratified_rows(self, y: np.ndarray, n: int | None, task: Task) -> IntVector:
        """Return ``n`` sorted row indices, drawn with :attr:`random_state` and stratified by class.

        The rows are stratified for classification where every class has enough rows. All rows
        are returned if ``n`` is ``None`` or at least the number of rows.
        """
        indices = np.arange(y.shape[0])
        if n is None or n >= indices.shape[0]:
            return indices
        stratify = y if task == "classification" else None
        try:
            rows, _ = train_test_split(
                indices, train_size=n, random_state=self.random_state, stratify=stratify
            )
        except ValueError:  # too few rows per class to stratify
            rows, _ = train_test_split(indices, train_size=n, random_state=self.random_state)
        return np.sort(rows)


@dataclass(frozen=True, kw_only=True)
class TabularSetup(DatasetSetup):
    """A setup on a registered dataset split into a training and a test part.

    Attributes:
        test_size: The fraction of the data used as test set. Defaults to ``0.2``.
    """

    test_size: float = 0.2

    def load_split(self) -> DatasetSplit:
        """Load the dataset and split it (seeded, stratified for classification)."""
        return self.load().split(test_size=self.test_size, random_state=self.random_state)


@dataclass(frozen=True, kw_only=True)
class ModelSetup(TabularSetup):
    """A tabular setup with a model of the registry (see :mod:`shapiq_benchmark.models`).

    Attributes:
        model: The model name (see :data:`shapiq_benchmark.models.MODEL_NAMES`).
        preset: ``"tuned"`` for the tuned hyperparameters of the model on the dataset, or
            ``None`` for the defaults of the registry.
        model_params: Hyperparameters of the model, overriding the preset.
    """

    model: ModelName = "random_forest"
    preset: Preset | None = None
    model_params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Check that a tuned preset exists."""
        super().__post_init__()
        if self.preset == "tuned":
            tuned_params(self.model, self.dataset)

    def estimator(self, task: Task, **overrides: Any) -> Any:  # noqa: ANN401
        """Return the unfitted, seeded model for a task.

        Args:
            task: ``"classification"`` or ``"regression"``.
            **overrides: Hyperparameters overriding :attr:`model_params`.
        """
        return build_model(
            self.model,
            task,
            random_state=self.random_state,
            preset=self.preset,
            dataset=self.dataset,
            **{**self.model_params, **overrides},
        )

    def fit(self, split: DatasetSplit) -> Any:  # noqa: ANN401
        """Return the seeded model fitted on the training part of ``split``."""
        return self.estimator(split.task).fit(split.x_train, split.y_train)
