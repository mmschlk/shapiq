"""Tabular local explanation games: a model's prediction for one row with features removed."""

from __future__ import annotations

import importlib.metadata
import re
from typing import TYPE_CHECKING, Any

import numpy as np

from shapiq.game import Game
from shapiq.imputer import (
    BaselineImputer,
    GenerativeConditionalImputer,
    MarginalImputer,
    TabPFNImputer,
)
from shapiq.imputer.base import Imputer
from shapiq.utils.modules import safe_isinstance
from shapiq_games._base import (
    as_bool_coalitions,
    make_predict_function,
    resolve_class_index,
    resolve_x,
)

if TYPE_CHECKING:
    from shapiq.typing import CoalitionMatrix, FloatVector, GameValues
    from shapiq_games.typing import ImputerName, PredictFunction

__all__ = ["TabularLocalExplanation", "require_inf_passthrough"]

_PASSTHROUGH_INF_TABPFN = (8, 1)  # the first tabpfn release with inference_config PASSTHROUGH_INF

# imputers whose value of a coalition does not depend on the other coalitions of the batch once
# their generator is reseeded per evaluation; others (e.g. the Gaussian imputers, which draw
# samples coalition after coalition) are evaluated one coalition at a time
_BATCH_SAFE_IMPUTERS = (
    MarginalImputer,
    BaselineImputer,
    GenerativeConditionalImputer,
    TabPFNImputer,
)


class TabularLocalExplanation(Game):
    """The tabular local explanation game: the prediction for ``x`` given a coalition of features.

    The players are the features. Absent features are removed with an imputer from
    :mod:`shapiq.imputer`:

    - ``"marginal"`` replaces them with background rows (interventional),
    - ``"conditional"`` samples them conditionally on the present features,
    - ``"baseline"`` replaces them with a baseline: the background mean (mode for categorical
      features) or a given ``baseline``. A baseline of ``np.nan`` passes them as missing values to
      models that read those natively, e.g. XGBoost, LightGBM, or scikit-learn's histogram
      gradient boosting; TabPFN also reads ``np.inf`` as missing when it is built with
      ``inference_config={"PASSTHROUGH_INF": True}`` (``tabpfn>=8.1``),
    - an :class:`~shapiq.imputer.TabPFNImputer` removes them from TabPFN's context
      (remove-and-recontextualize; :class:`shapiq_benchmark.setups.TabularLocalExplanationSetup` builds
      one with ``imputer="tabpfn"``).

    The values are deterministic: the imputers are seeded and reseeded before every evaluation, so
    the value of a coalition does not depend on call order or batching. An imputer that draws its
    samples coalition after coalition (e.g. :class:`~shapiq.imputer.GaussianImputer`) is therefore
    evaluated one coalition at a time.

    Attributes:
        x: The explained point.
        class_index: The explained class for classifiers, ``None`` for regressors or callables.
        imputer: The imputer turning the model into a game.
        empty_prediction_value: The prediction with all features absent.

    Examples:
        >>> from sklearn.datasets import make_regression
        >>> X, y = make_regression(n_samples=200, n_features=5, random_state=0)
        >>> from sklearn.ensemble import RandomForestRegressor
        >>> model = RandomForestRegressor(n_estimators=10, random_state=0).fit(X, y)
        >>> game = TabularLocalExplanation(model, data=X[:50], x=X[0], imputer="marginal")
        >>> game.n_players
        5

        Absent features passed as missing values (NaN):

        >>> from sklearn.ensemble import HistGradientBoostingRegressor
        >>> booster = HistGradientBoostingRegressor(max_iter=20, random_state=0).fit(X, y)
        >>> game = TabularLocalExplanation(booster, data=X, x=0, imputer="baseline", baseline=np.nan)
        >>> game.n_players
        5
        >>> # TabPFN: TabularLocalExplanation(tabpfn, X, x=0, imputer="baseline", baseline=np.inf)
    """

    def __init__(
        self,
        model: Any,  # noqa: ANN401
        data: np.ndarray,
        x: int | np.ndarray = 0,
        *,
        imputer: ImputerName | Imputer = "marginal",
        class_index: int | None = None,
        sample_size: int = 100,
        baseline: float | np.ndarray | None = None,
        random_state: int = 42,
        normalize: bool = True,
        verbose: bool = False,
    ) -> None:
        """Initialize the local explanation game.

        Args:
            model: A fitted scikit-learn compatible model, or a callable mapping a
                ``(n_samples, n_features)`` matrix to ``(n_samples,)`` predictions.
            data: The background data of shape ``(n_samples, n_features)``.
            x: The explained point, or its index in ``data``. Defaults to ``0``.
            imputer: ``"marginal"``, ``"conditional"``, ``"baseline"``, or an already fitted :class:`~shapiq.imputer.base.Imputer`. An imputer brings its own model,
                background data, point, and seed: ``x`` and ``random_state`` are then taken from
                it, and ``model`` and ``class_index`` only set :attr:`class_index`.
            class_index: The explained class for classifiers. Defaults to ``None``, which means
                class ``1`` for classifiers (the convention of the shapiq explainers).
            sample_size: The number of background rows the marginal imputer averages over.
                Defaults to ``100``.
            baseline: The values absent features take with ``imputer="baseline"``: one value
                for every feature (e.g. ``np.nan``, see above) or one per feature. ``None``
                (default) uses the mean (mode for categorical features) of ``data``.
            random_state: The seed of the imputer (unless an imputer is given). Defaults to
                ``42``.
            normalize: Whether to center the game such that the value of the empty coalition is
                zero. Defaults to ``True``.
            verbose: Whether to show a progress bar when evaluating the game.

        Raises:
            ValueError: If ``baseline`` is given for another imputer, or passes ``inf`` to a
                TabPFN model that would not read it as missing.
        """
        data = np.asarray(data)
        if isinstance(imputer, Imputer):  # the imputer brings its own point and seed
            self.x = np.asarray(imputer.x).reshape(-1)
            self.random_state = imputer.random_state
        else:
            self.x = resolve_x(x, data)
            self.random_state = random_state
        self.class_index: int | None = None
        if callable(model) and not hasattr(model, "predict"):
            predict: PredictFunction = model
        else:
            self.class_index = resolve_class_index(model, class_index)
            predict = make_predict_function(model, self.class_index)

        if baseline is not None and not (isinstance(imputer, str) and imputer == "baseline"):
            msg = f"baseline applies to imputer='baseline', got imputer={imputer!r}."
            raise ValueError(msg)
        if isinstance(imputer, Imputer):
            self.imputer = imputer
        elif imputer == "marginal":
            self.imputer = MarginalImputer(
                model=predict,
                data=data,
                x=self.x,
                sample_size=sample_size,
                random_state=random_state,
                normalize=False,
            )
        elif imputer == "conditional":
            self.imputer = GenerativeConditionalImputer(
                model=predict, data=data, x=self.x, random_state=random_state, normalize=False
            )
        elif imputer == "baseline":
            if baseline is not None:  # a single row is the baseline itself
                data = _baseline_row(baseline, data.shape[1])
                _check_tabpfn_baseline(model, data)
            self.imputer = BaselineImputer(
                model=predict, data=data, x=self.x, random_state=random_state, normalize=False
            )
        else:
            msg = (
                f"Unknown imputer {imputer!r}. Choose 'marginal', 'conditional', 'baseline', or "
                "pass an Imputer instance."
            )
            raise ValueError(msg)

        n_players = self.imputer.n_features
        # through the value function, as not every imputer sets its ``empty_prediction``
        self.empty_prediction_value = float(
            self.value_function(np.zeros((1, n_players), dtype=bool))[0]
        )
        super().__init__(
            n_players,
            normalize=normalize,
            normalization_value=self.empty_prediction_value,
            verbose=verbose,
        )

    def _impute(self, coalitions: CoalitionMatrix) -> GameValues:
        # reseeding makes every evaluation draw the same samples (the conditional imputer samples
        # its background from a stateful generator)
        self.imputer._rng = np.random.default_rng(self.random_state)  # noqa: SLF001
        return np.asarray(self.imputer.value_function(coalitions), dtype=float).reshape(-1)

    def value_function(self, coalitions: CoalitionMatrix) -> GameValues:
        """Return the imputed prediction for the coalitions."""
        coalitions = as_bool_coalitions(coalitions)
        if isinstance(self.imputer, _BATCH_SAFE_IMPUTERS):
            return self._impute(coalitions)
        return np.array([self._impute(coalition[None])[0] for coalition in coalitions])


def require_inf_passthrough() -> None:
    """Raise unless the installed TabPFN reads ``inf`` as a missing value (``tabpfn>=8.1``).

    Raises:
        ValueError: If the installed ``tabpfn`` is older than 8.1.
    """
    installed = importlib.metadata.version("tabpfn")
    version = tuple(int(part) for part in re.findall(r"\d+", installed)[:2])
    if version < _PASSTHROUGH_INF_TABPFN:
        msg = (
            "TabPFN reads inf as a missing value only from tabpfn 8.1 on (built with "
            f"inference_config={{'PASSTHROUGH_INF': True}}); installed: tabpfn {installed}."
        )
        raise ValueError(msg)


def _baseline_row(baseline: float | np.ndarray, n_features: int) -> FloatVector:
    """Return the baseline as one row of ``n_features`` values."""
    values = np.asarray(baseline, dtype=float).reshape(-1)
    if values.size not in (1, n_features):
        msg = f"baseline must be one value or one per feature ({n_features}), got {values.size}."
        raise ValueError(msg)
    return np.broadcast_to(values, (1, n_features)).copy()


def _check_tabpfn_baseline(model: Any, baseline: np.ndarray) -> None:  # noqa: ANN401
    """Reject an ``inf`` baseline for a TabPFN model that does not read it as a missing value.

    Without ``PASSTHROUGH_INF``, tabpfn 8.1 and later reject ``inf``, and older releases transform it
    in their preprocessing, so the game would silently explain something else.
    """
    if not np.isinf(baseline).any() or not safe_isinstance(
        model, ["tabpfn.TabPFNClassifier", "tabpfn.TabPFNRegressor"]
    ):
        return
    require_inf_passthrough()
    config = getattr(model, "inference_config", None) or {}
    if isinstance(config, dict):
        passthrough = config.get("PASSTHROUGH_INF")
    else:
        passthrough = getattr(config, "PASSTHROUGH_INF", None)
    if not passthrough:
        msg = "Build the TabPFN model with inference_config={'PASSTHROUGH_INF': True}."
        raise ValueError(msg)
