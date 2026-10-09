"""Tabular local explanation games: a model's prediction for one row with features removed."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from shapiq.game import Game
from shapiq.imputer import (
    BaselineImputer,
    GaussianImputer,
    GenerativeConditionalImputer,
    MarginalImputer,
    TabPFNImputer,
)
from shapiq.imputer.base import Imputer
from shapiq_games._base import (
    as_bool_coalitions,
    predicts_row_by_row,
    resolve_predict_function,
    resolve_x,
)
from shapiq_games._tabpfn import check_inf_baseline

if TYPE_CHECKING:
    from shapiq.typing import CoalitionMatrix, FloatVector, GameValues
    from shapiq_games.typing import ImputerName, PredictFunction

__all__ = ["TabularLocalExplanation"]

# imputers whose value of a coalition does not depend on the other coalitions of the batch once
# their generator is reseeded per evaluation; the Gaussian imputers draw their samples coalition
# after coalition, and other imputers are evaluated one coalition at a time
_BATCH_SAFE_IMPUTERS = (
    MarginalImputer,
    BaselineImputer,
    GenerativeConditionalImputer,
    TabPFNImputer,
)
# imputers that draw random samples, so that an unseeded one gives different values every call
_SAMPLING_IMPUTERS = (MarginalImputer, GenerativeConditionalImputer, GaussianImputer)
_MAX_ROWS = 2**16  # the most rows passed to the model at once


class TabularLocalExplanation(Game):
    """The tabular local explanation game: the prediction for ``x`` given a coalition of features.

    The players are the features. Absent features are removed with an imputer from
    :mod:`shapiq.imputer`:

    - ``"marginal"`` replaces them with background rows (interventional),
    - ``"conditional"`` samples them conditionally on the present features,
    - ``"baseline"`` replaces them with a baseline: the background mean (the mode of non-numeric
      features; numeric category codes are averaged) or a given ``baseline``. A baseline of
      ``np.nan`` passes them as missing values to models that read those natively, e.g. XGBoost or
      scikit-learn's trees (LightGBM reads NaN as ``0`` unless it was trained with missing
      values); TabPFN also reads ``np.inf`` as missing when it is built with
      ``inference_config={"PASSTHROUGH_INF": True}`` (``tabpfn>=8.1``),
    - an :class:`~shapiq.imputer.TabPFNImputer` removes them from TabPFN's context
      (remove-and-recontextualize; :class:`shapiq_benchmark.setups.TabularLocalExplanationSetup` builds
      one with ``imputer="tabpfn"``).

    The values are deterministic: the imputers are seeded and reseeded before every evaluation, so
    the value of a coalition does not depend on call order or batching. An imputer that draws its
    samples coalition after coalition (e.g. :class:`~shapiq.imputer.GaussianImputer`) therefore
    draws them for one coalition at a time; for a tree model, the Gaussian imputers' model calls
    are batched.

    Attributes:
        x: The explained point.
        class_index: The explained class for classifiers, ``None`` for regressors or callables.
        imputer: The imputer turning the model into a game.
        empty_value: The prediction with all features absent (before centering).
        original_model_output: The prediction with all features present (before centering).

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
            imputer: ``"marginal"``, ``"conditional"``, ``"baseline"``, or an already fitted
                :class:`~shapiq.imputer.base.Imputer`. An imputer brings its own model,
                background data, point, and seed: ``x`` and ``random_state`` are then taken from
                it, and ``model`` and ``class_index`` only set :attr:`class_index`.
            class_index: The explained class for classifiers. Defaults to ``None``, which means
                class ``1`` for classifiers (the convention of the shapiq explainers).
            sample_size: The number of background rows the marginal imputer averages over.
                Defaults to ``100``.
            baseline: The values absent features take with ``imputer="baseline"``: one value
                for every feature (e.g. ``np.nan``, see above) or one per feature. ``None``
                (default) uses the mean (the mode of non-numeric features) of ``data``.
            random_state: The seed of the imputer (unless an imputer is given). Defaults to
                ``42``.
            normalize: Whether to center the game such that the value of the empty coalition is
                zero. Defaults to ``True``.
            verbose: Whether to show a progress bar when evaluating the game.

        Raises:
            ValueError: If ``baseline`` is given for another imputer, or passes ``inf`` to a
                TabPFN model that would not read it as missing, or an imputer that samples has no
                ``random_state``.
        """
        if baseline is not None and not (isinstance(imputer, str) and imputer == "baseline"):
            msg = f"baseline applies to imputer='baseline', got imputer={imputer!r}."
            raise ValueError(msg)
        data = np.asarray(data)
        predict, self.class_index = resolve_predict_function(model, class_index)
        if isinstance(imputer, Imputer):  # the imputer brings its own model, point, and seed
            if imputer.random_state is None and isinstance(imputer, _SAMPLING_IMPUTERS):
                msg = (
                    f"The {type(imputer).__name__} draws random samples: give it a random_state, "
                    "so that the game's values are deterministic."
                )
                raise ValueError(msg)
            self.imputer = imputer
            self.x = np.asarray(imputer.x).reshape(-1)
            self.random_state = imputer.random_state
        else:
            self.x = resolve_x(x, data)
            self.random_state = random_state
            self.imputer = _build_imputer(
                imputer,
                predict,
                data,
                self.x,
                model=model,
                sample_size=sample_size,
                baseline=baseline,
                random_state=random_state,
            )

        n_players = self.imputer.n_features
        # through the value function, as not every imputer sets its ``empty_prediction``
        self.empty_value = float(self.value_function(np.zeros((1, n_players), dtype=bool))[0])
        self.original_model_output = float(
            self.value_function(np.ones((1, n_players), dtype=bool))[0]
        )
        super().__init__(
            n_players,
            normalize=normalize,
            normalization_value=self.empty_value,
            verbose=verbose,
        )

    def _impute(self, coalitions: CoalitionMatrix) -> GameValues:
        # reseeding makes every evaluation draw the same samples (the conditional imputer samples
        # its background from a stateful generator); the imputers' public set_random_state would
        # redraw their background and call the model again on every evaluation
        self.imputer._rng = np.random.default_rng(self.random_state)  # noqa: SLF001
        return np.asarray(self.imputer.value_function(coalitions), dtype=float).reshape(-1)

    def value_function(self, coalitions: CoalitionMatrix) -> GameValues:
        """Return the imputed prediction for the coalitions."""
        coalitions = as_bool_coalitions(coalitions)
        if isinstance(self.imputer, _BATCH_SAFE_IMPUTERS):
            return self._impute(coalitions)
        if isinstance(self.imputer, GaussianImputer) and predicts_row_by_row(self.imputer.model):
            return self._gaussian_values(coalitions)
        return np.array([self._impute(coalition[None])[0] for coalition in coalitions])

    def _gaussian_values(self, coalitions: CoalitionMatrix) -> GameValues:
        """Evaluate a Gaussian (or Gaussian copula) imputer of a tree model, batching model calls.

        Its samples come from a generator seeded anew for every call and drawn coalition after
        coalition, so every coalition's samples are drawn alone, as the first of a batch. Only
        the model calls, which dominate the time, are batched (the imputer calls the model once
        per coalition): a tree model's output for a row does not depend on the other rows.
        """
        imputer = cast("GaussianImputer", self.imputer)
        x = np.asarray(imputer.x).reshape(-1)
        n_samples = imputer.sample_size
        step = max(1, _MAX_ROWS // n_samples)
        values = np.zeros(coalitions.shape[0])
        for start in range(0, coalitions.shape[0], step):
            chunk = coalitions[start : start + step]
            samples = np.concatenate(
                [imputer._draw_samples(x, coalition[None]) for coalition in chunk]  # noqa: SLF001
            )
            predictions = np.asarray(imputer.predict(samples.reshape(-1, x.shape[0])), dtype=float)
            values[start : start + step] = np.mean(predictions.reshape(-1, n_samples), axis=1)
        return values


def _baseline_row(baseline: float | np.ndarray, n_features: int) -> FloatVector:
    """Return the baseline as one row of ``n_features`` values."""
    values = np.asarray(baseline, dtype=float).reshape(-1)
    if values.size not in (1, n_features):
        msg = f"baseline must be one value or one per feature ({n_features}), got {values.size}."
        raise ValueError(msg)
    return np.broadcast_to(values, (1, n_features)).copy()


def _build_imputer(
    imputer: ImputerName,
    predict: PredictFunction,
    data: np.ndarray,
    x: FloatVector,
    *,
    model: Any,  # noqa: ANN401
    sample_size: int,
    baseline: float | np.ndarray | None,
    random_state: int,
) -> Imputer:
    """Build the imputer named ``imputer`` for the point ``x``."""
    if imputer == "marginal":
        return MarginalImputer(
            model=predict,
            data=data,
            x=x,
            sample_size=sample_size,
            random_state=random_state,
            normalize=False,
        )
    if imputer == "conditional":
        return GenerativeConditionalImputer(
            model=predict, data=data, x=x, random_state=random_state, normalize=False
        )
    if imputer == "baseline":
        if baseline is not None:  # a single row is the baseline itself
            data = _baseline_row(baseline, data.shape[1])
            check_inf_baseline(model, data)
        return BaselineImputer(
            model=predict, data=data, x=x, random_state=random_state, normalize=False
        )
    msg = (
        f"Unknown imputer {imputer!r}. Choose 'marginal', 'conditional', 'baseline', or pass an "
        "Imputer instance."
    )
    raise ValueError(msg)
