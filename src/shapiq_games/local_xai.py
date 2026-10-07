"""Local feature attribution games: a model's prediction for one point with features removed."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from shapiq.game import Game
from shapiq.imputer import BaselineImputer, GenerativeConditionalImputer, MarginalImputer
from shapiq.imputer.base import Imputer
from shapiq_games._base import (
    as_bool_coalitions,
    make_predict_function,
    resolve_class_index,
    resolve_x,
)

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["LocalExplanation"]

type ImputerName = Literal["marginal", "conditional", "baseline"]


class LocalExplanation(Game):
    """The local explanation game: the prediction for ``x`` when only a coalition of features is known.

    The players are the features. Absent features are removed with an imputer from
    :mod:`shapiq.imputer`:

    - ``"marginal"`` replaces them with background rows (interventional),
    - ``"conditional"`` samples them conditionally on the present features,
    - ``"baseline"`` replaces them with a baseline value (the background mean or mode),
    - an :class:`~shapiq.imputer.TabPFNImputer` removes them from TabPFN's context
      (remove-and-recontextualize; :class:`shapiq_benchmark.setups.LocalExplanationSetup` builds
      one with ``imputer="tabpfn"``).

    The values are deterministic: the imputers are seeded and the conditional imputer is reseeded
    before every evaluation, so the value of a coalition does not depend on call order or batching.

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
        >>> game = LocalExplanation(model, data=X[:50], x=X[0], imputer="marginal")
        >>> game.n_players
        5
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
                :class:`~shapiq.imputer.base.Imputer` (then ``model``, ``data``, ``x``,
                ``class_index``, and ``sample_size`` are only used for the attributes).
            class_index: The explained class for classifiers. Defaults to ``None``, which means
                class ``1`` for classifiers (the convention of the shapiq explainers).
            sample_size: The number of background rows the marginal imputer averages over.
                Defaults to ``100``.
            random_state: The seed of the imputer. Defaults to ``42``.
            normalize: Whether to center the game such that the value of the empty coalition is
                zero. Defaults to ``True``.
            verbose: Whether to show a progress bar when evaluating the game.
        """
        data = np.asarray(data)
        self.x = resolve_x(x, data)
        self.random_state = random_state
        self.class_index: int | None = None
        if callable(model) and not hasattr(model, "predict"):
            predict: Callable[[np.ndarray], np.ndarray] = model
        else:
            self.class_index = resolve_class_index(model, class_index)
            predict = make_predict_function(model, self.class_index)

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
            self.imputer = BaselineImputer(
                model=predict, data=data, x=self.x, random_state=random_state, normalize=False
            )
        else:
            msg = (
                f"Unknown imputer {imputer!r}. Choose 'marginal', 'conditional', 'baseline', or "
                "pass an Imputer instance."
            )
            raise ValueError(msg)

        self.empty_prediction_value = float(self.imputer.empty_prediction)
        super().__init__(
            data.shape[1],
            normalize=normalize,
            normalization_value=self.empty_prediction_value,
            verbose=verbose,
        )

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        """Return the imputed prediction for the coalitions."""
        coalitions = as_bool_coalitions(coalitions)
        if hasattr(self.imputer, "_rng"):
            # the conditional imputer draws its background sample from a stateful generator;
            # reseeding makes every evaluation use the same sample
            self.imputer._rng = np.random.default_rng(self.random_state)  # noqa: SLF001
        return np.asarray(self.imputer.value_function(coalitions), dtype=float).reshape(-1)
