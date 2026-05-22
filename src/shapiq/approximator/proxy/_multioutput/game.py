"""Multivariate (vector-valued) value function for the multi-output ProxySHAP experiment.

This module provides a self-contained marginal (interventional) imputation value
function for a multiclass classifier. Given a fitted classifier with a
``predict_proba`` method, a background dataset and a single instance ``x`` to
explain, :class:`MultiOutputMarginalGame` produces, for any coalition matrix, a
``(n_coalitions, c)`` array where column ``j`` is the value of output class ``j``.

For a coalition ``S`` the value is the *interventional* expectation

    v(S)[j] = E_background[ predict_proba(z)[j] ]

where ``z`` agrees with ``x`` on the features in ``S`` and with a background
sample on the features outside ``S``. The expectation is estimated by averaging
over the background dataset.

It is deliberately *not* routed through shapiq's scalar :class:`shapiq.game.Game`
or imputer classes: those return a scalar per coalition, whereas the multi-output
proxy needs the full per-class vector.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


class MultiOutputMarginalGame:
    """Marginal (interventional) multivariate value function for a multiclass model.

    The callable instance maps a binary coalition matrix of shape
    ``(n_coalitions, n_features)`` to a ``(n_coalitions, n_outputs)`` array of
    per-class value-function outputs.

    Attributes:
        n_players: Number of features ``n``.
        n_outputs: Number of classes / outputs ``c``.
        x: The instance being explained (length-``n`` array).
        background: The background dataset used for the interventional baseline.
    """

    def __init__(
        self,
        model: Any,  # noqa: ANN401
        background_data: NDArray[np.floating],
        x: NDArray[np.floating],
        *,
        max_background_samples: int | None = 100,
        random_state: int | None = None,
    ) -> None:
        """Initialize the multivariate marginal game.

        Args:
            model: A fitted multiclass classifier exposing ``predict_proba``.
            background_data: A ``(n_background, n_features)`` array of background
                samples defining the interventional baseline.
            x: The instance to explain, a length-``n_features`` array.
            max_background_samples: If not ``None``, subsample the background
                dataset to at most this many rows (for speed). Defaults to
                ``100``.
            random_state: Random state for the background subsampling. Defaults
                to ``None``.
        """
        if not hasattr(model, "predict_proba"):
            msg = "model must be a classifier exposing a predict_proba method."
            raise TypeError(msg)

        self._model = model
        self.x: NDArray[np.float64] = np.asarray(x, dtype=np.float64).reshape(-1)
        self.n_players: int = int(self.x.shape[0])

        background = np.asarray(background_data, dtype=np.float64)
        if background.ndim != 2 or background.shape[1] != self.n_players:
            msg = (
                "background_data must be 2-D with the same number of features as x "
                f"({self.n_players}); got shape {background.shape}."
            )
            raise ValueError(msg)

        rng = np.random.default_rng(random_state)
        if max_background_samples is not None and background.shape[0] > max_background_samples:
            idx = rng.choice(background.shape[0], size=max_background_samples, replace=False)
            background = background[idx]
        self.background: NDArray[np.float64] = background

        # Determine the output dimensionality c from a single probe prediction.
        probe = self._model.predict_proba(self.x.reshape(1, -1))
        self.n_outputs: int = int(np.asarray(probe).shape[1])

    @property
    def normalization_value(self) -> NDArray[np.float64]:
        """Per-class value of the empty coalition (the interventional baseline)."""
        return self._value(np.zeros((1, self.n_players), dtype=np.int64))[0]

    def grand_coalition_value(self) -> NDArray[np.float64]:
        """Per-class value of the grand coalition (all features set to ``x``).

        With every feature fixed to ``x`` the imputed samples are all equal to
        ``x``, so this equals ``predict_proba(x)``.

        Returns:
            A length-``n_outputs`` array.
        """
        return self._value(np.ones((1, self.n_players), dtype=np.int64))[0]

    def _value(self, coalitions: NDArray[np.integer]) -> NDArray[np.float64]:
        """Evaluate the value function for a batch of coalitions.

        Args:
            coalitions: A ``(n_coalitions, n_features)`` binary matrix.

        Returns:
            A ``(n_coalitions, n_outputs)`` array of per-class values.
        """
        coalitions = np.asarray(coalitions)
        n_coalitions = coalitions.shape[0]

        out = np.empty((n_coalitions, self.n_outputs), dtype=np.float64)
        for i in range(n_coalitions):
            mask = coalitions[i].astype(bool)
            # Imputed batch: start from the background samples, overwrite the
            # in-coalition columns with x's values.
            imputed = self.background.copy()
            imputed[:, mask] = self.x[mask]
            proba = np.asarray(self._model.predict_proba(imputed), dtype=np.float64)
            out[i] = proba.mean(axis=0)
        return out

    def __call__(self, coalitions: NDArray[np.integer]) -> NDArray[np.float64]:
        """Evaluate the multivariate value function.

        Args:
            coalitions: A ``(n_coalitions, n_features)`` binary matrix where a
                ``1`` marks a feature fixed to ``x`` and a ``0`` a feature drawn
                from the background data.

        Returns:
            A ``(n_coalitions, n_outputs)`` array of per-class value-function
            outputs.
        """
        return self._value(coalitions)
