"""Multi-output (multivariate) ProxySHAP approximator.

This module is the *multivariate* counterpart of
:class:`shapiq.approximator.proxy.proxyshap.ProxySHAP`. Whereas the scalar
``ProxySHAP`` approximates a scalar value function ``v: {0,1}^n -> R``,
:class:`MultiOutputProxySHAP` approximates a *vector-valued* value function
``v: {0,1}^n -> R^c`` (one column per output / class) and returns one
:class:`~shapiq.interaction_values.InteractionValues` per output dimension.

Route
-----
:class:`MultiOutputProxySHAP` subclasses :class:`ProxySHAP` purely to reuse its
coalition sampler (the :class:`~shapiq.approximator.sampling.CoalitionSampler`
created by the :class:`~shapiq.approximator.base.Approximator` base) and its
constructor plumbing. It overrides :meth:`approximate` entirely:

#. sample ``budget`` coalitions,
#. evaluate the *multivariate* game -> ``(n_samples, c)`` value matrix,
#. take ``baseline = coalition_values[0]`` (the empty-coalition ``c``-vector)
   and normalize the value matrix by subtracting it,
#. fit a single XGBoost ``multi_strategy="multi_output_tree"`` proxy with
   vector-valued leaves,
#. convert the proxy and run the fused
   :class:`~shapiq.approximator.proxy._multioutput.explainer.MultiOutputInterventionalTreeExplainer`,
#. return a ``list[InteractionValues]`` of length ``c``, each carrying the
   restored per-output baseline.

There is **no adjustment / residual step** -- this is the non-adjustment
ProxySHAP only. The proxy model is therefore the sole estimator and the result
is exact *for the proxy tree* (the approximation error is entirely the proxy's
fit error).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from shapiq.approximator.proxy.proxyshap import ProxySHAP
from shapiq.interaction_values import InteractionValues

from .explainer import MultiOutputInterventionalTreeExplainer

if TYPE_CHECKING:
    from collections.abc import Callable

    from shapiq.typing import FloatVector


def _build_default_multioutput_proxy(random_state: int | None) -> object:
    """Build the default vector-leaf XGBoost proxy model.

    Args:
        random_state: Random state forwarded to the ``XGBRegressor``.

    Returns:
        An unfitted ``XGBRegressor`` configured with
        ``multi_strategy="multi_output_tree"`` so that a single tree per
        boosting round carries a length-``c`` leaf vector.
    """
    try:
        # Deferred import: xgboost is an optional dependency of shapiq.
        from xgboost import XGBRegressor
    except ImportError as e:  # pragma: no cover - exercised only without xgboost
        msg = (
            "XGBoost is required for the default multi-output proxy model. "
            "Install it with 'pip install xgboost' or provide a custom proxy_model."
        )
        raise ImportError(msg) from e
    return XGBRegressor(
        multi_strategy="multi_output_tree",
        random_state=random_state,
    )


class MultiOutputProxySHAP(ProxySHAP):
    """Proxy-based approximator for *multivariate* value functions.

    Approximates a vector-valued value function ``v: {0,1}^n -> R^c`` by fitting
    a single multi-output XGBoost proxy tree (vector-valued leaves) to sampled
    coalition values and explaining that proxy exactly with the fused
    interventional multi-output tree kernel.

    No adjustment / residual correction is applied: the proxy tree is the only
    estimator.

    Example:
        >>> import numpy as np
        >>> from sklearn.datasets import make_classification
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> from shapiq.approximator.proxy._multioutput import (
        ...     MultiOutputMarginalGame,
        ...     MultiOutputProxySHAP,
        ... )
        >>> X, y = make_classification(n_samples=200, n_features=6, n_classes=3,
        ...                            n_informative=4, random_state=0)
        >>> clf = RandomForestClassifier(random_state=0).fit(X, y)
        >>> game = MultiOutputMarginalGame(clf, X, X[0])
        >>> approx = MultiOutputProxySHAP(n=6, max_order=2, index="SII", random_state=0)
        >>> values = approx.approximate(budget=64, game=game)
        >>> len(values)  # one InteractionValues per class
        3
    """

    def __init__(
        self,
        n: int,
        *,
        max_order: int = 2,
        index: str = "SII",
        proxy_model: object | None = None,
        sampling_weights: FloatVector | None = None,
        pairing_trick: bool = True,
        random_state: int | None = None,
    ) -> None:
        """Initialize the multi-output ProxySHAP approximator.

        Args:
            n: Number of features (players).
            max_order: Maximum interaction order; one of ``{1, 2, 3}``. Defaults
                to ``2``.
            index: Interaction index. Supported: ``"SV"`` (use with
                ``max_order=1``) and ``"SII"``. Defaults to ``"SII"``.
            proxy_model: Optional custom proxy model. If ``None``, a default
                ``XGBRegressor(multi_strategy="multi_output_tree")`` is used. A
                custom model **must** support vector-valued targets and be
                convertible by
                :func:`~shapiq.approximator.proxy._multioutput.tree.convert_multioutput_xgboost`.
            sampling_weights: Optional coalition-size sampling weights of shape
                ``(n + 1,)``. Defaults to ``None``.
            pairing_trick: If ``True``, the pairing trick is used in sampling.
                Defaults to ``True``.
            random_state: Random state of the estimator. Defaults to ``None``.
        """
        if proxy_model is None:
            proxy_model = _build_default_multioutput_proxy(random_state)
        # adjustment="none": MultiOutputProxySHAP is the non-adjustment variant.
        super().__init__(
            n=n,
            max_order=max_order,
            index=index,
            proxy_model=proxy_model,
            adjustment="none",
            sampling_weights=sampling_weights,
            pairing_trick=pairing_trick,
            random_state=random_state,
        )

    def approximate(  # type: ignore[override]
        self,
        budget: int,
        game: Callable[[np.ndarray], np.ndarray],
        **_: dict,
    ) -> list[InteractionValues]:
        """Approximate per-output interaction values for a multivariate game.

        Args:
            budget: Number of coalitions to sample and evaluate.
            game: A callable mapping a binary coalition matrix of shape
                ``(n_coalitions, n)`` to a ``(n_coalitions, c)`` value matrix
                (e.g. a
                :class:`~shapiq.approximator.proxy._multioutput.game.MultiOutputMarginalGame`).

        Returns:
            A ``list`` of :class:`~shapiq.interaction_values.InteractionValues`
            of length ``c`` (one per output dimension). Element ``j`` carries
            the requested ``index`` up to ``max_order`` for output ``j``, with
            ``baseline_value`` (and the ``()`` entry) restored to that output's
            empty-coalition value.
        """
        # 1. Sample coalitions and evaluate the multivariate game.
        self._sampler.sample(budget)
        coalitions_matrix = self._sampler.coalitions_matrix
        coalition_values = np.asarray(game(coalitions_matrix), dtype=np.float64)
        if coalition_values.ndim != 2:
            msg = (
                "The multivariate game must return a 2-D (n_coalitions, c) array; "
                f"got an array with ndim={coalition_values.ndim}."
            )
            raise ValueError(msg)

        # baseline = empty-coalition value (c-vector); normalize by subtracting it.
        baseline_value = coalition_values[0].copy()  # shape (c,)
        coalition_values = coalition_values - baseline_value

        # 2. Fit the multi-output proxy tree on the normalized coalition values.
        proxy_model = self.proxy_model
        proxy_model.fit(coalitions_matrix, coalition_values)  # ty: ignore[unresolved-attribute]

        # 3. Explain the proxy exactly with the fused multi-output kernel.
        explainer = MultiOutputInterventionalTreeExplainer(
            proxy_model,
            index=self.index,
            max_order=self.max_order,
        )
        proxy_values = explainer.explain()

        # 4. Restore the per-output baseline and wrap as estimated results.
        estimated = budget < 2**self.n
        results: list[InteractionValues] = []
        for j, iv in enumerate(proxy_values):
            baseline_j = float(baseline_value[j])
            interactions = dict(iv.interactions)
            interactions[()] = baseline_j
            results.append(
                InteractionValues(
                    values=interactions,
                    index=iv.index,
                    max_order=self.max_order,
                    n_players=self.n,
                    min_order=0,
                    estimated=estimated,
                    estimation_budget=budget,
                    baseline_value=baseline_j,
                    target_index=self.index,
                )
            )
        return results
