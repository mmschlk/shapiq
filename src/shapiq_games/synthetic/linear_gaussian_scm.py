"""Synthetic linear-Gaussian SCM games with closed-form value functions.

This module provides a ground-truth benchmark for the three value functions that
Shapley-value explanations of a model are built on: the *marginal* (also called
interventional in the feature-perturbation sense), the *conditional*
(observational), and the *causal* one of Heskes et al. (2020)
:cite:t:`heskes2020`.

For a linear model on Gaussian features every conditional expectation involved is
itself linear, so all three value functions compose analytically. No sampling and
no random seed is involved, which makes the resulting games exact references for
benchmarking approximators and for comparing the three semantics against each
other on one and the same data-generating process.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from shapiq.game import Game

if TYPE_CHECKING:
    from collections.abc import Sequence

ValueFunctionMode = Literal["marginal", "conditional", "causal"]


class LinearGaussianSCM(Game):
    r"""A linear model on Gaussian features with marginal, conditional and causal value functions.

    The features follow :math:`X \sim \mathcal{N}(\mu, \Sigma)` and the model is linear,
    :math:`f(x) = \beta^\top x`. For an explanation point :math:`x^*` the three value
    functions of a coalition :math:`S` are

    .. math::
        :nowrap:

        \begin{eqnarray}
        v_{\text{marginal}}(S)    & = & \mathbb{E}[f(x^*_S, X_{\bar S})] \\
        v_{\text{conditional}}(S) & = & \mathbb{E}[f(X) \mid X_S = x^*_S] \\
        v_{\text{causal}}(S)      & = & \mathbb{E}[f(X) \mid do(X_S = x^*_S)]
        \end{eqnarray}

    The causal value function follows the causal chain graph of
    :cite:t:`heskes2020`, specified by a ``causal_ordering`` (feature groups in
    causal order) together with a ``confounding`` flag per group. Within a group,
    the features that were *not* intervened on are drawn conditional on every
    preceding group, plus the intervened features of the same group -- the latter
    only when that group is not confounded, because an intervention cuts a
    dependence that runs through an unobserved common cause. This is the rule the
    R package ``shapr`` implements in ``get_S_causal_steps``.

    Note that for the linear-Gaussian case the three value functions are fully
    determined by :math:`\beta`, :math:`x^*`, :math:`\mu`, :math:`\Sigma` and the
    causal chain graph. The structural coefficients themselves are not needed.

    Attributes:
        coefficients: The model coefficients :math:`\beta`.
        x_explain: The explanation point :math:`x^*`.
        mean: The feature mean :math:`\mu`.
        cov: The feature covariance :math:`\Sigma`.
        causal_ordering: The feature groups in causal order.
        confounding: One flag per group in ``causal_ordering``.
        mode: Which of the three value functions the game evaluates.

    Examples:
        >>> import numpy as np
        >>> game = LinearGaussianSCM(
        ...     coefficients=np.array([1.0, 2.0]),
        ...     x_explain=np.array([1.0, 1.0]),
        ...     cov=np.array([[1.0, 0.5], [0.5, 1.0]]),
        ...     mode="conditional",
        ...     normalize=False,
        ... )
        >>> coalitions = np.array([[0, 0], [1, 0], [0, 1], [1, 1]]).astype(bool)
        >>> game(coalitions)
        array([0. , 2. , 2.5, 3. ])

    """

    def __init__(
        self,
        coefficients: np.ndarray,
        x_explain: np.ndarray,
        cov: np.ndarray,
        *,
        mean: np.ndarray | None = None,
        causal_ordering: Sequence[Sequence[int]] | None = None,
        confounding: Sequence[bool] | None = None,
        mode: ValueFunctionMode = "causal",
        normalize: bool = True,
    ) -> None:
        """Initialize the LinearGaussianSCM game.

        Args:
            coefficients: The model coefficients of shape ``(n_features,)``.
            x_explain: The point to explain of shape ``(n_features,)``.
            cov: The feature covariance matrix of shape ``(n_features, n_features)``.
            mean: The feature mean of shape ``(n_features,)``. Defaults to ``None``, which
                is the zero vector.
            causal_ordering: The feature groups in causal order, as a sequence of sequences
                of feature indices that partitions all features. Defaults to ``None``, which
                places every feature in one single group. Only used by ``mode='causal'``.
            confounding: One boolean per group in ``causal_ordering``, marking whether the
                dependence inside that group is due to an unobserved common cause. Defaults
                to ``None``, which is ``False`` for every group. Only used by
                ``mode='causal'``.
            mode: Which value function to evaluate. One of ``'marginal'``, ``'conditional'``
                or ``'causal'``. Defaults to ``'causal'``.
            normalize: Whether to center the game at the empty coalition. Defaults to ``True``.

        Raises:
            ValueError: If ``mode`` is not one of the three supported value functions, if the
                shapes of the inputs disagree, or if ``causal_ordering`` does not partition
                the features.
        """
        if mode not in ("marginal", "conditional", "causal"):
            msg = f"mode must be 'marginal', 'conditional', or 'causal'; got {mode!r}."
            raise ValueError(msg)

        self.coefficients = np.asarray(coefficients, dtype=float).reshape(-1)
        self.x_explain = np.asarray(x_explain, dtype=float).reshape(-1)
        self.cov = np.asarray(cov, dtype=float)
        n_players = len(self.coefficients)

        if self.x_explain.shape != (n_players,):
            msg = (
                f"x_explain has shape {self.x_explain.shape}, expected ({n_players},) to "
                f"match coefficients."
            )
            raise ValueError(msg)
        if self.cov.shape != (n_players, n_players):
            msg = (
                f"cov has shape {self.cov.shape}, expected ({n_players}, {n_players}) to "
                f"match coefficients."
            )
            raise ValueError(msg)

        self.mean = (
            np.zeros(n_players) if mean is None else np.asarray(mean, dtype=float).reshape(-1)
        )
        if self.mean.shape != (n_players,):
            msg = (
                f"mean has shape {self.mean.shape}, expected ({n_players},) to match coefficients."
            )
            raise ValueError(msg)

        if causal_ordering is None:
            causal_ordering = [list(range(n_players))]
        self.causal_ordering = [list(group) for group in causal_ordering]
        flat = [feature for group in self.causal_ordering for feature in group]
        if sorted(flat) != list(range(n_players)):
            msg = (
                f"causal_ordering must partition the {n_players} features exactly once each; "
                f"got {self.causal_ordering}."
            )
            raise ValueError(msg)

        if confounding is None:
            confounding = [False] * len(self.causal_ordering)
        self.confounding = [bool(flag) for flag in confounding]
        if len(self.confounding) != len(self.causal_ordering):
            msg = (
                f"confounding has {len(self.confounding)} entries but causal_ordering has "
                f"{len(self.causal_ordering)} groups."
            )
            raise ValueError(msg)

        self.mode: ValueFunctionMode = mode
        self._empty_value = float(self.coefficients @ self.mean)

        super().__init__(
            n_players=n_players,
            normalize=normalize,
            normalization_value=self._empty_value,
        )

    def _conditional_mean(
        self, target: list[int], given: list[int], x_given: np.ndarray
    ) -> np.ndarray:
        """Return E[X_target | X_given = x_given] for the Gaussian features."""
        if not target:
            return np.zeros(0)
        if not given:
            return self.mean[target]
        cov_tg = self.cov[np.ix_(target, given)]
        cov_gg = self.cov[np.ix_(given, given)]
        residual = x_given - self.mean[given]
        return self.mean[target] + cov_tg @ np.linalg.solve(cov_gg, residual)

    def _feature_means(self, coalition: np.ndarray) -> np.ndarray:
        """Return the expected feature vector the model is evaluated on for one coalition."""
        in_coalition = [int(j) for j in np.flatnonzero(coalition)]
        out_coalition = [j for j in range(self.n_players) if j not in set(in_coalition)]

        means = self.mean.copy()
        means[in_coalition] = self.x_explain[in_coalition]

        if self.mode == "marginal" or not out_coalition:
            return means

        if self.mode == "conditional":
            means[out_coalition] = self._conditional_mean(
                out_coalition, in_coalition, self.x_explain[in_coalition]
            )
            return means

        # causal: walk the chain graph one group at a time
        intervened = set(in_coalition)
        preceding: list[int] = []
        for group, confounded in zip(self.causal_ordering, self.confounding, strict=True):
            group_intervened = [j for j in group if j in intervened]
            group_generated = [j for j in group if j not in intervened]
            if group_generated:
                given = list(preceding) if confounded else [*preceding, *group_intervened]
                means[group_generated] = self._conditional_mean(
                    group_generated, given, means[given]
                )
            preceding.extend(group)
        return means

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        """Evaluate the selected value function on a set of coalitions.

        Args:
            coalitions: Binary matrix of shape ``(n_coalitions, n_players)``.

        Returns:
            The coalition values of shape ``(n_coalitions,)``.
        """
        worth = np.empty(len(coalitions), dtype=float)
        for i, coalition in enumerate(coalitions):
            worth[i] = float(self.coefficients @ self._feature_means(coalition))
        return worth


class ConfoundedChainSCM(LinearGaussianSCM):
    r"""A three-feature confounded SCM on which the three value functions disagree in sign.

    The data-generating process places an unobserved common cause :math:`U` behind the
    first two features and makes the third a descendant of both:

    .. math::
        :nowrap:

        \begin{eqnarray}
        X_1 & = & U + \varepsilon_1 \\
        X_2 & = & U + \varepsilon_2 \\
        X_3 & = & 2 X_1 + X_2 + \varepsilon_3 \\
        f(x) & = & x_1 + 2 x_2 + 3 x_3
        \end{eqnarray}

    with :math:`U, \varepsilon_1, \varepsilon_2, \varepsilon_3 \sim \mathcal{N}(0, 1)`
    independent, which gives the causal chain graph ``[[0, 1], [2]]`` with
    ``confounding=[True, False]``. The explanation point is :math:`x^* = (1, 1, 1)`.

    At that point the three value functions attribute the prediction of ``6.0`` as

    ======================  =========  =========  =========
    value function          ``X_1``    ``X_2``    ``X_3``
    ======================  =========  =========  =========
    ``'marginal'``          ``1.00``   ``2.00``   ``3.00``
    ``'conditional'``       ``3.95``   ``4.38``   ``-2.33``
    ``'causal'``            ``4.00``   ``3.50``   ``-1.50``
    ======================  =========  =========  =========

    The sign of the third feature flips: against its unconditional mean of zero it is the
    largest positive contributor, while against what its parents predict for it
    (:math:`2 \cdot 1 + 1 \cdot 1 = 3`) the observed value of ``1`` falls short and drags
    the prediction down. The conditional and the causal value function then part ways on
    the second feature, because intervening on the first cuts the confounded path between
    them while conditioning on it does not.

    Examples:
        >>> import numpy as np
        >>> game = ConfoundedChainSCM(mode="causal")
        >>> coalitions = np.array([[0, 0, 0], [1, 1, 0], [1, 1, 1]]).astype(bool)
        >>> np.round(game(coalitions), 4)
        array([ 0., 12.,  6.])

    """

    COEFFICIENTS = np.array([1.0, 2.0, 3.0])
    """The model coefficients of the benchmark."""

    X_EXPLAIN = np.array([1.0, 1.0, 1.0])
    """The explanation point of the benchmark."""

    COV = np.array([[2.0, 1.0, 5.0], [1.0, 2.0, 4.0], [5.0, 4.0, 15.0]])
    """The feature covariance implied by the structural equations."""

    def __init__(self, *, mode: ValueFunctionMode = "causal", normalize: bool = True) -> None:
        """Initialize the ConfoundedChainSCM benchmark game.

        Args:
            mode: Which value function to evaluate. One of ``'marginal'``, ``'conditional'``
                or ``'causal'``. Defaults to ``'causal'``.
            normalize: Whether to center the game at the empty coalition. Defaults to ``True``.
        """
        super().__init__(
            coefficients=self.COEFFICIENTS,
            x_explain=self.X_EXPLAIN,
            cov=self.COV,
            causal_ordering=[[0, 1], [2]],
            confounding=[True, False],
            mode=mode,
            normalize=normalize,
        )
