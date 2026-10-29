"""Linear-Gaussian games with closed-form marginal, conditional, and causal value functions.

Shapley-value explanations of a model are built on one of three value functions: the *marginal*
one (interventional in the feature-perturbation sense), the *conditional* (observational) one,
and the *causal* one of :cite:t:`Heskes.2020`. For a linear model on Gaussian features every
conditional expectation involved is itself linear, so all three compose analytically. No sampling
is involved, which makes the games exact references for benchmarking approximators and for
comparing the three semantics on one and the same data-generating process.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, get_args

import numpy as np

from shapiq.game import Game
from shapiq_games._base import as_bool_coalitions
from shapiq_games.typing import ValueFunctionMode

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import ArrayLike

    from shapiq.typing import BoolVector, CoalitionMatrix, FloatVector, GameValues, IntVector

    type Indices = Sequence[int] | IntVector

__all__ = ["LinearGaussianSCM", "RandomLinearGaussianSCM"]

_MODES: tuple[str, ...] = get_args(ValueFunctionMode.__value__)


def _as_vector(values: ArrayLike, n: int, name: str) -> FloatVector:
    vector = np.asarray(values, dtype=float).reshape(-1)
    if vector.shape != (n,):
        msg = f"{name} has shape {vector.shape}, expected ({n},) to match coefficients."
        raise ValueError(msg)
    return vector


class LinearGaussianSCM(Game):
    r"""A linear model on Gaussian features with marginal, conditional, and causal value functions.

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

    The causal value function follows the causal chain graph of :cite:t:`Heskes.2020`, given by a
    ``causal_ordering`` (feature groups in causal order) and a ``confounding`` flag per group.
    The features of a group that are not intervened on are drawn conditional on every preceding
    group, and also on the intervened features of their own group unless the group is
    confounded: an intervention cuts a dependence that runs through an unobserved common cause.
    This is the rule the R package ``shapr`` implements in ``get_S_causal_steps``. Without a
    ``causal_ordering``, all features form one unconfounded group and the causal value function
    equals the conditional one.

    The three value functions depend on :math:`\beta`, :math:`x^*`, :math:`\mu`, :math:`\Sigma`,
    and the causal chain graph only, not on the structural equations behind :math:`\Sigma`.
    :class:`RandomLinearGaussianSCM` draws random instances from structural equations.

    Attributes:
        coefficients: The model coefficients :math:`\beta`.
        x_explain: The explanation point :math:`x^*`.
        mean: The feature mean :math:`\mu`.
        cov: The feature covariance :math:`\Sigma`.
        causal_ordering: The feature groups in causal order.
        confounding: One flag per group in ``causal_ordering``.
        mode: Which of the three value functions the game evaluates.
        empty_value: The value of the empty coalition, :math:`f(\mu)` (before centering).
        original_model_output: The value of the grand coalition, :math:`f(x^*)`.

    Examples:
        >>> game = LinearGaussianSCM(
        ...     coefficients=np.array([1.0, 2.0]),
        ...     x_explain=np.array([1.0, 1.0]),
        ...     cov=np.array([[1.0, 0.5], [0.5, 1.0]]),
        ...     mode="conditional",
        ...     normalize=False,
        ... )
        >>> game(np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=bool))
        array([0. , 2. , 2.5, 3. ])
    """

    def __init__(
        self,
        coefficients: ArrayLike,
        x_explain: ArrayLike,
        cov: ArrayLike,
        *,
        mean: ArrayLike | None = None,
        causal_ordering: Sequence[Sequence[int]] | None = None,
        confounding: Sequence[bool] | None = None,
        mode: ValueFunctionMode = "causal",
        normalize: bool = True,
    ) -> None:
        """Initialize the LinearGaussianSCM game.

        Args:
            coefficients: The model coefficients of shape ``(n_features,)``.
            x_explain: The point to explain of shape ``(n_features,)``.
            cov: The feature covariance matrix of shape ``(n_features, n_features)``. It must be
                symmetric and positive definite.
            mean: The feature mean of shape ``(n_features,)``. Defaults to ``None``, which is the
                zero vector.
            causal_ordering: The feature groups in causal order, as a sequence of sequences of
                feature indices that partitions all features. Defaults to ``None``, which places
                every feature in one single group. Only used by ``mode='causal'``.
            confounding: One boolean per group in ``causal_ordering``, marking whether the
                dependence inside that group is due to an unobserved common cause. Defaults to
                ``None``, which is ``False`` for every group. Only used by ``mode='causal'``.
            mode: Which value function to evaluate. One of ``'marginal'``, ``'conditional'``, or
                ``'causal'``. Defaults to ``'causal'``.
            normalize: Whether to center the game at the empty coalition. Defaults to ``True``.

        Raises:
            ValueError: If ``mode`` is not one of the three value functions, if the shapes of the
                inputs disagree, if ``cov`` is not symmetric positive definite, or if
                ``causal_ordering`` does not partition the features.
        """
        if mode not in _MODES:
            msg = f"mode must be one of {_MODES}, got {mode!r}."
            raise ValueError(msg)

        self.coefficients: FloatVector = np.asarray(coefficients, dtype=float).reshape(-1)
        n_players = len(self.coefficients)
        self.x_explain = _as_vector(x_explain, n_players, "x_explain")
        self.mean = np.zeros(n_players) if mean is None else _as_vector(mean, n_players, "mean")

        self.cov = np.asarray(cov, dtype=float)
        if self.cov.shape != (n_players, n_players):
            msg = (
                f"cov has shape {self.cov.shape}, expected ({n_players}, {n_players}) to match "
                "coefficients."
            )
            raise ValueError(msg)
        if not np.allclose(self.cov, self.cov.T):
            msg = "cov must be symmetric."
            raise ValueError(msg)
        try:
            np.linalg.cholesky(self.cov)
        except np.linalg.LinAlgError:
            msg = "cov must be positive definite."
            raise ValueError(msg) from None

        if causal_ordering is None:
            causal_ordering = [range(n_players)]
        self.causal_ordering = [[int(j) for j in group] for group in causal_ordering]
        if sorted(j for group in self.causal_ordering for j in group) != list(range(n_players)):
            msg = (
                f"causal_ordering must partition the {n_players} features exactly once each, "
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
        self.empty_value = float(self.coefficients @ self.mean)
        self.original_model_output = float(self.coefficients @ self.x_explain)
        super().__init__(
            n_players=n_players, normalize=normalize, normalization_value=self.empty_value
        )

    def _conditional_mean(
        self, target: Indices, given: Indices, x_given: FloatVector
    ) -> FloatVector:
        r"""Return :math:`\mathbb{E}[X_T \mid X_G = x_G]` of the Gaussian features."""
        if len(given) == 0:
            return self.mean[target]
        cov_tg = self.cov[np.ix_(target, given)]
        cov_gg = self.cov[np.ix_(given, given)]
        return self.mean[target] + cov_tg @ np.linalg.solve(cov_gg, x_given - self.mean[given])

    def _feature_means(self, coalition: BoolVector) -> FloatVector:
        """Return the expected features the model is evaluated on for one coalition.

        The model is linear, so its expected output is its output on the expected features.
        """
        means = np.where(coalition, self.x_explain, self.mean)
        if self.mode == "marginal" or coalition.all():
            return means

        if self.mode == "conditional":
            present, absent = np.flatnonzero(coalition), np.flatnonzero(~coalition)
            means[absent] = self._conditional_mean(absent, present, self.x_explain[present])
            return means

        # causal: walk the chain graph one group at a time; by linearity, the expected features
        # of the preceding groups stand in for the features drawn there
        preceding: list[int] = []
        for group, confounded in zip(self.causal_ordering, self.confounding, strict=True):
            intervened = [j for j in group if coalition[j]]
            generated = [j for j in group if not coalition[j]]
            if generated:
                given = preceding if confounded else preceding + intervened
                means[generated] = self._conditional_mean(generated, given, means[given])
            preceding = preceding + group
        return means

    def value_function(self, coalitions: CoalitionMatrix) -> GameValues:
        """Return the model's output on the expected features of each coalition."""
        coalitions = as_bool_coalitions(coalitions)
        return np.array(
            [self.coefficients @ self._feature_means(coalition) for coalition in coalitions],
            dtype=float,
        )


class RandomLinearGaussianSCM(LinearGaussianSCM):
    r"""A :class:`LinearGaussianSCM` drawn from a random linear structural causal model.

    The features are shuffled and split into ``n_groups`` non-empty groups in causal order. Every
    feature depends linearly on features of the preceding groups,

    .. math::
        X_j = c_j + \sum_{k \in \text{pa}(j)} w_{jk} X_k + \lambda_j U_{g(j)} + \varepsilon_j,

    where each feature of a preceding group is a parent with probability ``edge_probability``,
    and a group of several features is confounded with probability ``confounding_probability``:
    its features then share the unobserved common cause :math:`U_{g(j)}`, otherwise
    :math:`\lambda_j = 0`. The intercepts :math:`c_j`, the weights :math:`w_{jk}`, the loadings
    :math:`\lambda_j`, the unobserved causes, the noise :math:`\varepsilon_j`, and the model
    coefficients :math:`\beta` are standard normal, and the explanation point :math:`x^*` is a
    draw of :math:`X`.

    The features of a group do not affect each other, so the causal chain graph of the groups is
    exact for the drawn model: the causal value function is the interventional expectation
    :math:`\mathbb{E}[f(X) \mid do(X_S = x^*_S)]` of the structural causal model.

    Attributes:
        weights: The structural weights of shape ``(n, n)``: ``weights[j, k]`` is the direct
            effect of feature ``k`` on feature ``j``.
        intercepts: The intercepts :math:`c` of the structural equations.
        noise_cov: The covariance of the unobserved terms
            :math:`\lambda_j U_{g(j)} + \varepsilon_j`, which is not diagonal within the
            confounded groups.

    Examples:
        >>> game = RandomLinearGaussianSCM(8, n_groups=3, random_state=0)
        >>> len(game.causal_ordering)
        3
        >>> grand_coalition = np.ones((1, 8), dtype=bool)
        >>> np.allclose(game(grand_coalition), game.original_model_output - game.empty_value)
        True
    """

    def __init__(
        self,
        n: int,
        *,
        n_groups: int | None = None,
        edge_probability: float = 0.5,
        confounding_probability: float = 0.5,
        mode: ValueFunctionMode = "causal",
        normalize: bool = True,
        random_state: int | None = 42,
    ) -> None:
        """Initialize the RandomLinearGaussianSCM game.

        Args:
            n: The number of features (players).
            n_groups: The number of groups in the causal ordering, between ``1`` and ``n``.
                Defaults to ``None``, which is ``ceil(n / 2)``, groups of two features on average.
            edge_probability: The probability that a feature of a preceding group is a parent of
                a feature. Defaults to ``0.5``.
            confounding_probability: The probability that a group of several features shares an
                unobserved common cause. Defaults to ``0.5``.
            mode: Which value function to evaluate. One of ``'marginal'``, ``'conditional'``, or
                ``'causal'``. Defaults to ``'causal'``.
            normalize: Whether to center the game at the empty coalition. Defaults to ``True``.
            random_state: The seed of the model and the explanation point. Defaults to ``42``.

        Raises:
            ValueError: If ``n_groups`` is not between ``1`` and ``n`` or a probability is not
                between ``0`` and ``1``.
        """
        n_groups = (n + 1) // 2 if n_groups is None else n_groups
        if not 1 <= n_groups <= n:
            msg = f"n_groups must be between 1 and n={n}, got {n_groups}."
            raise ValueError(msg)
        for name, probability in (
            ("edge_probability", edge_probability),
            ("confounding_probability", confounding_probability),
        ):
            if not 0.0 <= probability <= 1.0:
                msg = f"{name} must be between 0 and 1, got {probability}."
                raise ValueError(msg)

        rng = np.random.default_rng(random_state)
        cuts = np.sort(rng.choice(np.arange(1, n), size=n_groups - 1, replace=False))
        causal_ordering = [sorted(int(j) for j in g) for g in np.split(rng.permutation(n), cuts)]
        confounding = [
            len(group) > 1 and bool(draw)
            for group, draw in zip(
                causal_ordering, rng.random(n_groups) < confounding_probability, strict=True
            )
        ]

        position = np.empty(n, dtype=int)
        for index, group in enumerate(causal_ordering):
            position[group] = index
        precedes = position[None, :] < position[:, None]  # [j, k]: k is in an earlier group
        edges = precedes & (rng.random((n, n)) < edge_probability)
        self.weights: FloatVector = np.where(edges, rng.standard_normal((n, n)), 0.0)
        self.intercepts: FloatVector = rng.standard_normal(n)
        self.noise_cov: FloatVector = np.eye(n)
        for group, confounded in zip(causal_ordering, confounding, strict=True):
            if confounded:
                loadings = rng.standard_normal(len(group))
                self.noise_cov[np.ix_(group, group)] += np.outer(loadings, loadings)

        transform = np.linalg.inv(np.eye(n) - self.weights)
        mean = transform @ self.intercepts
        cov = transform @ self.noise_cov @ transform.T
        super().__init__(
            coefficients=rng.standard_normal(n),
            x_explain=rng.multivariate_normal(mean, cov, method="cholesky"),
            cov=cov,
            mean=mean,
            causal_ordering=causal_ordering,
            confounding=confounding,
            mode=mode,
            normalize=normalize,
        )
