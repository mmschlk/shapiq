"""Linear-Gaussian games with closed-form value functions, and the structural models behind them.

Shapley-value explanations of a model are built on one of several value functions: the *marginal*
one (interventional in the feature-perturbation sense), the *conditional* (observational) one,
the *causal* one of :cite:t:`Heskes.2020`, and the expectation under an intervention on a known
structural causal model. For a linear model on Gaussian features every expectation involved is
itself linear, so all of them compose analytically. No sampling is involved, which makes the
games exact references for benchmarking approximators and for comparing the semantics on one and
the same data-generating process.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Self, get_args

import numpy as np

from shapiq.game import Game
from shapiq_games._base import as_bool_coalitions
from shapiq_games.typing import ValueFunctionMode

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import ArrayLike

    from shapiq.typing import BoolVector, CoalitionMatrix, FloatVector, GameValues, IntVector

    type Indices = Sequence[int] | IntVector
    type ChainGraph = tuple[tuple[tuple[int, ...], ...], tuple[bool, ...]]

__all__ = ["LinearGaussianGame", "LinearGaussianSCM", "RandomLinearGaussianGame"]

_MODES: tuple[str, ...] = get_args(ValueFunctionMode.__value__)


def _as_vector(values: ArrayLike, n: int, name: str) -> FloatVector:
    vector = np.asarray(values, dtype=float).reshape(-1)
    if vector.shape != (n,):
        msg = f"{name} has shape {vector.shape}, expected ({n},)."
        raise ValueError(msg)
    return vector


def _as_covariance(values: ArrayLike, n: int, name: str) -> FloatVector:
    matrix = np.asarray(values, dtype=float)
    if matrix.shape != (n, n):
        msg = f"{name} has shape {matrix.shape}, expected ({n}, {n})."
        raise ValueError(msg)
    if not np.allclose(matrix, matrix.T):
        msg = f"{name} must be symmetric."
        raise ValueError(msg)
    try:
        np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError:
        msg = f"{name} must be positive definite."
        raise ValueError(msg) from None
    return matrix


def _chain_graph(
    causal_ordering: Sequence[Sequence[int]], confounding: Sequence[bool] | None, n: int
) -> ChainGraph:
    """Return the groups and flags of a chain graph, checked to partition the ``n`` features."""
    groups = tuple(tuple(int(j) for j in group) for group in causal_ordering)
    if sorted(j for group in groups for j in group) != list(range(n)):
        msg = f"causal_ordering must partition the {n} features exactly once each, got {groups}."
        raise ValueError(msg)
    flags = (False,) * len(groups) if confounding is None else tuple(map(bool, confounding))
    if len(flags) != len(groups):
        msg = f"confounding has {len(flags)} entries but causal_ordering has {len(groups)} groups."
        raise ValueError(msg)
    return groups, flags


def _precedes(groups: Sequence[Sequence[int]], n: int) -> BoolVector:
    """Return the matrix whose entry ``[j, k]`` says that feature ``k`` is in an earlier group."""
    position = np.empty(n, dtype=int)
    for index, group in enumerate(groups):
        position[list(group)] = index
    return position[None, :] < position[:, None]


def _frozen(values: ArrayLike) -> FloatVector:
    array = np.array(values, dtype=float)
    array.setflags(write=False)
    return array


class LinearGaussianSCM:
    r"""A linear structural causal model with Gaussian noise, and the chain graph it is built on.

    The features follow the structural equations

    .. math::
        X = c + W X + \eta, \qquad \eta \sim \mathcal{N}(0, \Omega),

    where ``weights[j, k]`` is the direct effect of feature ``k`` on feature ``j`` and an
    off-diagonal entry of the noise covariance :math:`\Omega` is an unobserved common cause. The
    model comes with a causal chain graph of :cite:t:`Heskes.2020`: its features are partitioned
    into groups in causal order, effects only run from earlier to later groups, and the noise of
    two features is only correlated within a group flagged as confounded. The chain graph is then
    exact for the model: on it, the ``'causal'`` value function of :class:`~shapiq_games.LinearGaussianGame`
    equals the expectation under an intervention, ``'do'``.

    The model is not a game: :class:`~shapiq_games.LinearGaussianGame` explains a linear model on
    its features, and :class:`~shapiq_games.RandomLinearGaussianGame` draws both at random.

    Attributes:
        n_features: The number of features.
        weights: The structural weights :math:`W` of shape ``(n_features, n_features)``.
        intercepts: The intercepts :math:`c`.
        noise_cov: The noise covariance :math:`\Omega`.
        causal_ordering: The feature groups in causal order.
        confounding: One flag per group in ``causal_ordering``.
        mean: The feature mean :math:`(I - W)^{-1} c`.
        cov: The feature covariance :math:`(I - W)^{-1} \Omega (I - W)^{-\top}`.

    Examples:
        >>> scm = LinearGaussianSCM(np.array([[0.0, 0.0], [2.0, 0.0]]))  # X2 = 2 X1 + noise
        >>> scm.cov
        array([[1., 2.],
               [2., 5.]])
        >>> scm.interventional_mean(np.array([True, False]), np.array([1.0, 0.0]))
        array([1., 2.])
        >>> scm.interventional_mean(np.array([False, True]), np.array([0.0, 1.0]))
        array([0., 1.])
    """

    def __init__(
        self,
        weights: ArrayLike,
        intercepts: ArrayLike | None = None,
        noise_cov: ArrayLike | None = None,
        *,
        causal_ordering: Sequence[Sequence[int]] | None = None,
        confounding: Sequence[bool] | None = None,
    ) -> None:
        """Initialize the structural causal model.

        Args:
            weights: The structural weights of shape ``(n_features, n_features)``; ``weights[j, k]``
                is the direct effect of feature ``k`` on feature ``j``.
            intercepts: The intercepts of shape ``(n_features,)``. Defaults to ``None``, which is
                the zero vector.
            noise_cov: The noise covariance of shape ``(n_features, n_features)``, symmetric and
                positive definite. Defaults to ``None``, which is the identity.
            causal_ordering: The feature groups in causal order, as a sequence of sequences of
                feature indices that partitions all features. Defaults to ``None``, which places
                every feature in its own group, in the order of the indices.
            confounding: One boolean per group in ``causal_ordering``, marking the groups whose
                noise may be correlated. Defaults to ``None``, which is ``False`` for every group.

        Raises:
            ValueError: If the shapes disagree, if ``noise_cov`` is not symmetric positive
                definite, if ``causal_ordering`` does not partition the features, or if a weight or
                a noise correlation does not fit the chain graph.
        """
        weights = np.asarray(weights, dtype=float)
        if weights.ndim != 2 or weights.shape[0] != weights.shape[1]:
            msg = f"weights must be a square matrix, got shape {weights.shape}."
            raise ValueError(msg)
        n = weights.shape[0]
        intercepts = np.zeros(n) if intercepts is None else _as_vector(intercepts, n, "intercepts")
        noise_cov = np.eye(n) if noise_cov is None else _as_covariance(noise_cov, n, "noise_cov")
        if causal_ordering is None:
            causal_ordering = [[j] for j in range(n)]
        groups, flags = _chain_graph(causal_ordering, confounding, n)

        if np.any((weights != 0) & ~_precedes(groups, n)):
            msg = "weights may only point from an earlier group of causal_ordering to a later one."
            raise ValueError(msg)
        correlated = np.eye(n, dtype=bool)
        for group, confounded in zip(groups, flags, strict=True):
            correlated[np.ix_(group, group)] |= confounded
        if np.any((noise_cov != 0) & ~correlated):
            msg = "noise_cov may only correlate the features of a confounded group."
            raise ValueError(msg)

        transform = np.linalg.inv(np.eye(n) - weights)
        self.n_features = n
        self.weights = _frozen(weights)
        self.intercepts = _frozen(intercepts)
        self.noise_cov = _frozen(noise_cov)
        self.causal_ordering = groups
        self.confounding = flags
        self.mean = _frozen(transform @ intercepts)
        self.cov = _frozen(transform @ noise_cov @ transform.T)

    def interventional_mean(self, intervened: ArrayLike, values: ArrayLike) -> FloatVector:
        r"""Return :math:`\mathbb{E}[X \mid do(X_S = x_S)]`, the mean under an intervention.

        The intervention cuts the incoming effects of the intervened features and sets them to
        their values; the noise has mean zero either way, so :math:`\Omega` does not enter.

        Args:
            intervened: A boolean vector of shape ``(n_features,)`` marking the features :math:`S`.
            values: The values of shape ``(n_features,)``; only those of :math:`S` are used.

        Returns:
            The mean of all features under the intervention.
        """
        intervened = np.asarray(intervened, dtype=bool).reshape(-1)
        values = _as_vector(values, self.n_features, "values")
        weights = np.where(intervened[:, None], 0.0, self.weights)
        intercepts = np.where(intervened, values, self.intercepts)
        return np.linalg.solve(np.eye(self.n_features) - weights, intercepts)

    def sample(
        self, n_samples: int, *, random_state: int | np.random.Generator | None = None
    ) -> FloatVector:
        """Draw samples of the features from the structural equations.

        Args:
            n_samples: The number of samples.
            random_state: The seed or generator of the noise. Defaults to ``None``, which draws
                fresh samples on every call.

        Returns:
            The samples of shape ``(n_samples, n_features)``.
        """
        rng = np.random.default_rng(random_state)
        noise = rng.multivariate_normal(
            np.zeros(self.n_features), self.noise_cov, size=n_samples, method="cholesky"
        )
        return np.linalg.solve(
            np.eye(self.n_features) - self.weights, (self.intercepts + noise).T
        ).T

    @classmethod
    def random(
        cls,
        n: int,
        *,
        n_groups: int | None = None,
        edge_probability: float = 0.5,
        confounding_probability: float = 0.5,
        random_state: int | np.random.Generator | None = 42,
    ) -> Self:
        r"""Draw a random structural causal model.

        The features are shuffled and split into ``n_groups`` non-empty groups in causal order.
        Each feature of a preceding group is a parent of a feature with probability
        ``edge_probability``, and a group of several features is confounded with probability
        ``confounding_probability``: its features then share an unobserved common cause,
        :math:`\Omega = I + \lambda \lambda^\top` within the group. The weights, intercepts, and
        loadings :math:`\lambda` are standard normal.

        Args:
            n: The number of features.
            n_groups: The number of groups, between ``1`` and ``n``. Defaults to ``None``, which is
                ``ceil(n / 2)``, groups of two features on average.
            edge_probability: The probability that a feature of a preceding group is a parent of a
                feature. Defaults to ``0.5``.
            confounding_probability: The probability that a group of several features shares an
                unobserved common cause. Defaults to ``0.5``.
            random_state: The seed or generator. Defaults to ``42``.

        Returns:
            The drawn model.

        Raises:
            ValueError: If ``n_groups`` is not between ``1`` and ``n`` or a probability is not
                between ``0`` and ``1``.

        Examples:
            >>> scm = LinearGaussianSCM.random(8, n_groups=3, random_state=0)
            >>> len(scm.causal_ordering), scm.n_features
            (3, 8)
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
        edges = _precedes(causal_ordering, n) & (rng.random((n, n)) < edge_probability)
        weights = np.where(edges, rng.standard_normal((n, n)), 0.0)
        intercepts = rng.standard_normal(n)
        noise_cov = np.eye(n)
        for group, confounded in zip(causal_ordering, confounding, strict=True):
            if confounded:
                loadings = rng.standard_normal(len(group))
                noise_cov[np.ix_(group, group)] += np.outer(loadings, loadings)
        return cls(
            weights,
            intercepts,
            noise_cov,
            causal_ordering=causal_ordering,
            confounding=confounding,
        )


class LinearGaussianGame(Game):
    r"""A linear model on Gaussian features with marginal, conditional, causal, and do values.

    The features follow :math:`X \sim \mathcal{N}(\mu, \Sigma)`, given directly (``cov`` and
    ``mean``) or by a :class:`~shapiq_games.synthetic.LinearGaussianSCM` (``scm``), and the model
    is linear, :math:`f(x) = \beta^\top x`. For an explanation point :math:`x^*` the value
    functions of a coalition :math:`S` are

    .. math::
        :nowrap:

        \begin{eqnarray}
        v_{\text{marginal}}(S)    & = & \mathbb{E}[f(x^*_S, X_{\bar S})] \\
        v_{\text{conditional}}(S) & = & \mathbb{E}[f(X) \mid X_S = x^*_S] \\
        v_{\text{causal}}(S)      & = & \mathbb{E}[f(X) \mid do(X_S = x^*_S)]
            \text{ on the causal chain graph} \\
        v_{\text{do}}(S)          & = & \mathbb{E}[f(X) \mid do(X_S = x^*_S)]
            \text{ on the structural causal model}
        \end{eqnarray}

    The causal value function follows the causal chain graph of :cite:t:`Heskes.2020`, given by a
    ``causal_ordering`` (feature groups in causal order) and a ``confounding`` flag per group.
    The features of a group that are not intervened on are drawn conditional on every preceding
    group, and also on the intervened features of their own group unless the group is
    confounded: an intervention cuts a dependence that runs through an unobserved common cause.
    This is the rule the R package ``shapr`` implements in ``get_S_causal_steps``. It only needs
    the distribution of the features, while ``'do'`` needs the structural equations, so only a
    game built from an ``scm`` has it.

    Built from an ``scm``, the game takes the chain graph of the model, on which ``'causal'``
    equals ``'do'``; another ``causal_ordering`` or ``confounding`` gives the causal values under
    a misspecified chain graph. Built from a ``cov``, the chain graph defaults to one unconfounded
    group of all features, on which ``'causal'`` equals ``'conditional'``.

    Attributes:
        coefficients: The model coefficients :math:`\beta`.
        x_explain: The explanation point :math:`x^*`.
        scm: The structural causal model, or ``None`` for a game built from a covariance.
        mean: The feature mean :math:`\mu`.
        cov: The feature covariance :math:`\Sigma`.
        causal_ordering: The feature groups in causal order.
        confounding: One flag per group in ``causal_ordering``.
        mode: Which value function the game evaluates.
        empty_value: The value of the empty coalition, :math:`f(\mu)` (before centering).
        original_model_output: The value of the grand coalition, :math:`f(x^*)`.

    Examples:
        >>> game = LinearGaussianGame(
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
        *,
        scm: LinearGaussianSCM | None = None,
        cov: ArrayLike | None = None,
        mean: ArrayLike | None = None,
        causal_ordering: Sequence[Sequence[int]] | None = None,
        confounding: Sequence[bool] | None = None,
        mode: ValueFunctionMode = "causal",
        normalize: bool = True,
    ) -> None:
        """Initialize the LinearGaussianGame.

        Args:
            coefficients: The model coefficients of shape ``(n_features,)``.
            x_explain: The point to explain of shape ``(n_features,)``.
            scm: The structural causal model of the features. Give either ``scm`` or ``cov``.
            cov: The feature covariance matrix of shape ``(n_features, n_features)``, symmetric
                and positive definite. Give either ``scm`` or ``cov``.
            mean: The feature mean of shape ``(n_features,)``, only with ``cov``. Defaults to
                ``None``, which is the zero vector.
            causal_ordering: The feature groups in causal order, as a sequence of sequences of
                feature indices that partitions all features. Defaults to ``None``, which is the
                chain graph of the ``scm``, or one group of all features for a ``cov``. Only used
                by ``mode='causal'``.
            confounding: One boolean per group in ``causal_ordering``, marking whether the
                dependence inside that group is due to an unobserved common cause. Defaults to
                ``None``, which is the flags of the ``scm`` when ``causal_ordering`` is its own,
                and ``False`` for every group otherwise. Only used by ``mode='causal'``.
            mode: Which value function to evaluate. One of ``'marginal'``, ``'conditional'``,
                ``'causal'``, or ``'do'`` (only with an ``scm``). Defaults to ``'causal'``.
            normalize: Whether to center the game at the empty coalition. Defaults to ``True``.

        Raises:
            ValueError: If ``mode`` is not one of the value functions, if not exactly one of
                ``scm`` and ``cov`` is given, if ``mode='do'`` comes without an ``scm``, if the
                shapes of the inputs disagree, if ``cov`` is not symmetric positive definite, or
                if ``causal_ordering`` does not partition the features.
        """
        if mode not in _MODES:
            msg = f"mode must be one of {_MODES}, got {mode!r}."
            raise ValueError(msg)
        either = "Give either scm or cov; with an scm, mean and cov are taken from it."

        self.coefficients: FloatVector = np.asarray(coefficients, dtype=float).reshape(-1)
        n_players = len(self.coefficients)
        self.x_explain = _as_vector(x_explain, n_players, "x_explain")
        self.scm = scm
        if scm is None:
            if cov is None:
                raise ValueError(either)
            if mode == "do":
                msg = "mode='do' needs an scm; a covariance alone has no structural equations."
                raise ValueError(msg)
            self.cov = _as_covariance(cov, n_players, "cov")
            self.mean = np.zeros(n_players) if mean is None else _as_vector(mean, n_players, "mean")
            if causal_ordering is None:
                causal_ordering = [range(n_players)]
        else:
            if cov is not None or mean is not None:
                raise ValueError(either)
            if scm.n_features != n_players:
                msg = f"scm has {scm.n_features} features, expected {n_players} to match coefficients."
                raise ValueError(msg)
            self.cov, self.mean = scm.cov, scm.mean
            if causal_ordering is None:
                causal_ordering = scm.causal_ordering
                confounding = scm.confounding if confounding is None else confounding
        self.causal_ordering, self.confounding = _chain_graph(
            causal_ordering, confounding, n_players
        )

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
        if self.mode == "do" and self.scm is not None:
            return self.scm.interventional_mean(coalition, self.x_explain)
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
            preceding = preceding + list(group)
        return means

    def value_function(self, coalitions: CoalitionMatrix) -> GameValues:
        """Return the model's output on the expected features of each coalition."""
        coalitions = as_bool_coalitions(coalitions)
        return np.array(
            [self.coefficients @ self._feature_means(coalition) for coalition in coalitions],
            dtype=float,
        )


class RandomLinearGaussianGame(LinearGaussianGame):
    r"""A :class:`LinearGaussianGame` on a random linear structural causal model.

    The model is drawn by :meth:`LinearGaussianSCM.random
    <shapiq_games.synthetic.LinearGaussianSCM.random>`, the coefficients :math:`\beta` are standard
    normal, and the explanation point :math:`x^*` is a sample of the model. The game uses the
    chain graph of the model, so its ``'causal'`` and ``'do'`` values agree.

    Examples:
        >>> game = RandomLinearGaussianGame(8, n_groups=3, random_state=0)
        >>> len(game.scm.causal_ordering)
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
        """Initialize the RandomLinearGaussianGame.

        Args:
            n: The number of features (players).
            n_groups: The number of groups in the causal ordering, between ``1`` and ``n``.
                Defaults to ``None``, which is ``ceil(n / 2)``.
            edge_probability: The probability that a feature of a preceding group is a parent of
                a feature. Defaults to ``0.5``.
            confounding_probability: The probability that a group of several features shares an
                unobserved common cause. Defaults to ``0.5``.
            mode: Which value function to evaluate. One of ``'marginal'``, ``'conditional'``,
                ``'causal'``, or ``'do'``. Defaults to ``'causal'``.
            normalize: Whether to center the game at the empty coalition. Defaults to ``True``.
            random_state: The seed of the model, the coefficients, and the explanation point.
                Defaults to ``42``.
        """
        rng = np.random.default_rng(random_state)
        scm = LinearGaussianSCM.random(
            n,
            n_groups=n_groups,
            edge_probability=edge_probability,
            confounding_probability=confounding_probability,
            random_state=rng,
        )
        super().__init__(
            coefficients=rng.standard_normal(n),
            x_explain=scm.sample(1, random_state=rng)[0],
            scm=scm,
            mode=mode,
            normalize=normalize,
        )
