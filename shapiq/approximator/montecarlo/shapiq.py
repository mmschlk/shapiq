"""This module contains the shapiq estimators. Namely, the SHAPIQ and UnbiasedKernelSHAP estimators.
The Unbiased KernelSHAP method is a variant of KernelSHAP. However, it was shown that Unbiased
KernelSHAP is a more specific variant of the ShapIQ interaction method."""

from ._base import MonteCarlo


class SHAPIQ(MonteCarlo):
    """SHAP-IQ approximator for estimating Shapley interactions.

    The SHAP-IQ estimator[1]_ is a MonteCarlo approximation algorithm that estimates Shapley
    interactions. It is the default method from MonteCarlo approximator with no stratification.
    For details, see the original paper by Fumagalli et al. (2023)[1]_. SHAP-IQ can be seen as
    a generalization of the Unbiased KernelSHAP method[2]_ for any-order Shapley interactions.

    Args:
        n: The number of players.
        max_order: The interaction order of the approximation. Defaults to ``2``.
        index: The interaction index to compute.
        top_order: If ``True``, then only highest order interaction values are computed, e.g. required
            for ``'FSII'``. Defaults to ``False``.
        sampling_weights: An optional array of weights for the sampling procedure. The weights must
            be of shape ``(n + 1,)`` and are used to determine the probability of sampling a coalition.
        pairing_trick: If ``True``, the pairing trick is applied to the sampling procedure. Defaults
            to ``False``.
        random_state: The random state of the estimator. Defaults to ``None``.

    Examples:
        >>> from shapiq.games.benchmark import DummyGame
        >>> from shapiq import SHAPIQ
        >>> game = DummyGame(n=5, interaction=(1, 2))
        >>> approximator = SHAPIQ(game.n_players, max_order=2, index="k-SII")
        >>> approximator.approximate(budget=20, game=game)
        InteractionValues(
            index=k-SII, order=2, estimated=True, estimation_budget=20
        )

    See Also:
        - :class:`~shapiq.approximator.montecarlo.shapiq.UnbiasedKernelSHAP`: The Unbiased
        KernelSHAP approximator.

    References:
        .. [1] Fumagalli, F., Muschalik, M., Kolpaczki, P., Hüllermeier, E., (2023). SHAP-IQ: Unified Approximation of any-order Shapley Interactions. In Thirty-seventh Conference on Neural Information Processing Systems. url: https://openreview.net/forum?id=IEMLNF4gK4

        .. [2] Covert, I., and Lee, S.-I. (2021). Improving KernelSHAP: Practical Shapley Value Estimation via Linear Regression. In Proceedings of The 24th International Conference on Artificial Intelligence and Statistics, PMLR 130:3457-3465. url: https://proceedings.mlr.press/v130/covert21a.html
    """

    def __init__(
        self,
        n: int,
        max_order: int = 2,
        index: str = "k-SII",
        top_order: bool = False,
        sampling_weights: float | None = None,
        pairing_trick: bool = False,
        random_state: int | None = None,
    ):
        super().__init__(
            n,
            max_order,
            index=index,
            top_order=top_order,
            stratify_coalition_size=False,
            stratify_intersection=False,
            random_state=random_state,
            sampling_weights=sampling_weights,
            pairing_trick=pairing_trick,
        )


class UnbiasedKernelSHAP(SHAPIQ):
    """The Unbiased KernelSHAP approximator for estimating the Shapley value (SV).

    The Unbiased KernelSHAP estimator[1]_ is a variant of the KernelSHAP estimator (though deeply
    different). Unbiased KernelSHAP was proposed by Covert and Lee (2021)[1]_ as an unbiased
    version of KernelSHAP. In Fumagalli et al. (2023)[2]_ it was shown that Unbiased KernelSHAP is
    a more specific variant of the SHAP-IQ approximation method (Theorem 4.5).

    Args:
        n: The number of players.
        pairing_trick: If ``True``, the pairing trick is applied to the sampling procedure. Defaults
            to ``False``.
        sampling_weights: An optional array of weights for the sampling procedure. The weights must
            be of shape ``(n + 1,)`` and are used to determine the probability of sampling a coalition
            of a certain size. Defaults to ``None``.
        random_state: The random state of the estimator. Defaults to ``None``.

    Example:
        >>> from shapiq.games.benchmark import DummyGame
        >>> from shapiq.approximator import UnbiasedKernelSHAP
        >>> game = DummyGame(n=5, interaction=(1, 2))
        >>> approximator = UnbiasedKernelSHAP(n=5)
        >>> approximator.approximate(budget=20, game=game)
        InteractionValues(
            index=SV, order=1, estimated=True, estimation_budget=20,
            values={
                (0,): 0.2,
                (1,): 0.7,
                (2,): 0.7,
                (3,): 0.2,
                (4,): 0.2,
            }
        )

    See Also:
        - :class:`~shapiq.approximator.montecarlo.shapiq.SHAPIQ`: The SHAPIQ approximator.

    References:
        .. [1] Covert, I., and Lee, S.-I. (2021). Improving KernelSHAP: Practical Shapley Value Estimation via Linear Regression. In Proceedings of The 24th International Conference on Artificial Intelligence and Statistics, PMLR 130:3457-3465. url: https://proceedings.mlr.press/v130/covert21a.html

        .. [2] Fumagalli, F., Muschalik, M., Kolpaczki, P., Hüllermeier, E., (2023). SHAP-IQ: Unified Approximation of any-order Shapley Interactions. In Thirty-seventh Conference on Neural Information Processing Systems. url: https://openreview.net/forum?id=IEMLNF4gK4
    """

    def __init__(
        self,
        n: int,
        pairing_trick: bool = False,
        sampling_weights: float | None = None,
        random_state: int | None = None,
        **kwargs,
    ):
        super().__init__(
            n,
            max_order=1,
            index="SV",
            top_order=False,
            random_state=random_state,
            pairing_trick=pairing_trick,
            sampling_weights=sampling_weights,
        )
