"""SVARM-IQ approximation."""

from typing import Optional

from ._base import MonteCarlo


class SVARMIQ(MonteCarlo):
    """The SVARM-IQ[1]_ approximator for Shapley interactions.

    SVARM-IQ utilizes MonteCarlo approximation with two stratification strategies. SVARM-IQ is a
    generalization of the SVARM algorithm[2]_ and can approximate any-order Shapley interactions
    efficiently. For details about the algorithm see the original paper by Kolpaczki et al.
    (2024)[1]_.

    Args:
        n: The number of players.
        max_order: The interaction order of the approximation.
        index: The interaction index
        random_state: The random state of the estimator. Defaults to ``None``.
        pairing_trick: If ``True``, the pairing trick is applied to the sampling procedure. Defaults
            to ``False``.
        sampling_weights: An optional array of weights for the sampling procedure. The weights must
            be of shape ``(n + 1,)`` and are used to determine the probability of sampling a
            coalition of a certain size. Defaults to ``None``.

    References:
        .. [1] Kolpaczki, P., Muschalik M., Fumagalli, F., Hammer, B., and Hüllermeier, E., (2024). SVARM-IQ: Efficient Approximation of Any-order Shapley Interactions through Stratification. Proceedings of The 27th International Conference on Artificial Intelligence and Statistics, PMLR 238:3520-3528. url: https://proceedings.mlr.press/v238/kolpaczki24a

        .. [2] Kolpaczki, P., Bengs, V., Muschalik, M., & Hüllermeier, E. (2024). Approximating the Shapley Value without Marginal Contributions. Proceedings of the AAAI Conference on Artificial Intelligence, 38(12), 13246-13255. https://doi.org/10.1609/aaai.v38i12.29225
    """

    def __init__(
        self,
        n: int,
        max_order: int = 2,
        index: str = "k-SII",
        top_order: bool = False,
        pairing_trick: bool = False,
        sampling_weights: Optional[float] = None,
        random_state: Optional[int] = None,
    ):
        super().__init__(
            n,
            max_order,
            index=index,
            top_order=top_order,
            stratify_coalition_size=True,
            stratify_intersection=True,
            random_state=random_state,
            sampling_weights=sampling_weights,
            pairing_trick=pairing_trick,
        )


class SVARM(SVARMIQ):
    """The SVARM[1]_ approximator for estimating the Shapley value (SV).

    SVARM is a MonteCarlo approximation algorithm that estimates the Shapley value. For details
    about the algorithm see the original paper by Kolpaczki et al. (2024)[1]_.

    Args:
        n: The number of players.
        random_state: The random state of the estimator. Defaults to ``None``.
        pairing_trick: If `True`, the pairing trick is applied to the sampling procedure. Defaults
            to ``False``.
        sampling_weights: An optional array of weights for the sampling procedure. The weights must
            be of shape ``(n + 1,)`` and are used to determine the probability of sampling a
            coalition of a certain size. Defaults to `None`.

    References:
        .. [1] Kolpaczki, P., Bengs, V., Muschalik, M., & Hüllermeier, E. (2024). Approximating the Shapley Value without Marginal Contributions. Proceedings of the AAAI Conference on Artificial Intelligence, 38(12), 13246-13255. https://doi.org/10.1609/aaai.v38i12.29225
    """

    def __init__(
        self,
        n: int,
        random_state: Optional[int] = None,
        pairing_trick: bool = False,
        sampling_weights: Optional[float] = None,
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
