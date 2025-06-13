"""SVARM-IQ approximation."""

from __future__ import annotations

from typing import Any, Literal, get_args

from .base import MonteCarlo, ValidMonteCarloIndices


class SVARMIQ(MonteCarlo):
    """The SVARM-IQ [Kol24a]_ approximator for Shapley interactions.

    SVARM-IQ utilizes MonteCarlo approximation with two stratification strategies. SVARM-IQ is a
    generalization of the SVARM algorithm [Kol24b]_ and can approximate any-order Shapley interactions
    efficiently. For details about the algorithm see the original paper by Kolpaczki et al.
    (2024) [Kol24a]_.

    References:
        .. [Kol24a] Kolpaczki, P., Muschalik M., Fumagalli, F., Hammer, B., and Hüllermeier, E., (2024). SVARM-IQ: Efficient Approximation of Any-order Shapley Interactions through Stratification. Proceedings of The 27th International Conference on Artificial Intelligence and Statistics, PMLR 238:3520-3528. url: https://proceedings.mlr.press/v238/kolpaczki24a
        .. [Kol24b] Kolpaczki, P., Bengs, V., Muschalik, M., & Hüllermeier, E. (2024). Approximating the Shapley Value without Marginal Contributions. Proceedings of the AAAI Conference on Artificial Intelligence, 38(12), 13246-13255. https://doi.org/10.1609/aaai.v38i12.29225

    """

    def __init__(
        self,
        n: int,
        max_order: int = 2,
        index: ValidMonteCarloIndices = "k-SII",
        *,
        top_order: bool = False,
        pairing_trick: bool = False,
        sampling_weights: float | None = None,
        random_state: int | None = None,
    ) -> None:
        """Initialize the SVARMIQ approximator.

        Args:
            n: The number of players.

            max_order: The interaction order of the approximation. Defaults to ``2``.

            index: The interaction index to be used. Choose from ``['k-SII', 'SII']``. Defaults to
                ``'k-SII'``.

            top_order: If ``True``, the top-order interactions are estimated. Defaults to ``False``.

            pairing_trick: If ``True``, the pairing trick is applied to the sampling procedure.
                Defaults to ``False``.

            sampling_weights: An optional array of weights for the sampling procedure. The weights
                must be of shape ``(n + 1,)`` and are used to determine the probability of sampling
                a coalition of a certain size. Defaults to ``None``.

            random_state: The random state of the estimator. Defaults to ``None``.

        """
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


ValidIndicesSVARM = Literal["SV", "BV"]


class SVARM(SVARMIQ):
    """The SVARM [Kol24]_ approximator for estimating the Shapley value (SV).

    SVARM is a MonteCarlo approximation algorithm that estimates the Shapley value. For details
    about the algorithm see the original paper by Kolpaczki et al. (2024) [Kol24]_.

    References:
        .. [Kol24] Kolpaczki, P., Bengs, V., Muschalik, M., & Hüllermeier, E. (2024). Approximating the Shapley Value without Marginal Contributions. Proceedings of the AAAI Conference on Artificial Intelligence, 38(12), 13246-13255. https://doi.org/10.1609/aaai.v38i12.29225

    """

    valid_indices: tuple[ValidIndicesSVARM] = tuple(get_args(ValidIndicesSVARM))
    """The valid indices for the SVARM approximator."""

    def __init__(
        self,
        n: int,
        index: ValidIndicesSVARM = "SV",
        *,
        random_state: int | None = None,
        pairing_trick: bool = False,
        sampling_weights: float | None = None,
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        """Initialize the SVARM approximator.

        Args:
            n: The number of players.

            index: The interaction index to be used. Choose from ``['SV', 'BV']``. Defaults to
                ``'SV'``.

            random_state: The random state of the estimator. Defaults to ``None``.

            pairing_trick: If ``True``, the pairing trick is applied to the sampling procedure.
                Defaults to ``False``.

            sampling_weights: An optional array of weights for the sampling procedure. The weights
                must be of shape ``(n + 1,)`` and are used to determine the probability of sampling
                a coalition of a certain size. Defaults to ``None``.

            **kwargs: Additional keyword arguments (not used only for compatibility).
        """
        super().__init__(
            n,
            max_order=1,
            index=index,
            top_order=False,
            random_state=random_state,
            pairing_trick=pairing_trick,
            sampling_weights=sampling_weights,
        )
