"""SVARM-IQ approximation."""

from typing import Optional

from ._base import MonteCarlo


class SVARMIQ(MonteCarlo):
    """SVARM-IQ approximator uses standard form of Shapley interactions.
    SVARM-IQ utilizes MonteCarlo approximation with both stratification strategies.
    For details, refer to `Kolpaczki et al. (2024) <https://doi.org/10.48550/arXiv.2401.13371>`_.

    Args:
        n: The number of players.
        max_order: The interaction order of the approximation.
        index: The interaction index
        random_state: The random state of the estimator. Defaults to ``None``.
        pairing_trick: If ``True``, the pairing trick is applied to the sampling procedure. Defaults
            to ``False``.
        sampling_weights: An optional array of weights for the sampling procedure. The weights must
            be of shape ``(n + 1,)`` and are used to determine the probability of sampling a coalition
            of a certain size. Defaults to `None`.


    Attributes:
        n: The number of players.
        N: The set of players (starting from ``0`` to ``n - 1``).
        max_order: The interaction order of the approximation.
        min_order: The minimum order of the approximation. For the regression estimator, ``min_order``
            is equal to ``1``.
        iteration_cost: The cost of a single iteration of the regression SII.
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
    """The SVARM approximator for estimating the Shapley value (SV).

    For details, refer to `Kolpaczki et al. (2024) <https://doi.org/10.48550/arXiv.2302.00736>`_.

    Args:
        n: The number of players.
        random_state: The random state of the estimator. Defaults to ``None``.
        pairing_trick: If `True`, the pairing trick is applied to the sampling procedure. Defaults
            to ``False``.
        sampling_weights: An optional array of weights for the sampling procedure. The weights must
            be of shape ``(n + 1,)`` and are used to determine the probability of sampling a coalition
            of a certain size. Defaults to `None`.

    Attributes:
        n: The number of players.
        N: The set of players (starting from ``0`` to ``n - 1``).
        max_order: The interaction order of the approximation.
        min_order: The minimum order of the approximation. For the regression estimator, ``min_order``
            is equal to ``1``.
        iteration_cost: The cost of a single iteration of the regression SII.
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
