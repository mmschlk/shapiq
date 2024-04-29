"""This module contains the shapiq estimators. Namely, the SHAPIQ and UnbiasedKernelSHAP estimators.
The Unbiased KernelSHAP method is a variant of KernelSHAP. However, it was shown that Unbiased
KernelSHAP is a more specific variant of the ShapIQ interaction method (cf.
https://proceedings.neurips.cc/paper_files/paper/2023/hash/264f2e10479c9370972847e96107db7f-Abstract-Conference.html)."""

from typing import Optional

from ._base import MonteCarlo


class SHAPIQ(MonteCarlo):
    """SHAP-IQ approximator uses standard form of Shapley interactions.
    Algorithm described in https://proceedings.neurips.cc/paper_files/paper/2023/hash/264f2e10479c9370972847e96107db7f-Abstract-Conference.html.
    This is the default method from MonteCarlo approximator with no stratification.

    Args:
        n: The number of players.
        max_order: The interaction order of the approximation.
        index: The interaction index
        random_state: The random state of the estimator. Defaults to `None`.
        pairing_trick: If `True`, the pairing trick is applied to the sampling procedure. Defaults
            to `False`.
        sampling_weights: An optional array of weights for the sampling procedure. The weights must
            be of shape `(n + 1,)` and are used to determine the probability of sampling a coalition

    Attributes:
        n: The number of players.
        N: The set of players (starting from 0 to n - 1).
        max_order: The interaction order of the approximation.
        min_order: The minimum order of the approximation. For the regression estimator, min_order
            is equal to 1.
        iteration_cost: The cost of a single iteration of the regression SII.
    """

    def __init__(
        self,
        n: int,
        max_order: int = 2,
        index: str = "k-SII",
        top_order: bool = False,
        sampling_weights: Optional[float] = None,
        pairing_trick: bool = False,
        random_state: Optional[int] = None,
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

    The Unbiased KernelSHAP estimator is a variant of the KernelSHAP estimator (though deeply
    different). Unbiased KernelSHAP was proposed in Covert and Lee's
    [original paper](http://proceedings.mlr.press/v130/covert21a/covert21a.pdf) as an unbiased
    version of KernelSHAP. Recently, in Fumagalli et al.'s
    [paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/264f2e10479c9370972847e96107db7f-Abstract-Conference.html),
    it was shown that Unbiased KernelSHAP is a more specific variant of the ShapIQ approximation
    method (Theorem 4.5).

    Args:
        n: The number of players.
        random_state: The random state of the estimator. Defaults to `None`.
        pairing_trick: If `True`, the pairing trick is applied to the sampling procedure. Defaults
            to `False`.
        sampling_weights: An optional array of weights for the sampling procedure. The weights must
            be of shape `(n + 1,)` and are used to determine the probability of sampling a coalition
            of a certain size. Defaults to `None`.

    Example:
        >>> from shapiq.games.benchmark import DummyGame
        >>> from shapiq.approximator import UnbiasedKernelSHAP
        >>> game = DummyGame(n=5, interaction=(1, 2))
        >>> approximator = UnbiasedKernelSHAP(n=5)
        >>> approximator.approximate(budget=100, game=game)
        InteractionValues(
            index=SV, order=1, estimated=False, estimation_budget=32,
            values={
                (0,): 0.2,
                (1,): 0.7,
                (2,): 0.7,
                (3,): 0.2,
                (4,): 0.2,
            }
        )
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
