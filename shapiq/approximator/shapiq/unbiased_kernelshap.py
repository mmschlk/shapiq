"""This module contains the Unbiased KernelSHAP approximation method for the Shapley value (SV).
The Unbiased KernelSHAP method is a variant of the KernelSHAP. However, it was shown that
Unbiased KernelSHAP is a more specific variant of the ShapIQ interaction method.
"""
from typing import Optional

from .shapiq import ShapIQ


class UnbiasedKernelSHAP(ShapIQ):

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

    Example:
        >>> from shapiq.games import DummyGame
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
    ):
        super().__init__(n, 1, "SV", False, random_state)
