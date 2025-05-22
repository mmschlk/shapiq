from __future__ import annotations

from ._base import Sparse


class SPEX(Sparse):
    """SPEX (SParse EXplainer) via Fourier transform sampling.

    An approximator for cardinal interaction indices using Fourier transform sampling
    to efficiently compute sparse higher-order interactions.

    Args:
        n: Number of features (players).
        max_order: Maximum interaction order to consider.
        index: The Shapley interaction index type to use. Defaults to "FBII".
        top_order: If True, only reports interactions of order `max_order`.
        random_state: Seed for random number generator.
        decoder_type: Type of decoder to use, either "soft" or "hard". Defaults to "soft".
        transform_error: Error tolerance parameter for the sparse Fourier transform.
            Higher values increase accuracy but require more samples. Defaults to 5.

    References:
        .. [1] Kang, J.S., Butler, L., Agarwal. A., Erginbas, Y.E., Pedarsani, R., Ramchandran, K.,
            Yu, Bin (2025). "SPEX: Scaling Feature Interaction Explanations for LLMs"
           https://arxiv.org/abs/2502.13870
    """

    def __init__(
        self,
        n: int,
        index: str = "FBII",
        max_order: int | None = None,
        top_order: bool = False,
        random_state: int | None = None,
        decoder_type: str = "soft",
        transform_error: int = 5,
    ) -> None:
        super().__init__(
            n=n,
            max_order=max_order,
            index=index,
            top_order=top_order,
            random_state=random_state,
            transform_type="fourier",
            decoder_type=decoder_type,
            transform_error=transform_error,
        )
