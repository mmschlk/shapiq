"""SPEX approximator for sparse higher-order interactions."""

from __future__ import annotations

from typing import Literal

from ._base import Sparse


class SPEX(Sparse):
    """SPEX (SParse EXplainer) via Fourier transform sampling.

    An approximator for cardinal interaction indices using Fourier transform sampling to efficiently
    compute sparse higher-order interactions. SPEX is presented in [Kan25]_.


    References:
        .. [Kan25] Kang, J.S., Butler, L., Agarwal. A., Erginbas, Y.E., Pedarsani, R., Ramchandran, K., Yu, Bin (2025). SPEX: Scaling Feature Interaction Explanations for LLMs https://arxiv.org/abs/2502.13870
    """

    def __init__(
        self,
        *,
        n: int,
        index: Literal["SII", "k-SII", "FBII", "FSII", "STII", "SV"] = "FBII",
        max_order: int | None = None,
        top_order: bool = False,
        random_state: int | None = None,
        decoder_type: Literal["soft", "hard"] = "soft",
        transform_tolerance: int = 5,
    ) -> None:
        """Initialize the SPEX approximator.

        Args:
            n: Number of players (features).

            max_order: Maximum interaction order to consider.

            index: The Interaction index to use. All indices supported by shapiq's
                :class:`~shapiq.game_theory.moebius_converter.MoebiusConverter` are supported.

            top_order: If ``True``, only reports interactions of exactly order ``max_order``.
                Otherwise, reports all interactions up to order ``max_order``. Defaults to
                ``False``.

            random_state: Seed for random number generator. Defaults to ``None``.

            decoder_type: Type of decoder to use, either "soft" or "hard". Defaults to "soft".

            transform_tolerance: Error tolerance parameter for the sparse Fourier transform.
                Higher values increase accuracy but require more samples. Defaults to ``5``.

        """
        super().__init__(
            n=n,
            max_order=max_order,
            index=index,
            top_order=top_order,
            random_state=random_state,
            transform_type="fourier",
            decoder_type=decoder_type,
            transform_tolerance=transform_tolerance,
        )
