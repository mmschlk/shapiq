"""SPEX approximator for sparse higher-order interactions."""

from __future__ import annotations

from typing import Literal

from .base import Sparse, ValidSparseIndices


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
        index: ValidSparseIndices = "FBII",
        max_order: int | None = None,
        top_order: bool = False,
        random_state: int | None = None,
        decoder_type: Literal["soft", "hard"] = "soft",
        degree_parameter: int = 5,
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

            degree_parameter: A parameter that controls the maximum degree of the interactions to
                extract during execution of the algorithm. Note that this is a soft limit, and in
                practice, the algorithm may extract interactions of any degree. We typically find
                that there is little value going beyond ``5``. Defaults to ``5``. Note that
                increasing this parameter will need more ``budget`` in the :meth:`approximate`
                method.

        """
        super().__init__(
            n=n,
            max_order=max_order,
            index=index,
            top_order=top_order,
            random_state=random_state,
            transform_type="fourier",
            decoder_type=decoder_type,
            degree_parameter=degree_parameter,
        )
