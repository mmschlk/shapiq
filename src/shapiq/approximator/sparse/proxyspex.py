"""ProxySPEX approximator for sparse higher-order interactions."""

from __future__ import annotations

from .base import Sparse, ValidSparseIndices


class ProxySPEX(Sparse):
    """ProxySPEX (SParse EXplainer) via Fourier transform sampling.

    An approximator for cardinal interaction indices using Fourier transform sampling to efficiently
    compute sparse higher-order interactions. ProxySPEX is presented in [But25]_.


    References:
        .. [But25] Butler, L., Kang, J.S., Agarwal. A., Erginbas, Y.E., Yu, Bin, Ramchandran, K. (2025). ProxySPEX: Inference-Efficient Interpretability via Sparse Feature Interactions in LLMs https://arxiv.org/pdf/2505.17495
    """

    def __init__(
        self,
        *,
        n: int,
        index: ValidSparseIndices = "FBII",
        max_order: int | None = None,
        top_order: bool = False,
        random_state: int | None = None,
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


        """
        try:
            import lightgbm as lgb  # noqa: F401
        except ImportError as err:
            msg = (
                "The 'lightgbm' package is required when decoder_type is 'proxyspex' "
                "but it is not installed. Please see the installation instructions at "
                "https://github.com/microsoft/LightGBM/tree/master/python-package"
            )
            raise ImportError(msg) from err
        super().__init__(
            n=n,
            max_order=max_order,
            index=index,
            top_order=top_order,
            random_state=random_state,
            transform_type="fourier",
            decoder_type="proxyspex",
            degree_parameter=n,
        )
