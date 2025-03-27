from ._base import Sparse


class SPEX(Sparse):
    """SPEX (SParse EXplainer) via Fourier transform sampling.

       An approximator for cardinal interaction indices using Fourier transform sampling
       to efficiently compute sparse higher-order interactions.

       Parameters
       ----------
       n : int
           Number of features.
       max_order : int, default=2
           Maximum interaction order to consider.
       index : str
           The Shapley interaction index type to use. Defaults to "FBII"
       top_order : bool, default=False
           If True, only reports interactions of order `max_order`.
       sampling_weights : float, optional
           Weights used for sampling.
       random_state : int, optional
           Seed for random number generator.

       References
       ----------
           .. [1] Kang, J.S., Butler, L., Agarwal. A., Erginbas, Y.E., Pedarsani, R., Ramchandran, K., Yu, Bin (2025).
              "SPEX: Scaling Feature Interaction Explanations for LLMs"
              https://arxiv.org/abs/2502.13870
       """
    def __init__(
        self,
        n: int,
        index: str = "FBII",
        max_order: int | None = None,
        top_order: bool = False,
        sampling_weights: float | None = None,
        random_state: int | None = None,
        decoder_type: str = "soft",
    ):
        super().__init__(
            n,
            max_order,
            index=index,
            top_order=top_order,
            random_state=random_state,
            sampling_weights=sampling_weights,
            transform_type='fourier',
            decoder_type=decoder_type,
        )