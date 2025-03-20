from ._base import Sparse


class SMT(Sparse):
    """Sparse Mobius Transform (SMT)

       An approximator for computing the Shapley interaction indices when the Mobius transform is sparse.

       Parameters
       ----------
       n : int
           Number of features.
       max_order : int, default=2
           Maximum interaction order to consider.
       index : str
           The Shapley interaction index type to use. Defaults to "STII".
       top_order : bool, default=False
           If True, only reports interactions of order `max_order`.
       sampling_weights : float, optional
           Weights used for sampling.
       random_state : int, optional
           Seed for random number generator.

       References
       ----------
           .. [1] Kang, J.S., Erginbas, Y.E., Butler, L.,  Pedarsani, R., Ramchandran, K. (2024).
              "Learning to Understand: Identifying Interactions via the Möbius Transform"
              https://arxiv.org/abs/2402.02631
       """
    def __init__(
        self,
        n: int,
        max_order: int = 2,
        index: str = "STII",
        top_order: bool = False,
        sampling_weights: float | None = None,
        random_state: int | None = None,
    ):
        super().__init__(
            n,
            max_order,
            index=index,
            top_order=top_order,
            random_state=random_state,
            sampling_weights=sampling_weights,
            transform_type='mobius'
        )