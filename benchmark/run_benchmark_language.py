"""This script runs the benchmark for the language model as an example."""

from shapiq.approximator import (
    SHAPIQ,
    SVARMIQ,
    InconsistentKernelSHAPIQ,
    KernelSHAPIQ,
    PermutationSamplingSII,
)

if __name__ == "__main__":

    index = "k-SII"
    order = 2

    n_players = 14

    # get approximators
    approximators = [
        KernelSHAPIQ(n=n_players, index=index, max_order=order),
        InconsistentKernelSHAPIQ(n=n_players, index=index, max_order=order),
        SHAPIQ(n=n_players, index=index, max_order=order),
        SVARMIQ(n=n_players, index=index, max_order=order),
        PermutationSamplingSII(n=n_players, index=index, max_order=order),
    ]
