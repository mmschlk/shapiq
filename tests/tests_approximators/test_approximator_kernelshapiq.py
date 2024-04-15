import numpy as np

from shapiq.approximator.kernelshapiq import KernelSHAPIQ
from shapiq.games.soum import SOUM


def test_approximator_kernelshapiq():
    n = np.random.randint(low=2, high=10)
    order = 2
    n_basis_games = np.random.randint(low=1, high=100)
    soum = SOUM(n, n_basis_games=n_basis_games)

    predicted_value = soum(np.ones(n))[0]

    # Compute via sparse Möbius representation
    # moebius_converter = MoebiusConverter(N, soum.moebius_coefficients)
    # sii = moebius_converter("SII", order)

    budget = 2**n
    kernelshapiq = KernelSHAPIQ(n=n, order=order, index="SII")
    sii_approximated = kernelshapiq.approximate(budget=budget, game=soum)
    assert (
        np.sum(sii_approximated.values[:n]) + sii_approximated.baseline_value
    ) - predicted_value < 10e-7
