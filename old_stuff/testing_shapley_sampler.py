from __future__ import annotations

import numpy as np

from shapiq.approximator.efficient_sampling import ShapleySampler

if __name__ == "__main__":
    n = 10
    sampler = ShapleySampler(n_players=n, sampling_weights=np.ones(n + 1))
    sampler.sample(1000)
