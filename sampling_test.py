from shapiq.approximator.sampling import CoalitionSampler, CoalitionSamplerFast
import numpy as np

n = 100
sampling_weights = np.ones(n+1)
sampler = CoalitionSamplerFast(n, pairing_trick=False, sampling_weights=sampling_weights)

sampler.sample(1000)
print(sampler.coalitions_matrix)
print(sampler.sampled_coalitions_dict)
print(sampler.adjusted_sampling_weights)
