from shapiq.approximator.sampling import CoalitionSampler, CoalitionSamplerFast
import numpy as np
import time

n = 10
budget = 1000
sampling_weights = np.ones(n+1)

# Speed of CoalitionSampler vs CoalitionSamplerFast

start = time.time()
sampler = CoalitionSamplerFast(n, pairing_trick=False, sampling_weights=sampling_weights)
sampler.sample(budget)
end_new = time.time()
print(f"CoalitionSamplerFast time: {end_new - start:.4f} seconds")
#print(sampler.sampling_adjustment_weights)

start = time.time()
sampler = CoalitionSampler(n, pairing_trick=False, replacement=False, sampling_weights=sampling_weights)
sampler.sample(budget)
end_old = time.time()
print(f"CoalitionSampler time: {end_old - start:.4f} seconds")
#print(sampler.sampling_adjustment_weights)

