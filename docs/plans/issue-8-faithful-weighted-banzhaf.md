# Issue 8 — Faithful weighted Banzhaf and the FIxLIP path

Status: **not started**

## Goal

Bring the weighted Banzhaf family (shipped exactly as `WeightedBV(p)` / `WeightedBII(p, order)`,
cardinal capability, Marichal & Mathonet, arXiv:1001.3052) into the sampled world, as the
foundation for FIxLIP-style explanations (arXiv:2508.05430), which make heavy use of weighted
Banzhaf interactions.

## Mathematical basis

Marichal-Mathonet define the weighted Banzhaf interaction index through weighted least squares
under the product measure ``w(T) = p^|T| (1-p)^(n-|T|)`` (players join independently with
probability ``p``; ``0 < p < 1``). Their Proposition 3 is the load-bearing fact for a faithful
estimator: the best ``k``-additive approximation under that measure **preserves the weighted
Banzhaf interaction index for every coalition of size <= k** — the sampled fit estimates exactly
the index the exact cardinal path computes.

## Work breakdown

- [ ] `WeightedBanzhafKernelSampler`: generalize `BanzhafKernelSampler`
  (`src/shapiq/sampling/_kernel.py`) from `bernoulli(0.5)` to `bernoulli(p)` — the
  sample-proportional-to-kernel architecture then gives unit-weight rows for the product
  measure. Paired complements need care: under `p != 1/2` the complement is *not*
  equally likely, so pairing must either resample or reweight — decide before building.
- [ ] Faithful solve: the unconstrained free-intercept fit (`Regression`'s FBII branch)
  carries over; only the sampler differs. `min_budget` arithmetic is unchanged.
- [ ] Exact faithful check: extend the FBII dedicated solver in
  `src/shapiq/explainers/_exact.py` to accept the product-measure weights (row weights
  ``sqrt(w)``), or verify Proposition 3 numerically against the shipped cardinal
  `WeightedBII` and skip a dedicated exact solver entirely.
- [ ] Per-player probabilities ``p_i`` (the general Marichal-Mathonet form) are
  player-specific, not cardinal — they arrive with the deferred player-specific
  capability, not this issue.

## Acceptance criteria

- Sampled faithful weighted Banzhaf converges to exact `WeightedBII` values on k-additive
  games (Proposition 3), with the FBII-style identification gate.
- `p = 0.5` reproduces the existing FBII stream-for-stream where the sampler alignment
  makes that meaningful.
