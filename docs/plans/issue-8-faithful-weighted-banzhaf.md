# Issue 8 — Faithful weighted Banzhaf and the FIxLIP path

Status: **landed 2026-07-09** (scalar ``p``; per-player probabilities stay deferred)

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

- [x] Sampler: landed as `ProductKernelSampler(n_players, p)` — iid membership flips,
  exact for any player count. It sits in the kernel sampler family grown for this issue:
  abstract `KernelSampler` (single-coalition units from a per-unit key, shared batching)
  with concretes `SizeKernelSampler(n, size_weights)` (any size-based distribution:
  size first from the normalized weights, then a uniform coalition via the permutation
  trick; `from_coalition_kernel` converts per-coalition kernel weights) and
  `ProductKernelSampler`. `ShapleyKernelSampler` is now `SizeKernelSampler` with the
  ``1/(t(n-t))`` size weights and `BanzhafKernelSampler` is `ProductKernelSampler` at
  ``p = 0.5`` — both bit-stream-identical to their pre-family implementations (pinned
  in `tests/shapiq/test_kernel_samplers.py`). Product measures deliberately do NOT go
  through size-then-uniform: coin flips are exact (no float binomial coefficients),
  cheaper, and are the only mechanism that survives per-player ``p_i``.
- [x] **Pairing decision: it goes.** Under ``p != 1/2`` the complement is not equally
  likely, so paired rows would enter the unweighted fit with the wrong implicit
  weighting. `Regression`'s ``paired`` default became ``None`` — pair exactly when the
  index's kernel is complement-symmetric (always, except `WeightedFBII` with
  ``p != 0.5``); an explicit ``paired=True`` against an asymmetric kernel raises a
  teaching `ValueError`. Reweighting complements was rejected as a variance-reduction
  research question, not a sampler feature.
- [x] Faithful solve: `WeightedFBII(p, order)` joined `Regression`'s closed set; the
  FBII free-intercept branch carries over verbatim (sampling from the product measure
  supplies the kernel), `min_budget` arithmetic shared with FBII.
- [x] Exact solve: the FBII dedicated solver generalized to
  `_free_intercept_regression_attributions(..., sqrt_weights=None)`; `WeightedFBII`
  passes ``sqrt(kernel / max(kernel))`` row weights, FBII stays on the unweighted path
  bit-identically.
- [x] Index: `WeightedFBII(p, order)` with the product-measure `regression_kernel`,
  ``p = 0.5`` extensionally equal to `FBII`, and `generalizes` following the parameter
  to `WeightedBV(p)` (verified numerically: the 1-additive product-measure fit's
  coefficients are the weighted Banzhaf value).
- [ ] Per-player probabilities ``p_i`` (the general Marichal-Mathonet form) are
  player-specific, not cardinal — they arrive with the deferred player-specific
  capability, not this issue.

## Acceptance criteria (met)

- Exact `WeightedFBII` matches a brute-force product-measure weighted least squares fit
  on the full powerset for ``p`` in {0.2, 0.5, 0.8}; sampled `Regression(WeightedFBII)`
  recovers 2-additive games exactly once identified (any ``p``) and converges to the
  exact fit on non-additive games (`tests/shapiq/test_regression_weighted_fbii.py`).
- ``p = 0.5`` reproduces the existing FBII evidence stream state-for-state
  (`test_uniform_weighted_sampling_matches_the_fbii_stream`).
