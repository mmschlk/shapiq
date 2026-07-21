# Index-Dispatched Estimator Entry Points

*The orientation rider below is reverted (2026-07-09): orientation left the index objects and explainers and survives only as a representation field on explanation arrays — whether interaction keys are sets or tuples is not a game-theoretic property of an index. The entry-point decision itself stands.*

Each estimation-method family has one user-facing entry point, and the interaction index object selects the concrete implementation: `PermutationSampling(game, index)` accepts `SV()`, `SII(order=k)`, or `STII(order=k)` and dispatches to the matching walk layout and estimator; `Regression(game, index)` accepts `SV()` or `FSII(order=k)`; `ExactExplainer(game, index)` already dispatched on capability protocols. The class-per-index names (`PermutationSamplingSV`, `PermutationSamplingSII`, `PermutationSamplingSTII`, `RegressionFSII`) are removed rather than aliased — version two is unreleased, and one grammar beats two documented ones. Dispatch is keyed on index types through internal tables, never on name strings; support is statically visible through closed unions in the constructors, and unsupported indices fail at construction with an error that names the supported set.

The permutation and regression families use closed unions instead of open capability protocols deliberately. Regression's estimator relies on the sampler drawing coalitions with probability proportional to the index kernel, which is what lets every sampled row enter the least squares problem with unit weight; an open `RegressionIndex` bound would accept a custom-kernel index whose estimates would be silently wrong under the Shapley-kernel sampler. The union widens when the sampler becomes kernel-parametrized. Permutation support is a closed set for a different reason: walk layouts are bespoke per index and not derivable from any declared capability.

The Shapley value carries both capabilities. The KernelSHAP kernel and the order-one faithful-interaction kernel differ by a constant factor, and weighted least squares is invariant to constant kernel scaling, so `SV` provides a `regression_kernel` alongside its discrete-derivative weights: `Regression(game, SV())` is KernelSHAP, verified to produce the same evidence stream and estimates as `Regression(game, FSII(order=1))` up to explanation metadata. The exact explainer prefers the cheaper derivative path when both capabilities are present. The Banzhaf value's least squares formulation is unconstrained (Hammer–Holzman) and does not match the constrained-interpolation contract of the regression capability, so BV stays derivative-only until an unconstrained regression variant exists.

Interaction orientation moved onto the index objects with this change, completing the pattern set by order: whether attributions attach to player sets or ordered tuples is part of an index's definition, not an explainer request, so the `orientation` constructor parameter left `Explainer` entirely and `Explainer.orientation` derives from the index. Requesting an index at a foreign orientation is now unrepresentable rather than validated.

## Amendment (2026-07-10)

The closed per-method index sets are now realized as internal single-dispatch family
registries (`permutation_family`, `regression_family`) rather than hardcoded tables: the
sets are exact-and-explicit instead of closed, teaching errors derive their supported
lists from the registry, and the exact-type guards stand unchanged. See ADR 0011 for the
dispatch boundary rules.

## Amendment (2026-07-21)

Two registry categories exist, and the rule for choosing the dispatch axis is: **register
where the algorithm's variance lives, and capability-check the other axis.** Sampling
estimators vary by index (walk layouts, kernels) and take any game, so their families
register on the index type. The closed-form tree explainer varies by game construction
(interventional constraints today, a path-dependent sibling planned) and serves any
cardinal index, so its `tree_explanation` registry — the **tree-game registry** — registers
on the game type and capability-checks the index. With this amendment the exact-type
subclass guards on all entry points are removed (see ADR 0011, amendment 2026-07-21):
registries dispatch along the MRO, and subclasses inherit their parent's implementation
at the experimenter's own semantic risk.
