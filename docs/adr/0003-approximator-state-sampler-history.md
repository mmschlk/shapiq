# Approximator State, Sampler Evolution, And History

> Superseded in part by ADR 0014 (2026-07-22): samplers are stateless draw
> values and never evolve; sampler history and the mutable/history interlocks
> are gone; history is always on via state checkpoints. The functional
> transition contract (`sample` returns a new approximator) stands.

Approximators use functional transitions: `sample(budget)` returns a new approximator after sampling coalitions, evaluating the game, and updating approximation evidence, while `explain()` materializes an `ExplanationArray` from the current state. `approximate(budget)` is shorthand for `sample(budget).explain()`, and `sample(0)` is a no-op returning the same approximator.

Sampling state is split between `ApproximationState` and `Sampler`. `ApproximationState` stores the evidence needed to explain, while `Sampler.sample(state, budget)` returns both a `CoalitionArray` and the next sampler so RNG state, duplicate tracking, and adaptive sampling machinery can evolve explicitly without being hidden in the approximator.

Approximation history is optional and tracks value-equivalent past states for rollback and convergence analysis. State history is owned by the `ApproximationState`; approximators keep aligned sampler snapshots only when history is enabled. Mutable states or samplers are incompatible with history, because rollback must reconstruct value-equivalent `(state, sampler)` pairs rather than depend on mutated objects.

Approximators are constructed with an `EmptyState` and perform no game evaluations until the first sample call. Approximation history begins at the first evidence state: an empty state with history enabled lists only itself, sampler snapshots reset when the first evidence state is created, and rollback cannot reach the unseeded approximator.
