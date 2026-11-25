# Everything Is a Game

**Status: Accepted.** Replaces the explainer-to-explanation pipeline, supersedes ADR 0003, and updates the result/storage choices in ADRs 0001 and 0002.

## Decision and Trade-Off

The central computational interface is a game transformation. A model can be embedded in a game and then transformed into other games; an explanation is an interpretation of a resulting game, not a separate universal result container. This makes results composable through the same player/coalition/value interface, at the cost of requiring explicit mathematical semantics for each result rather than relying on a common attribution-array schema.

The Shapley transformation produces a zero-empty additive game with Shapley values as singleton values. Its grand-coalition value is the source's grand-minus-empty value. Exact and approximate algorithms implement this common result form; approximation policy must be explicit. Mathematical capability interfaces, such as `KAdditiveGame` and `AdditiveGame`, remain separate from concrete storage types built on ndlike's protected-axis support.

## Casts Preserve a Game; Transformations May Change It

`flextype.cast` registrations (`@caster` / `@castmethod`) are restricted to **Game Casts**: conversions that preserve players, coalition admissibility, logical batch shape, and all coalition values while changing representation or exposing a capability interface. Player-changing or value-changing operations are **Game Transformations**, never casts, even when they are cheap.

Cast costs distinguish **Game Views** from materializing conversions:

- **Zero-cost casts (`cost=0.0`)** are constant-time, storage-sharing views (such as exposing `AdditiveGame` or `KAdditiveGame` on a compatible game or wrapper).
- **Nonzero-cost casts (`cost > 0`)** may perform computation or materialize a representation (for example, evaluating an arbitrary `FinitePlayerGame` into a dense `TableGame` at a high fallback `cost=1000.0` so any cheaper multi-step cast chain is preferred, or taking a faster `cost=1.0` path on `ArithmeticGame` and `ReductionGame` when all leaves are `TableGame`s).

Callers bound allowed conversion work explicitly through `max_cost`. Composite wrappers such as `AffineGame`, `_ScalarDivGame`, and `SumReductionGame` use `flextype.autocast(..., max_cost=0.0)` so they transparently satisfy capability interfaces like `KAdditiveGame` (and, transitively, `AdditiveGame`) whenever their operands admit zero-cost views, without ever triggering hidden materialization.

Concrete transformation handlers, approximators, maskers, and explanation games use `flextype.flexdispatch` for type-based routing. Cast lookup and transformation dispatch remain distinct mechanisms. Mathematical capability conformance is an implementer's guarantee: concrete wrappers own their evaluation contract, and subclasses must honor parent semantics so callers and casters can rely on `isinstance` rather than exact-type guards.

## Partial Games Preserve Missing Inputs

**Superseded by [ADR 0005](0005-family-specific-partial-stages.md).** Partial games remain callable families rather than games or arbitrary callables treated as games. The current interface uses an abstract `Partial[T]` with family-specific `__call__` and `map` implementations, rather than result-dependent facades or a universal parameter signature.

Explicit nested mapping preserves the order and concrete types of pending-input stages. A callback can introduce another partial family, retained intact. Shape and dtype tracking are outside the minimal partial interface for now; concrete games and players retain their batch semantics.

The former automatic recursive lifting, partial-shape propagation, and full-graph deferral requirements are not promises of the current functional core. Each stage executes when called; scheduling and inspection are separate future decisions.
