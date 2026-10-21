# Core Interfaces: Everything Is a Game

**Earlier API sketch.** The evolving [game algebra specification](game-algebra.md) revises cast costs, values as zero-player games, reduction semantics, and deferred attribute access. This document and its notebook describe the initial skeleton; consult the algebra specification for the current discussion.

This is the v2 API sketch. The glossary is in `CONTEXT.md`; [ADR 0004](../adr/0004-everything-is-a-game.md) records the architectural shift. Only the abstract contracts and capability protocols are defined in code. Constructors, named transformations, storage classes, views, and the lifting layer below are proposed interfaces.

The notebook [game_api.ipynb](../../examples/game_api.ipynb) explores these contracts with a tiny payoff table, a dummy transformation, and nested partial binding with shape refinement.

## The Main Flow

```python
g1 = BaselineGame(model, background_data)
g2 = shapley_game(g1)
values = g2(singleton_coalitions)
```

The model is embedded in a cooperative game. Explaining that game transforms it into another game, which can itself be evaluated, transformed, or viewed through another interface. There is no required `Explainer -> ExplanationArray` boundary.

For the exact Shapley transformation:

```text
g2.players = g1.players
g2(S) = sum(phi_i(g1) for i in S)
g2(empty) = 0
g2(all_players) = g1(all_players) - g1(empty)
```

Thus `g2({i})` is exactly the Shapley value, without a baseline offset. The source baseline remains available from `g1(empty)`. `shapley_game` is the proposed public spelling; helpers for extracting values can come later. An approximate implementation must make its accuracy/budget policy explicit; it still returns an additive game, with estimated singleton values.

## Base Contracts and Shapes

| Contract | Meaning |
| --- | --- |
| `Players[C]` | JAX-like batch of player sets accepting coalition representation `C` |
| `Game[C, V]` | JAX-like batch with `players: Players[C]` and `__call__(coalition: C) -> V` |
| `KAdditive` | Semantic capability declaring `additivity: int` as an upper bound |
| `Additive` | K-additivity bound one, additionally guaranteeing zero empty value |
| `PGame[Params, Out]` | Deferred callable with a positional/keyword parameter signature; `Out` is a game or another partial game |
| `PartialShape` | Known/unknown dimensions of the eventual game batch, or unknown rank |
| `Transformation[A, B]` | Public transformation of concrete `A` to `B`, also lifted over partial inputs |

`Game` and `Players` inherit `ndlike.jax_like.JaxLikeImplementation` without selecting storage. Games are **always batches**, including the zero-dimensional batch `()`. Their logical shapes count games and player sets, respectively. `players.shape` must broadcast into `game.shape` without enlarging it. Evaluation broadcasts the game batch with the coalition's logical batch. Internal player/value axes are excluded.

`Players.is_valid(coalition, *, batch_shape=None) -> bool` owns coalition admissibility. It normally checks static metadata rather than array contents. `Game.is_valid` delegates with `batch_shape=game.shape`, which matters when scalar player sets are shared across a larger game batch.

Partial games carry the same eventual batch perspective at every nesting level:

```text
PGame.shape: PartialShape
PartialShape = tuple[int | None, ...] | None

(8, 3)       known rank and dimensions
(8, None)    known rank, unresolved second dimension
None         unresolved rank
()           known zero-dimensional batch, not unknown
```

The shape describes the eventual game batch, not the number of missing parameters or nesting levels. Each `None` dimension is independently unresolved; this is not a symbolic algebra. A binding refines unknowns without contradicting known dimensions, and the final `Game.shape` must satisfy the declared shape. If missing inputs may introduce an unknown number of batch axes, rank remains unknown rather than guessing a scalar shape.

`Transformation.output_shape(input_shape) -> PartialShape` propagates metadata using the input shape and bound configuration, without evaluating games. Shape-preserving transformations retain partial dimensions: `shapley_game.output_shape((8, None))` is `(8, None)`. Shape-changing transformations declare their own rule. Missing shape-relevant configuration leaves conservative unknowns until binding refines it. This rule also describes the eventual output of partial-producing transformations.

For example, a family with shape `(8, None)` stays `(8, None)` after lifting a Shapley transformation; binding a five-value feature sweep may refine it to `(8, 5)`. The partial-producing transformation defines how parameter batch axes map or broadcast into game batch axes. Shape refinement and compatibility checks are metadata work and do not trigger deferred game computation.

## Semantic Capabilities and Concrete Representations

The hierarchy has two independent dimensions:

```text
Semantic capabilities:       KAdditive <- Additive
Core game abstraction:      Game[C, V]
Future storage families:    Game + JaxAxisProtected + applicable capabilities
Deferred construction:      PGame (separate from Game)
```

`KAdditive` promises that Möbius coefficients above its declared order vanish. The bound is not necessarily tight. A k-additive representation may contain a constant coefficient; `Additive` additionally promises normalization, so singleton values directly determine every coalition value. These guarantees apply to suitable additive value spaces, not arbitrary `Value` objects.

Capability protocols describe mathematical promises, not proofs or layouts. An `additivity` attribute alone is not evidence that an arbitrary game satisfies the mathematics. The protocols are used alongside `Game`, not as replacements for its array/evaluation contract, and are not structural runtime dispatch tests.

Future concrete families can use singleton storage, dense interaction coefficients, sparse coefficients, or model-backed evaluation. Materialized array-backed families compose `Game` with `ndlike.protected_axis.jax.JaxAxisProtected`; player/interaction/value axes are protected and batch axes remain public. An additive game need not use one particular storage family. Likewise, a Shapley result does not need a special storage class solely because of how it was computed.

Layouts are intentionally unspecified here. In particular, ndlike's current protected fields require equal batch shapes, while `Game.players` permits broadcasting. A representation must preserve that distinction when indexing/reshaping; a shared player set cannot simply be treated as an identically batched storage field.

## Cheap Views Versus Transformations

Use the existing flextype API directly:

```python
from flextype.cast import cast, caster, castmethod

view = cast(g2, SingletonViewGame)  # proposed view family
```

Register a free-standing view adapter with `@caster` or an instance method with `@castmethod`. Each registered game cast must:

- preserve player identities, admissible coalitions, logical batch shape, and every coalition value;
- share existing storage, with only constant-size wrapper/metadata work;
- avoid evaluation, sampling, data scans, data-sized copies, and backend transfers.

Constant time is with respect to game/player/storage size; dispatcher bookkeeping is separate. Flextype's numeric cast cost ranks routes but does not establish or relax this contract. A cheap cast may expose helpers or an already-guaranteed capability; it cannot establish additivity by inspecting arbitrary game values.

`shapley_game`, player removal, and expensive materialization are named transformation functions, even when a particular input admits a cheap implementation. Preserving the value function alone does not make a costly representation conversion a cast.

## Transformation Dispatch

The installed flextype entry point is `flextype.flexdispatch`, implemented in `flextype.singledispatch`. Concrete handlers are ordinary `A -> B` callables; `B` may be a `Game` or a `PGame` introducing new parameters. Functions may expose configuration keywords; `Transformation[A, B]` describes their game argument once configuration is fixed.

The proposed public transformation layer first recognizes partial inputs, propagates `output_shape`, and defers the operation. Once a concrete input is available, flextype can select a handler by game type. Model-backed families can then route by the wrapped model type, using a second dispatcher or `dispatch_on`. For example, a tree-specific handler computes the same Shapley-game semantics as the general handler. Public transformation callables expose the shape rule alongside dispatch; concrete numerical handlers do not each need to implement partial-game plumbing.

Dispatch does not automatically search the cast graph, and ordinary protocol attributes do not prove semantic capabilities. A handler may explicitly request an available cheap view. Nominal game families and explicit registrations are sufficient initially; there is no new `RegistryMeta` hierarchy in this sketch.

## Partial Games and Deferred Flow

A partial game is a family of game batches with missing construction parameters, not a game missing some coalition evaluations. It always exposes partial batch-shape information, even when players and dimensions are not yet fully known. Calling a `PGame` supplies parameters; calling its eventual `Game` evaluates coalitions.

```python
# Proposed transformation calls; no concrete implementations exist yet.
p1 = remove_players(g1, k=2)  # missing: which two players?
p2 = shapley_game(p1)        # same missing inputs; no game computation
g2 = p2(removed=indices)     # supply them, then compute the reduced Shapley game

# A PDP-like transformation introduces a feature-value stage.
pp = with_feature_value(p1, feature=feature_id)  # missing: removal, then value
explained = shapley_game(pp)                   # lifts through both stages
remaining = explained(removed=indices)         # still a PGame; no computation
g3 = remaining(value=new_value)                # fully specified additive game
```

Each transformation that changes players defines the remaining player identities, coalition mapping, and treatment of removed players. `k` specifies the removal count here; the parameter values select the actual members. Feature-value changes require an appropriate model-backed family and define their own PDP-like semantics. The example does not imply that any of these transformations commute.

`PGame` uses a `ParamSpec` so named and keyword-only inputs are part of its signature. One call binds one declared stage; nested stages retain separate signatures even when parameter names coincide. Families of families can nest indefinitely: a further partial transformation adds a stage, and subsequent transformations lift through every existing stage. `ppgame` is an informal name for a nested `PGame`, not a distinct type or a special depth limit. There is no implicit merging of stages or arbitrary Python-callable currying.

The public typing shape is:

```text
T: A -> B                         B may itself be a PGame
T: PGame[P, A] -> PGame[P, B]
T: PGame[P, LiftedGame[A]] -> PGame[P, LiftedGame[B]]

LiftedGame[G] = G | PGame[..., LiftedGame[G]]
```

The immediate parameter signature is retained. The recursive alias deliberately erases deeper parameter lists; runtime nesting must still preserve them. This is a typing compromise, not permission to collapse stages.

The denotational lifting law is `T(p)(args) = T(p(args))`, recursively through partial results. For game-producing transformations this is mapping; allowing a transformation to introduce another partial family gives bind-like composition. The law specifies the eventual result, not an eager evaluation order.

**Deferral contract:** composing a graph and binding a nonterminal stage must not evaluate games, estimate coefficients, or materialize transformed games. Those computations run when all parameters needed by the final game have been supplied. Repeated full calls need not share computation; caching, graph optimization, and persistence are separate future decisions.

A naive closure implementing the lifting law can compute too early when a later transformation returns a new `PGame`. The eventual shared lifting layer must carry deferred dependencies across such partial-producing transformations, and those transformations must expose remaining stages without running their upstream game computation. Wrapping arbitrary opaque callables cannot provide that guarantee automatically. The initial API states this requirement without selecting a graph-node representation, scheduler, or decorator implementation.

## Scope of the Skeleton

The Python changes are limited to callable/capability/shape contracts and exports: `PGame`, `LiftedGame`, `PartialShape`, `Transformation`, `KAdditive`, and `Additive`. Concrete games, layouts, transformation algorithms, cast registrations, shape-inference implementations, and deferred execution follow separately. Sampling/state/history can be algorithm-local concerns when an approximation implementation needs them; they are not a mandatory public explainer lifecycle.
