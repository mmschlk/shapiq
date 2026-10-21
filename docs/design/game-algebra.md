# Game Algebra

**Current v2 design.** Examples describe the intended API; the runtime skeleton is still being developed.

Models become games, transformations produce games, and explanations are interpretations of those results:

```python
g = BaselineGame(model, background_data)
h = shapley_game(g)
values = h.eval(singleton_coalitions)
```

## Games and Players

`Game[C, V]` maps admissible coalitions `C` to values `V` and owns `Players[C]`. Both games and player sets are always batches, including shape `()`. Indexing selects batch elements; `g(S)` means `g.eval(S)`.

`Game` and `Players` implement `JaxLikeImplementation`. Concrete array-backed families can compose `JaxAxisProtected` to hide player, coefficient, and value axes from their logical batch shape.

```python
g.shape                    # game batch, e.g. (8, 3)
g.players.shape            # broadcasts into g.shape without enlarging it
g[0].shape                 # (3,)
g(S)                       # broadcasts game and coalition batches
```

`Players.is_valid` returns one Python boolean for the whole coalition batch, normally checking type, dtype, internal shape, and broadcasting metadata. Games delegate with their own batch shape:

```python
g.is_valid(S) == g.players.is_valid(S, batch_shape=g.shape)
```

## Representations and Transformations

A **cast** preserves players, batch shape, and coalition values. A **view** is a constant-time, storage-sharing cast. Constructors, casts, and evaluation can each require substantial computation.

| Representation | Stored quantities |
| --- | --- |
| Coalition table | Coalition values |
| Möbius | Möbius coefficients (dividends) |
| Fourier | Fourier coefficients |

For suitable numeric value spaces, these bases represent the same game. Evaluation always returns coalition values; coefficient lookup accesses the representation.

```python
m = cast(g, MobiusGame)
f = cast(g, FourierGame)
assert_allclose(g.eval(S), m.eval(S))
assert_allclose(g.eval(S), f.eval(S))
m.coefficient(T)             # Möbius dividend for T
```

Mathematical capabilities such as `KAdditive` and normalized `Additive` are independent of storage. The exact Shapley transformation produces an additive game with the same players:

```text
h(S) = sum(phi_i(g) for i in S)
h(empty) = 0
h(all_players) = g(all_players) - g(empty)
```

Ordinary values also have a zero-player-game interpretation. Transformation routing considers compatible game casts, including this interpretation:

```python
x = jnp.array(7.0)
g0 = cast(x, ZeroPlayerGame)
g0.eval(jnp.zeros((0,), dtype=bool))  # 7.0
shapley_game(x)                     # through the zero-player-game cast
```

## Partial Stages and Deferral

`Partial[P, T]` describes a suspended stage with parameter signature `P` and result `T`. A partial game eventually produces a game batch; the same core supports partial players and values.

```python
class Partial[**P, T]:
    def bind(self, *args: P.args, **kwargs: P.kwargs) -> T: ...
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T: ...
    def map[U](self, f: Callable[[T], U]) -> Partial[P, U]: ...
```

Binding supplies one complete stage's inputs, executes that stage, and returns its result. Calling is shorthand for binding. Each reduction removes one pending-input requirement:

```python
ppg: Partial[ScaleParams, Partial[OffsetParams, Game]]
pg = ppg(scale=2.0)          # Partial[OffsetParams, Game]
g = pg(offset=3.0)           # Game
```

A zero-parameter partial defers execution until `()`. Use `delay` around the computation to defer its cost; `defer` wraps an existing value.

```python
pg = delay(lambda: cast(g, MobiusGame))
pv = pg.eval(S)              # deferred coalition evaluation
value = pv()                # executes conversion and evaluation

p = defer(g)                # already constructed game, suspended for composition
```

## Mapping and Transparent Lifting

`map` transforms one stage's result while preserving its parameters. Composition records work; reduction executes it. A callback's complete result becomes the mapped result, including any further partial stages.

```text
p.map(f)(args) = f(p(args))

p : Partial[P, A]
f : A -> Partial[Q, B]
p.map(f) : Partial[P, Partial[Q, B]]
```

`@lifted` extends an ordinary handler through arbitrary surrounding stages, stopping at its declared input type. For a handler `A -> B` and any finite partial stack `C[-]`:

```text
lifted(f) : C[A] -> C[B]
```

Concrete inputs run the handler directly; partial inputs preserve the surrounding stages until reduction. Core transformations and domain methods provide this behavior through their ordinary spelling:

```python
@lifted
def to_mobius(g: TableGame) -> MobiusGame:
    return cast(g, MobiusGame)

to_mobius(g)                 # MobiusGame
to_mobius(pg)                # same pending parameters, MobiusGame result
shapley_game(ppg).eval(S)     # preserves both stages, eventually returns values
```

The target may itself be partial. The handler receives that target intact and its output is inserted intact:

```text
t : Partial[Q, Game] -> Partial[R, Game]

lifted(t)(pg)   = t(pg)
lifted(t)(ppg)  = ppg.map(t)
lifted(t)(pppg) = pppg.map(lambda pp: pp.map(t))
```

Each stage carries its declared result structure so lifting can locate the target before execution. Partial-aware handlers compose deferred work; enclosing maps determine when those handlers run.

## Shape Propagation and Jaxtyping

Every nested partial game retains metadata for its eventual game batch. `PartialShape = tuple[int | None, ...] | None`: `(8, None)` has one unresolved dimension, `None` has unresolved rank, and `()` is a known zero-dimensional batch.

Binding refines this metadata. Transformations supply metadata-only output-shape rules; casts preserve batch shape and players. These rules propagate through every surrounding stage:

```text
ppg                    -> pg                     -> g
eventual batch (8, None)   eventual batch (8, None)   batch (8, 5)
```

```python
shapley_game.output_shape((8, None))  # (8, None)
```

Unknown output metadata stays unknown until a rule or supplied input refines it. Concrete results must satisfy the accumulated metadata.

Jaxtyping relates logical batches to backing arrays at construction, cast, and evaluation boundaries. For a finite scalar-valued table, check the related fields in one context:

```python
with jaxtyped("context"):
    assert isinstance(g, Shaped[Game, "*batch"])
    assert isinstance(g.players, Shaped[Players, "#*batch"])
    assert isinstance(g.players.ids, Int[jax.Array, "#*batch n"])
    assert isinstance(g.values, Float[jax.Array, "*batch 2**n"])
```

`Shaped` describes the domain object's logical shape; numeric annotations describe its storage. Anchoring `*batch` on the game makes the player check enforce one-way broadcasting.

## Implementation Status

`ParamSpec` preserves stage signatures through binding and explicit nested maps. Automatic recursive lifting's static typing, runtime matching priority, and the metadata/projection API are still being specified.

[Selected references](partiality-theory.md).
