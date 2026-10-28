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

`Partial[T]`, exported from `shapiq.partial`, is an abstract suspended stage with immediate result `T`. The same interface supports partial games, players, and values. [ADR 0005](../adr/0005-family-specific-partial-stages.md) records the family-specific interface and explicit traversal decision.

```python
class Partial[T](ABC):
    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> T: ...

    @abstractmethod
    def map[U](self, function: Callable[[T], U]) -> Partial[U]: ...
```

Each concrete family implements its own call signature and repackages itself in `map`. The concrete return annotation replaces only the result parameter, preserving the family and any family-specific state. Mapping must not execute the stage or mutate the original.

```python
class Offset[T](Partial[T]):
    def __init__(self, function: Callable[[float], T]):
        self.function = function

    def __call__(self, *, offset: float) -> T:
        return self.function(offset)

    def map[U](self, function: Callable[[T], U]) -> Offset[U]:
        return Offset(lambda offset: function(self(offset=offset)))
```

Calling supplies one family's inputs, executes exactly that stage, and returns its result. There is no required `bind` method. A zero-parameter family defers execution until `()`.

```python
family: Factors[Offset[Game]]
remaining = family(factors=xs)  # Offset[Game]
game = remaining(offset=3.0)    # Game
```

## Mapping and Explicit Lifting

Transformations are ordinary unary callables described by `Transformation[A, B]`. They do not inspect result annotations or traverse partials automatically. Explicit nested `map` calls select the target and retain every surrounding family's concrete type and call signature:

```python
family: Factors[Offset[Game]]
explained = family.map(lambda offset: offset.map(shapley_game))
# Factors[Offset[ShapleyGame]]

values = explained.map(lambda offset: offset.map(lambda game: game.eval(S)))
result = values(factors=xs)(offset=3.0)
```

Mapping obeys deferred postcomposition. The callback's complete result is retained; new partial stages are never flattened:

```text
p.map(f)(args) = f(p(args))

p : Offset[A]
f : A -> Offset[B]
p.map(f) : Offset[Offset[B]]
```

The target may itself be partial. Apply such a handler directly to that stage, or map explicitly over its surroundings:

```python
def replace_offset(stage: Offset[Game]) -> Offset[Game]:
    ...

changed = family.map(replace_offset)  # Factors[Offset[Game]]
```

Each stage executes when called. This does not imply that all upstream computation waits for parameters introduced by later callbacks. Full-pipeline scheduling, caching, inspection, and generic projection methods are not part of this core.

## Concrete Shapes and Jaxtyping

The minimal partial interface does not track shape or dtype. Concrete games and players retain their logical batch shapes and coalition-broadcasting contracts.

Jaxtyping relates logical batches to backing arrays at construction, cast, and evaluation boundaries. For a finite scalar-valued table, check the related fields in one context:

```python
with jaxtyped("context"):
    assert isinstance(g, Shaped[Game, "*batch"])
    assert isinstance(g.players, Shaped[Players, "#*batch"])
    assert isinstance(g.players.ids, Int[jax.Array, "#*batch n"])
    assert isinstance(g.values, Float[jax.Array, "*batch 2**n"])
```

`Shaped` describes the domain object's logical shape; numeric annotations describe its storage. Anchoring `*batch` on the game makes the player check enforce one-way broadcasting.

Values follow the same logical-shape convention: their `shape` is exactly the
broadcasted game/coalition batch. Internal value axes are protected by the concrete
representation; ordinary arrays are batches of scalar values, not unwrapped
vector-valued results ([ADR 0009](../adr/0009-logical-value-shapes.md)).

Shared typing aliases describe context-independent representations without named
dimension bindings. Shape relationships belong inline in a checked operation or
constructor. See the [typing conventions](../../tests/README.md#typing-conventions).

## Implementation Status

The runtime core consists of the `Partial[T]` ABC and the ordinary `Transformation[A, B]` callable protocol. Concrete families, deferred mapping, exact nested call order, and partial-producing callbacks are demonstrated in the notebook. Partial metadata and a more inspectable representation are deliberately deferred.

[Selected references](partiality-theory.md).
