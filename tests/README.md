# Tests and Runtime Shape Checks

Run the normal suite, using the same selective beartype checks as production:

```bash
uv run --no-default-groups --group test pytest
```

No global import hook or typeguard override is enabled by default. Invalid-shape/dtype tests for explicitly decorated coalition conversions and game/player boundaries still run, so normal runs verify that beartype actually enforces these contracts. The `test` dependency group is also included by `dev`; neither group requires typeguard.

For the additional exhaustive sanity check, opt in explicitly:

```bash
uv run --no-default-groups --group test-typeguard pytest --typeguard
```

The `test-typeguard` group includes the normal test dependencies plus the typeguard 2.x series recommended by jaxtyping. `--typeguard` installs jaxtyping's import hook for both `shapiq` and `tests`, checking annotated arguments, return values, and dataclass constructor fields. It also switches the shared boundary decorator to typeguard before collection. Tests marked `typeguard` exercise exhaustive instrumentation and are skipped unless the option is enabled; production-boundary tests run in both modes.

Use normal Python annotations for runtime-checked contracts, not the stringifying `from __future__ import annotations`. Python 3.14+ supplies native lazy annotation evaluation without that import. The project still supports Python 3.13, so references to a class from inside its own body may need quotes.

With `--typeguard`, `test_runtime_typing.py` and `test_typeguard_takeover.py` verify automatic instrumentation and exhaustive container checking. The notebook tests execute the actual code in `examples/game_api.ipynb` in both modes. They exercise constructors, player/game broadcasting, coalition batches, and evaluation return shapes with beartype normally and typeguard when opted in.

## Selective Runtime Checks

Beartype is a core dependency for explicitly checked runtime boundaries. Use shapiq's shared `@typechecked` decorator, which combines jaxtyping with beartype so that arguments and return values share a shape-checking context:

```python
from jax import Array
from jaxtyping import Bool

from shapiq.typing import typechecked


@typechecked
def accept_coalition(
    coalition: Bool[Array, "*batch {n_players}"], n_players: int
) -> Bool[Array, "*batch {n_players}"]:
    return coalition
```

Only decorated boundaries are checked at runtime; importing shapiq does not install an import hook. Coalition pack/unpack functions and the notebook's concrete game/player constructors and public methods use these explicit decorators. For dataclass constructors, place `@typechecked` above `@dataclass`. Game-specific admissibility still belongs to `Players.is_valid`; a shape annotation does not replace that domain check.

Only with `--typeguard`, `tests/conftest.py` replaces the shared decorator before collection with `jaxtyped(typechecker=typeguard.typechecked)`. Explicit boundaries then use exhaustive typeguard checks, including dataclass constructors. Merely stacking the import hook with explicit beartype decorators is insufficient: jaxtyping does not rewrap an already-checked dataclass constructor. Use the shared decorator instead of hardcoding beartype at boundaries.

The opt-in import hook covers functions in `shapiq` and `tests`, including functions without explicit decorators or jaxtyping annotations. Ordinary annotations such as `list[int]` are checked too, without beartype's random container sampling. Functions with no annotations still have no argument/return contracts to validate; shape checks require shape annotations or explicit `isinstance` checks. Imports made before the hook or before the shared decorator is switched cannot be retroactively instrumented.

Keep reusable jaxtyping aliases in assignment form (`BoolCoalition = Bool[...]`): the test suite's typeguard 2.x does not check Python's `type Alias = ...` aliases. Under JAX JIT, metadata checks run during tracing, not on every execution of compiled code; shape-determining integer arguments such as unpacking's `n_players` must be static.

`test_runtime_beartype.py` starts a fresh Python process with typeguard unavailable and exercises both coalition conversions and the notebook. This ensures explicit checks work independently of pytest's automatic instrumentation.

`test_finite_player_game.py` runs evaluation and ordinary abstractness checks for `FinitePlayerGame`, `BoolFinitePlayerGame`, and `PackedFinitePlayerGame` with both the normal ABC metaclass and a test-only adapter that applies `_py_abc`'s namespace-based calculation. Resetting native evaluators to abstract base hooks is also tested without a dataclass decorator. These tests need no separate interpreter or additional dependencies.

## Typing Conventions

- Shared aliases describe individual representations, not relationships:
  `BoolCoalition = Bool[Array, "... _"]` requires boolean storage and a final
  membership axis without binding a batch name or player count. Packed-coalition
  aliases likewise require uint8 storage and a byte axis; the player set owns
  the relationship between byte width and player count.
- Write relational shapes inline at the checked operation. For example, packing
  relates `Bool[Array, "*batch n_players"]` to
  `UInt8[Array, "*batch (n_players+7)//8"]` in one signature. Repeating the shape
  there makes its input/output contract visible rather than hiding bindings in
  aliases.
- Dimension names are local variables in a checking context, not a global
  schema. Reuse a name only to require equality or deliberate `#` broadcasting.
  Use different names for independent dimensions and `...` or `_` where no
  relationship is checked. Separately decorated calls have independent bindings.
- Decorating a dataclass checks its constructor; it does not create bindings
  shared by every method or persistent shape invariants across mutation.
- Constructors validate representation invariants. `Players.is_valid` owns
  coalition admissibility. Numerical helpers relate their array shapes locally.
  Preserve the distinction between symmetric broadcasting and player sets
  broadcasting into a game batch without enlarging it.
- Values expose logical `shape` and `dtype` metadata. Their shape is exactly the
  broadcasted evaluation batch; vector or structured representations hide their
  internal axes, for example with ndlike's protected axes. A plain array is a
  batch of scalar values. Generic bounds describe this shape/dtype protocol;
  concrete operations can apply `Shaped[V, "*batch"]` without nested variadics.
  Runtime checks use the bound, not the concrete generic specialization.
- `NumericValue` and `FloatVectorValue` constrain dtype without binding dimension
  names. The concrete representation, not `FloatVectorValue`, validates any
  internal vector layout.
- Keep `Game.eval -> Shaped[V, "..."]` generic: the coalition representation owns
  its internal axes. Static typing preserves `V`; jaxtyping checks its bound with
  `isinstance`, avoiding typeguard 2.x's unsupported `issubclass` check for bare
  type variables bounded by protocols with properties. Finite-player evaluation
  relates the game and coalition batches in a
  checked dispatch helper and verifies the returned value's exact logical shape.
  This runs after admissibility validation, preserving domain-specific errors.
  The base `Game[C, V]` and `Players[C]` leave `C` unbounded: concrete player
  sets declare admissible representations rather than a finite-array alias
  imposing a representation on every game family.

`test_typing_aliases.py` checks that aliases do not couple independent shapes and
that local annotations still enforce relationships. Finite-game tests cover
logical result shapes, shaped structured values with protected internal axes,
unshaped-result rejection, empty batches, and eager/JIT evaluation.

## CI

`.github/workflows/tests.yml` runs the normal beartype suite on Linux, macOS, and Windows with Python 3.13, 3.14, and 3.15. A separate Linux/Python 3.15 job runs the complete suite with `--typeguard`. Each job installs only its required test dependency group, not the development or machine-learning extras.

On Python 3.15+, the opt-in setup adapts two importlib API changes for jaxtyping 0.3.11: removal of `cache_from_source`'s deprecated `debug_override` argument and addition of `source_to_code`'s `fullname` argument. Remove these adapters once the upstream import hook supports the changed APIs; normal tests and production do not use them.
