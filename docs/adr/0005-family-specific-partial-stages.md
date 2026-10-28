# Family-Specific Partial Stages

**Status: Accepted.** Supersedes the partial-stage interface and automatic lifting described in ADR 0004. Shape tracking and inspection are deferred while the functional core is explored.

## Decision

`Partial[T]` is an abstract base class with exactly two operations: `__call__` and `map`. `T` is the immediate result, not an eventual leaf reached through an unknown number of stages. Every concrete partial family implements its own call signature and its own `map`, whose return annotation preserves that family with a new result parameter. Each family is responsible for repackaging its own state without mutating the original or executing the suspended computation.

There is no required `bind` method, universal parameter list, or generic partial-game facade. Families may add their own binding-like operations. Calling supplies one family's inputs and executes exactly that stage; a zero-parameter family defers execution until `()`.

Mapping obeys `p.map(f)(args) = f(p(args))`. The callback's complete result is retained, including any new partial stages. Identity and composition hold for pure callbacks. Mapping is deferred postcomposition, not an implicit join, cache, or scheduler.

Transformations are ordinary unary callables. Traversal is explicit:

```python
source: Scale[Offset[Game]]
result = source.map(lambda offset: offset.map(transform))
# Scale[Offset[TransformedGame]]
```

The implementation does not inspect return annotations, search result contracts, or automatically traverse a recursive `Layers` alias. It has no shape/dtype tracking, generic `.eval()` projection, or built-in callable-wrapper factory. Concrete-game shape checks with jaxtyping remain useful and independent of this interface.

## Trade-Off

Requiring each family to implement `map` introduces a small amount of local code, but lets that family preserve its own state, concrete type, and call signature. A universal `Partial[P, T]` wrapper plus recursive `Layers` traversal preserves runtime stage order but erases deeper signatures and nominal families in the result type. Python typing cannot directly express arbitrary type-constructor rebinding; explicit family-preserving maps avoid depending on that capability or imposing a fixed typing depth limit.

## Scope

The abstract base does not provide a useful universal signature for binding an opaque partial; callers retain concrete family types when they need signature checking. Full-pipeline scheduling, inspection, metadata propagation, stub generation, and additional lifting conveniences are separate decisions. In particular, mapping does not promise that upstream work waits for parameters introduced by a later callback: each stage executes when it is called.
