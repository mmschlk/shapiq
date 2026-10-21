# Game Algebra: Selected References

Reading behind the [current design](game-algebra.md):

- [Haskell `Compose`](https://hackage.haskell.org/package/base/docs/Data-Functor-Compose.html): mapping through nested contexts while preserving their structure.
- [Autoray lazy computation](https://autoray.readthedocs.io/en/latest/lazy_computation.html): deferred operation nodes carrying dependencies and output shapes.
- [JAX `eval_shape`](https://docs.jax.dev/en/latest/_autosummary/jax.eval_shape.html) and [jaxpr](https://docs.jax.dev/en/latest/601/jaxpr.html): shape inference and typed computation representations.
- [TensorFlow `ExtensionType.Spec`](https://www.tensorflow.org/guide/extension_type): metadata descriptions for structured domain objects.
- [Jaxtyping arrays](https://docs.kidger.site/jaxtyping/api/array/) and [checking contexts](https://docs.kidger.site/jaxtyping/api/runtime-type-checking/): related logical and physical shape checks.
- [Abstract interpretation](https://www.di.ens.fr/~cousot/COUSOTpapers/POPL77.shtml): sound inference over incomplete metadata.
- [Build Systems à la Carte](https://simon.peytonjones.org/build-systems-a-la-carte/): static dependencies and dependencies discovered during execution.
