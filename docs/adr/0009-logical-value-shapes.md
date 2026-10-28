# Logical Value Shapes

**Status: Accepted.** Refines the value contract of ADRs 0001 and 0002.

## Decision

Game values expose `shape` and `dtype` metadata usable by jaxtyping. Their
`shape` is exactly the logical evaluation batch: the broadcast of the game batch
and the coalition's logical batch. Internal vector, tensor, or structured-value
axes are hidden by the concrete representation, just as game and player-set
representations hide their internal axes. No particular wrapper class is required.

Ordinary arrays represent batches of scalar values. Vector and structured values
use shaped representations, for example ndlike objects with protected value axes;
plain dictionaries and Python scalars are not game values.

The generic value bound is a structural shape/dtype protocol, not an already
variadic jaxtyping annotation. This lets concrete checked operations apply local
shape contracts such as `Shaped[V, "*batch"]` without nesting variadic shapes.
Jaxtyping checks the type variable's bound at runtime, not a concrete generic
specialization. Concrete representations still own internal layout validation.

## Trade-Off

A single logical shape makes result-batch checks independent of value layout and
avoids guessing where batch axes end. In exchange, raw vector/tensor outputs and
unshaped structured outputs need a representation that hides their internal axes.
