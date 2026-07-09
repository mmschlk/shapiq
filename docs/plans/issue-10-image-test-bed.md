# Issue 10 — Image test bed: chunked torch games

Status: **landed 2026-07-09 (rudimentary)** — the seam exists; this file tracks what the test
bed should measure next and the design decisions still open.

## What landed

- `SuperpixelMasker(inputs, baseline, labels)` (`src/shapiq/games/torch/_superpixel.py`):
  players are superpixels via an integer label map; a grid is just `grid_labels(h, w, grid)`,
  and irregular (SLIC-style) layouts are different label maps, not new masker code. The masker
  stays a plain mask-and-replace: one gather (`masks[..., labels]`), one `where`, masks moved
  to the inputs' device. Baselines: float, scalar tensor, `(channels, 1, 1)`, or a full image.
- `ImageGame(masker, model, link_function, batch_size)` (`src/shapiq/games/torch/_image.py`):
  the game owns efficient masker use. Coalitions stream through masker and model in chunks
  whose flat image count stays within `batch_size` (explanation-target batches divide the
  coalition samples per chunk; at least one coalition per chunk), only one chunk of masked
  images is alive at a time, the model receives flat `(batch, c, h, w)` tensors as trained,
  and the link function runs once on the concatenated predictions.
- Device seam: maskers validate one device across their tensors and move coalition masks to
  it; `to_jax` falls back through host memory when JAX has no backend for the tensor's device.
- `examples/image_superpixels.py`: transparent-model sanity check (the blob superpixel holds
  the payout share), a trained tiny CNN, a direct-game `batch_size` throughput scan, and a
  sampled-vs-exact FSII cross-check.

## Review findings applied (2026-07-09, three-lens review)

Float baselines accepted; `batch_size` bounds the flat model batch under target batches
(was: multiplied by the target count); transient two-chunk residency removed (`del` before
the next chunk); scalar coalition arrays evaluate instead of raising IndexError; channels-last
inputs and 1-based/SLIC label maps get teaching errors; per-channel baseline docs corrected to
`(channels, 1, 1)`; example reports via `attributions_by_order` and times the game directly.

## Recorded trade-offs (not bugs)

- Fixed per-chunk overhead is ~30-40 us (JAX-side coalition slice + DLPack import is about
  half); it dominates only when a model forward costs less than ~100 us/chunk — toy models,
  not target workloads. Fixing it would require maskers to accept pre-converted torch masks,
  dirtying the clean CoalitionArray-only masker boundary. Keep the boundary; revisit only if
  a profile of a real workload says otherwise.
- `torch.no_grad` vs `torch.inference_mode`: measured at parity on CPU; inference_mode is the
  stricter default but lets inference tensors escape into user link functions. Re-measure on
  GPU before switching, and switch `TorchCallableGame` in the same breath.
- An empty coalition array still makes one zero-size model call (shape discovery); models must
  accept empty batches. Pinned by test.
- A torch `DataLoader` (workers, pin_memory) is not the right tool here: masked images are
  generated on-device from tiny bool masks, so there is no host-side dataset to overlap.

## What to measure next on this test bed

1. A resnet-scale model at 224x224 on MPS/CUDA: masked-images/s vs `batch_size` to find the
   flat-batch knee; confirm chunk overhead < 1%.
2. `torch.cuda.max_memory_allocated` vs `batch_size` and vs target-batch size — validates the
   one-chunk residency claim under real memory pressure.
3. Per-phase attribution (slice/convert vs gather/where vs forward vs cat/link) in a game-only
   harness; the model-size crossover falls out of it.
4. Dedup + sampling end-to-end at realistic budgets: confirm the ~1.4 us/eval sampler overhead
   stays invisible next to ms-scale forwards.
5. One-line experiments this seam exists to host: `channels_last` memory format (the flat
   reshape stops being a view — measure the copy), `torch.compile` on the model, AMP autocast.
   The pinned equivalence tests (chunked == unchunked == composed MaskedGame) keep them honest.

## Open design decisions

- **ImageGame vs a `ChunkedMaskedPredictor`**: ImageGame is a third game shape (masker + model
  + link, bypassing MaskedPredictor) and duplicates MaskedGame's link plumbing. The chunk loop
  should be extracted into a shared helper when a second chunked modality (text/tokens)
  arrives — not before, and not as a generic ChunkedGame (which would either apply the link
  per chunk or need concat-axis-from-value_shape logic).
- **`batch_size` naming**: the glossary reserves "batch size" against Sampling Quantum
  vocabulary; here it is the standard torch model-inference batch. Either bless that reading
  in the glossary or rename to `chunk_size`.
- **`grid_labels(height, width)`** invites argument transposition on non-square images; a
  `grid_labels_like(image, grid)` convenience reading the trailing axes would remove it.
- **Masker-side "baseline"** overloads the glossary's Baseline (v(empty) on explanations);
  consider a glossary entry naming the input-space replacement values.
