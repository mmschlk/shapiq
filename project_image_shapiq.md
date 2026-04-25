# Project: Image Explanations with shapiq — Imputers for Vision Models

**Type:** Pull Request(s) + Demo

## Overview

Explaining image classifiers with Shapley values is one of the most visually compelling applications of XAI: the "players" are patches, superpixels, or attention tokens, and the interaction values highlight which regions drive a model's prediction — and, more interestingly, *how they interact*. shapiq already has all the game-theoretic and algorithmic machinery to compute these explanations, but it has **no first-class support for images**. The only existing image handling lives inside a single benchmark game (`shapiq_games.benchmark.local_xai.ImageClassifier`) where masking logic is hard-coded per model, which is enough for benchmarking but nowhere near enough for users who want to explain their own vision models.

The goal of this project is to close that gap. Your team will design and implement a proper **image imputer** (or a new `shapiq.image` subpackage — see Task 0) that plugs cleanly into the existing `Explainer → Game → Imputer` pipeline and works across a wide range of vision models: CNNs, vision transformers (ViTs), and potentially segmentation or vision-language models. On top of that, you will build a polished **demo** that shows the breadth and robustness of your implementation, including reproductions of recent image-explanation papers such as [**FixLIP**](https://arxiv.org/abs/2508.05430).

Two constraints are central to this project:

1. **Efficiency matters.** Vision models are expensive to evaluate; imputation with thousands of coalitions is the dominant cost of the pipeline. The imputer must support batched model calls, reasonable memory use, and ideally GPU-friendly tensor paths.
2. **Framework-agnostic.** The imputer must work with **PyTorch and JAX/Flax** models alike. Prediction should go through a user-supplied callable; image representation should not be tied to a single tensor library.

## Tasks

### Task 0: Design — one imputer or a new subpackage?

Before implementation, decide on the right abstraction. Image explanations need several orthogonal concerns:

- **Player definition:** fixed-grid patches (ViT-style), SLIC / Felzenszwalb / watershed superpixels, semantic segments, attention tokens, or user-provided masks.
- **Masking strategy:** mean-color / blur / zero / attention-mask / learned inpainting / Gaussian noise / dataset-mean baseline.
- **Model coupling:** some strategies (e.g. zeroing attention weights) require hooking into the model itself; most do not.

Study the existing imputer hierarchy (`src/shapiq/imputer/`) and the hard-coded image handling in `src/shapiq_games/benchmark/local_xai/benchmark_image.py` and `src/shapiq_games/benchmark/_setup/_vit_setup.py` / `_resnet_setup.py`. Then decide — and briefly justify in the PR — whether the right design is:

- **(a)** A single configurable `ImageImputer(Imputer)` that composes a *player-definer* (patch/superpixel/...) with a *masking strategy*, or
- **(b)** A new `shapiq.image` subpackage with dedicated modules (e.g. `players.py`, `masking.py`, `imputer.py`, `explainer.py`) that mirrors `shapiq.tree` / `shapiq.graph` in structure.

Either is acceptable — what matters is that the chosen design is extensible, well-documented, and lets users combine player definitions with masking strategies freely.

### Task 1: Implement the image imputer

Implement the core imputer(s) as a subclass of `shapiq.imputer.base.Imputer`. It must:

- Accept an image (numpy array, PIL image, or tensor) and a user-supplied `model` callable that returns predictions for a batch of images.
- Define *players* via a pluggable strategy — at minimum supporting:
  - **Fixed-grid patches** for ViT-style models (configurable grid size).
  - **Superpixels** via at least one algorithm from `skimage.segmentation` (SLIC is a sensible default).
  - **Custom player masks** provided by the user (arbitrary binary masks per player).
- Provide *masking strategies* as a pluggable component — at minimum supporting:
  - **Mean-color** imputation (current baseline in `_resnet_setup.py`).
  - **Zero / baseline-value** imputation.
  - **Blur** imputation (Gaussian blur of the masked region).
  - **Attention masking** for transformer-based models (using `bool_masked_pos` or equivalent, see `_vit_setup.py`).
  - Students should research further mechanisms (e.g. inpainting, learned imputers, dataset-mean baselines) and include at least one that goes beyond the above.
- Implement `value_function(coalitions: np.ndarray) -> np.ndarray` efficiently — **coalitions must be evaluated in batches** through a single model call where possible, not one coalition at a time.
- Work with **PyTorch and JAX/Flax models** via the user-supplied callable. The imputer itself should not import torch or jax at module level; only the player/masking strategies that actually need them should.
- Integrate with the existing explainer pipeline so that `Explainer(model=vit_model, data=image, imputer=ImageImputer(...))` works end-to-end.

### Task 2: Explainer integration & dispatch

Hook the imputer into shapiq's explainer layer. Depending on your Task 0 decision, this is either:

- A small addition to `TabularExplainer` (or a dedicated `ImageExplainer`) that accepts the new imputer and the image input, or
- A full `ImageExplainer` subclass registered in `src/shapiq/explainer/utils.py`, auto-dispatched when the input is detected as an image / vision model.

Follow the pattern of `TreeExplainer` and `TabularExplainer`. The user-facing API should be as clean as `Explainer(model, data=image)` — the explainer should pick sensible defaults (e.g. ViT → patch players + attention masking; CNN → SLIC superpixels + mean-color masking) while allowing overrides.

### Task 3: Efficiency

This is not optional. Images are the slowest modality shapiq will support, so the implementation must be deliberate about performance. Specifically:

- **Batched coalition evaluation:** the imputer should assemble all masked images for a batch of coalitions and make **one** model call per batch, not one per coalition.
- **Pre-computed player masks:** player-to-pixel mappings (patch indices, superpixel labels) should be computed once at `fit` time and reused for every coalition.
- **Memory-aware batching:** add a `batch_size` parameter so users can trade memory for speed; auto-select a sensible default based on image and model size.
- **Device handling:** if the model is on GPU, masked images should be assembled and passed on GPU without a round-trip through CPU.
- **Benchmark your implementation.** Report wall-clock time and peak memory for explaining a single image with a ViT-B/16 and a ResNet-50 across several grid/superpixel sizes and budgets. Compare against the existing hard-coded benchmark games to show at least parity.

### Task 4: Testing

Comprehensive tests under `tests/shapiq/tests_unit/tests_imputer/` (and, if you add a new subpackage, a matching test directory):

- **Correctness:** for small image sizes and few players, compare imputer outputs against `ExactComputer` on the resulting game. Shapley values on tiny images must match exactly.
- **Framework coverage:** tests with a small torch model **and** a small JAX/Flax model, ensuring identical behavior.
- **Strategy coverage:** one test per player definition × masking strategy combination (matrix test), ensuring each combination produces a valid `InteractionValues` output and sensible shapes.
- **Integration:** end-to-end test through the explainer API on a tiny ViT-like model.
- Follow the style of existing imputer tests (`test_marginal_imputer.py`, `test_baseline_imputer.py`).

### Task 5: Demo — breadth, robustness, and paper reproduction

Build a polished demo (notebook collection, Gradio app, Hugging Face Space — choose what fits best) that showcases the implementation across **several SOTA models** and **interesting examples**. Required elements:

1. **Breadth across models:** demonstrate the same imputer on at least three architectures — e.g. ViT-B/16, ResNet-50, and one more (ConvNeXt, DINOv2, CLIP image encoder, SAM encoder — your choice).
2. **Breadth across strategies:** show visual comparisons of different player definitions (patches vs. superpixels vs. custom masks) and different masking strategies (mean / blur / attention / zero) on the same image. Discuss where each strategy works well and where it fails.
3. **Interaction visualizations:** use higher-order interactions (k-SII, STII, or FSII) to show *how regions interact* — not just which region matters. Include qualitative examples where pairwise interactions reveal something first-order SVs miss (e.g. context effects, co-occurring objects, counter-factual regions).
4. **Paper reproduction:** reproduce a recent image-explanation result using your implementation. The **FixLIP** paper ([arXiv:2508.05430](https://arxiv.org/abs/2508.05430)) is a strong candidate. Pick at least one experiment, reproduce it with shapiq, and discuss agreement/discrepancies.
5. **Comparison with existing libraries:** benchmark your implementation against existing image-level Shapley explainers — at minimum [`shap.PartitionExplainer`](https://shap.readthedocs.io/en/latest/image_examples.html) with an image masker and captum's [`ShapleyValueSampling`](https://captum.ai/api/shapley_value_sampling.html) / `FeatureAblation`. Compare: (i) runtime and memory, (ii) fidelity of explanations (do they agree? where do they disagree and why?), (iii) API ergonomics. Pick a shared image + model and show the results side-by-side.
6. **Failure / edge cases:** include at least one example where the explanation is surprising or misleading — low-confidence predictions, adversarial examples, or out-of-distribution images. Honest examples make a better demo than cherry-picked ones.

The demo must be reproducible: fixed seeds, pinned dependencies, and clear install instructions.

## Relevant Existing Code

| Path | Description |
|------|-------------|
| `src/shapiq/imputer/base.py` | `Imputer` base class — subclass this (see `value_function` contract) |
| `src/shapiq/imputer/baseline_imputer.py` | Lightweight imputer reference (sample_size=1, no resampling) |
| `src/shapiq/imputer/marginal_imputer.py` | More elaborate imputer reference (sampling, categorical handling) |
| `src/shapiq/imputer/__init__.py` | Imputer registry — new imputers must be added here |
| `src/shapiq/explainer/tabular.py` | Shows how an imputer is plugged into an explainer |
| `src/shapiq/explainer/utils.py` | Explainer registry and auto-dispatch |
| `src/shapiq_games/benchmark/local_xai/benchmark_image.py` | Existing hard-coded `ImageClassifier` benchmark game |
| `src/shapiq_games/benchmark/_setup/_vit_setup.py` | ViT patch masking via `bool_masked_pos` |
| `src/shapiq_games/benchmark/_setup/_resnet_setup.py` | ResNet + SLIC superpixel mean-color imputation |
| `src/shapiq/game_theory/exact.py` | `ExactComputer` — ground truth for correctness tests |
| `src/shapiq/plot/` | Visualization utilities (extend or reuse for image overlays) |

## References

- **Original SHAP (KernelSHAP / image SHAP):** Lundberg & Lee, *A Unified Approach to Interpreting Model Predictions*, NeurIPS 2017.
- **SLIC superpixels:** Achanta et al., *SLIC Superpixels Compared to State-of-the-Art Superpixel Methods*, TPAMI 2012 — available in `skimage.segmentation.slic`.
- **ViT:** Dosovitskiy et al., *An Image is Worth 16x16 Words*, ICLR 2021. [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)
- **h-SHAP / H-SHAP / hierarchical image SHAP:** recommended starting points for your literature survey on efficient image-level Shapley methods.
- **shapiq paper:** Muschalik et al., *shapiq: Shapley Interactions for Machine Learning*, NeurIPS 2024 — for the architecture you are extending.

## Expected Deliverables

**PR(s):**

- A clean, well-designed image-imputation module — either a single `ImageImputer` or a new `shapiq.image` subpackage, depending on the design you choose in Task 0.
- Support for at least three player-definition strategies (patches, superpixels, custom masks) and at least four masking strategies (mean, zero, blur, attention).
- Works with both **PyTorch and JAX/Flax** models through a framework-agnostic callable interface.
- Efficient batched coalition evaluation with configurable `batch_size` and correct device handling.
- Comprehensive tests validating correctness against `ExactComputer` and covering every player × masking combination.
- Integration into the existing explainer framework with sensible auto-defaults per model type.
- All existing tests and pre-commit checks must continue to pass (`uv run pre-commit run --all-files`, `uv run pytest tests/shapiq`).

**Demo:**

- Polished, reproducible demo (notebooks, Gradio app, or HF Space) covering at least three SOTA vision models, multiple masking strategies, higher-order interactions, and a reproduction of a recent paper (FixLIP or equivalent).
- Performance report: wall-clock and memory benchmarks against the existing hard-coded benchmark game on at least two model families.
