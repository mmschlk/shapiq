# Tree-explainer runtime figures

Benchmark scripts behind the figures for the write-up on the tree-explainer work:
[#588](https://github.com/mmschlk/shapiq/pull/588) (Quadrature-TreeSHAP + the
numerical-precision fix for the path-dependent explainers) and
[#590](https://github.com/mmschlk/shapiq/pull/590) (InterventionalTreeSHAPIQ in C++ +
the vectorized k-SII aggregation). Both are merged; every result file here is measured
against `main` with those in it.

```
fetch_data.py           download the three TabArena datasets into $SHAPIQ_BENCH_DATA
bench_common.py         single-thread environment, dataset loaders, timing helpers
bench_interventional.py interventional: shapiq vs. Woodelf vs. shap over n, to n = 10,000
bench_depth.py          path-dependent: runtime vs. tree depth, real and synthetic trees
bench_accuracy.py       the same trees, but measuring round-off instead of runtime
bench_ksii_isolated.py  the k-SII aggregation timed alone, on frozen inputs
profile_hotspots.py     the three diagnoses behind the fixes in #590
run_final.sh            re-measure everything, in order, against one build
make_panels.py          render results/*.json as one plot per panel (the post figures)
make_figures.py         render the same results as the original multi-panel figures
```

## Running

```bash
export SHAPIQ_BENCH_DATA=~/bench_data
uv run python benchmarks/pr588/fetch_data.py
rm -rf build && uv run python setup.py build_ext --inplace   # never skip this
benchmarks/pr588/run_final.sh                                # ~50 min, single thread
uv run python benchmarks/pr588/make_panels.py
```

`run_final.sh` is the whole measurement, in the order the figures need it. The individual
scripts take `--tag <name>` so two versions' results can sit side by side.

The scripts need `main`'s C extensions built in place, plus `shap` and the optional
`woodelf-explainer` package (`uv sync --extra tree`).

**Rebuild before measuring.** A stale in-place build silently measures the previous version:
the extensions compile from one listed source each, and editing a file that is only
`#include`d does not trigger a rebuild (see `CLAUDE.md`). `rm -rf build` before
`build_ext --inplace` is the only reliable way to be sure of what is being timed.

## Measurement rules

* Single thread throughout: `bench_common` pins `OMP_NUM_THREADS` and friends to 1 *before*
  numpy is imported.
* Median of at most `--repeats` runs with the warm-up call excluded. A configuration stops
  contributing further runs once it has spent 20 s, and a curve stops being extended once one
  of its measurements passes 20 s (marked ✕ in the figures).
* **Figure 1** reports *end-to-end* time (explainer construction + explanation of `n`
  instances). That is the only fair basis there: the Woodelf fast path re-parses the model on
  every `explain_X` call and cannot amortize its setup through shapiq's public API.
* **Figures 2 and 3** report *single-explanation* time with construction excluded and recorded
  separately (`construct_s` in the JSON), matching the convention of the PR's benchmark
  comment.
* Every run asserts that the compared backends compute the same values: figure 1 records the
  maximum absolute deviation between shapiq, Woodelf and shap in its metadata; figures 2 and 3
  record the efficiency error (`|Σ values + baseline − prediction|`) of every order-1 point.

## Data

All three datasets are TabArena members and live on OpenML
(`superconductivity` 43174, `heloc` 45023, `Bioresponse` 4134). `fetch_data.py` pulls
byte-identical copies from public mirrors because the machine these numbers were produced on
had no route to `openml.org`; it asserts the OpenML shapes after download. Models are fitted
on an 80 % train split (`random_state=0`), which is where the 17k×81 / 8.4k×23 / 3k×1776 row
counts in the figure subtitles come from.

The synthetic suite needs no download: it regenerates the rare-indicator regime of
[issue #545](https://github.com/mmschlk/shapiq/issues/545), where CART splits on a fresh
feature at every level so that *distinct features per decision path* equals the tree depth —
the quantity that governs the polynomial explainers' round-off.

## `ksii_aggregation.patch`

The order-2 profiling in `profile_hotspots.py` traced 97% of a k-SII explanation to
`aggregate_base_attributions`, which tests every intermediate value with `np.all(value == 0)` --
a full numpy reduction dispatch on what is a Python float on every single-instance explanation.
`ksii_aggregation.patch` is the fix against `src/shapiq/game_theory/aggregation.py`, kept as a
patch rather than applied so that the figures on this branch keep measuring shipped behaviour:

```bash
git apply benchmarks/pr588/ksii_aggregation.patch
```

Measured 3.9-5.4x on the k-SII path, with identical keys and bit-identical values on both the
scalar and the batched (array-valued) paths; `tests_game_theory` and the tree-explainer suite
pass unchanged (461 passed, 2 skipped).
