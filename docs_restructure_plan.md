# Restructure API Reference Documentation

## Context

The current API reference is a plain toctree linking to category pages that use `automodule` to dump all classes/functions into one unstructured wall of text. No summary tables, no logical sub-grouping, no class names visible in navigation. The goal is a sklearn-style API reference with summary tables, clear organization, and navigable structure.

## Decisions Made

- **Overview page**: Single page with `list-table` tables organized by category (not separate pages)
- **Detail pages**: Keep inline docs on category pages (no per-class stub pages)
- **Approximators grouping**: By algorithm family (Permutation, Regression, Monte Carlo, Marginal, Sparse, Proxy)
- **Sidebar**: Right-side "On this page" TOC is sufficient (classes appear there via section headers)

## Implementation

### Step 1: Delete auto-generated stub files

Remove all `.rst` files in these subdirectories (they become orphans):
- `docs/source/api/approximators/` (17 files)
- `docs/source/api/plot/` (9 files)
- `docs/source/api/utils/` (4 files)
- `docs/source/api/datasets/` (3 files)

```bash
rm -r docs/source/api/approximators/ docs/source/api/plot/ docs/source/api/utils/ docs/source/api/datasets/
```

### Step 2: Rewrite `docs/source/api_reference.rst`

Replace the plain toctree with a structured overview page:

- Hidden `toctree` to keep sidebar navigation to category pages
- `list-table` per category with `:class:`/`:func:` cross-references (tilde prefix for short names)
- Each table: columns "Class/Function" and "Description"
- Approximators section has sub-tables per algorithm family
- Brief intro text per category with `:doc:` link to the full category page

### Step 3: Rewrite category pages (7 files)

Replace `automodule` with individual `autoclass`/`autofunction` directives under section headers. Each page gets:
1. Page title + intro text
2. `.. currentmodule:: shapiq`
3. Top-of-page `autosummary` table (no `:toctree:`) for quick navigation
4. Section header per class/function (enables right-side TOC)
5. `autoclass`/`autofunction` directive under each header

#### `docs/source/api/core.rst`
Sections: `Game`, `InteractionValues`, `ExactComputer`

#### `docs/source/api/explainers.rst`
Sections: `Explainer`, `TabularExplainer`, `TabPFNExplainer`, `AgnosticExplainer`, `TreeExplainer`

#### `docs/source/api/approximators.rst` (most complex)
Sub-grouped by algorithm family with section headers:
- **Permutation-based**: `PermutationSamplingSV`, `PermutationSamplingSII`, `PermutationSamplingSTII`
- **Regression-based**: `KernelSHAP`, `UnbiasedKernelSHAP`, `kADDSHAP`, `KernelSHAPIQ`, `InconsistentKernelSHAPIQ`, `RegressionFSII`, `RegressionFBII`
- **Monte Carlo**: `SHAPIQ`, `SVARM`, `SVARMIQ`
- **Marginal Sampling**: `OwenSamplingSV`, `StratifiedSamplingSV`
- **Sparse**: `SPEX`, `ProxySPEX`
- **Proxy**: `ProxySHAP`, `MSRBiased` (these live in `shapiq.approximator`, not top-level `shapiq`)
- **Approximator Groups** (constants): `SV_APPROXIMATORS`, `SI_APPROXIMATORS`, `SII_APPROXIMATORS`, `STII_APPROXIMATORS`, `FSII_APPROXIMATORS`, `FBII_APPROXIMATORS`

Note for Proxy section: Use `.. currentmodule:: shapiq.approximator` before the autoclass directives, then reset back to `.. currentmodule:: shapiq` after. In autosummary, reference as `approximator.ProxySHAP` and `approximator.MSRBiased`. In the overview page list-table, use `:class:\`~shapiq.approximator.ProxySHAP\`` and `:class:\`~shapiq.approximator.MSRBiased\``.

For Approximator Groups, use `.. autodata:: shapiq.approximator.SV_APPROXIMATORS` etc.

#### `docs/source/api/imputers.rst`
Sections: `MarginalImputer`, `GenerativeConditionalImputer`, `BaselineImputer`, `TabPFNImputer`, `GaussianImputer`, `GaussianCopulaImputer`

#### `docs/source/api/plotting.rst`
Sections per plot function: `bar_plot`, `beeswarm_plot`, `force_plot`, `network_plot`, `sentence_plot`, `si_graph_plot`, `stacked_bar_plot`, `upset_plot`, `waterfall_plot`, `abbreviate_feature_names`

Note: `abbreviate_feature_names` lives in `shapiq.plot`, not top-level. Use `plot.abbreviate_feature_names` in autosummary and `.. autofunction:: shapiq.plot.abbreviate_feature_names` for the directive.

#### `docs/source/api/datasets.rst`
Sections: `load_bike_sharing`, `load_adult_census`, `load_california_housing`

#### `docs/source/api/utilities.rst`
Sub-grouped:
- **Sets & Coalitions**: `powerset`, `get_explicit_subsets`, `split_subsets_budget`, `utils.pair_subset_sizes`, `utils.generate_interaction_lookup`, `utils.generate_interaction_lookup_from_coalitions`, `utils.transform_coalitions_to_array`, `utils.transform_array_to_coalitions`, `utils.count_interactions`
- **Module Utilities**: `safe_isinstance`, `utils.check_import_module`
- **Data Utilities**: `utils.shuffle_data`

Note: Functions not in top-level `shapiq.__all__` (e.g. `pair_subset_sizes`, `check_import_module`, `shuffle_data`) need fully qualified `shapiq.utils.` prefix in autofunction directives. In autosummary tables, use `utils.` prefix.

### Step 4: No `conf.py` changes needed

Existing autodoc/autosummary settings work with the new approach. The switch from `automodule` to `autoclass`/`autofunction` is compatible with current `autodoc_default_options`.

## Files Modified

| File | Action |
|------|--------|
| `docs/source/api_reference.rst` | Rewrite: toctree → list-tables |
| `docs/source/api/core.rst` | Rewrite: automodule → autoclass sections |
| `docs/source/api/explainers.rst` | Rewrite: automodule → autoclass sections |
| `docs/source/api/approximators.rst` | Rewrite: automodule → grouped autoclass sections |
| `docs/source/api/imputers.rst` | Rewrite: automodule → autoclass sections |
| `docs/source/api/plotting.rst` | Rewrite: automodule → autofunction sections |
| `docs/source/api/datasets.rst` | Rewrite: automodule → autofunction sections |
| `docs/source/api/utilities.rst` | Rewrite: automodule → autofunction sections |
| `docs/source/api/approximators/*.rst` | Delete (17 stub files) |
| `docs/source/api/plot/*.rst` | Delete (9 stub files) |
| `docs/source/api/utils/*.rst` | Delete (4 stub files) |
| `docs/source/api/datasets/*.rst` | Delete (3 stub files) |

## Verification

1. Build docs: `cd docs && uv run make html` (or `uv run sphinx-build docs/source docs/build -W`)
2. Check no broken cross-references (warnings-as-errors flag)
3. Open `docs/build/api_reference.html` — verify summary tables render with working links
4. Open category pages — verify right-side "On this page" TOC shows class/function names
5. Verify approximators page has clear algorithm family groupings
6. Click cross-references on overview page → should land on correct class on category page
