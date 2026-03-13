# shapiq Documentation Improvement Plan

> **Purpose:** Step-by-step instructions for a coding agent to overhaul the shapiq documentation.
> Each stage is self-contained and can be executed independently. Stages should be executed in order as later stages depend on earlier ones.
> The repo lives at `mmschlk/shapiq`. Source code is under `src/shapiq/`. Docs are under `docs/`.

---

## Stage 1 — Enforce `__all__` across all public submodules ✅ ALREADY DONE

**Summary:** This stage is complete. All major submodules already have explicit `__all__` lists defined: `approximator`, `explainer`, `imputer`, `plot`, `game_theory`, `datasets`, `tree`, `utils`, and the top-level `src/shapiq/__init__.py` (44 exported symbols). No action required.

### Acceptance criteria (verified)
- Every major submodule has an explicit `__all__`. ✓
- `python -c "import shapiq; print(shapiq.__all__)"` runs without error. ✓
- No private or abstract-base-only symbols appear in any `__all__`. ✓

---

## Stage 2 — Add a CI test that guards `__all__` completeness

**Summary:** Add a pytest test that automatically fails if someone adds a new concrete public class (e.g. a new approximator) without registering it in `__all__`. This prevents the "silent omission from docs" problem permanently, without any manual doc maintenance.

### Tasks

1. **Create `tests/test_public_api.py`** (or add to an existing API test file if one exists). Write one test per major category:

   ```python
   # Example pattern — replicate for each submodule
   import inspect
   import shapiq
   from shapiq.approximator._base import Approximator  # adjust import to actual base class

   def test_all_approximators_registered():
       """Every concrete Approximator subclass must appear in shapiq.approximator.__all__."""
       from shapiq import approximator as approx_module
       all_subclasses = {
           name for name, obj in inspect.getmembers(approx_module, inspect.isclass)
           if issubclass(obj, Approximator)
           and obj is not Approximator
           and not inspect.isabstract(obj)
           and not name.startswith("_")
       }
       exported = set(approx_module.__all__)
       missing = all_subclasses - exported
       assert not missing, (
           f"Concrete Approximator subclasses not listed in __all__: {missing}. "
           "Add them to shapiq/approximator/__init__.py __all__."
       )
   ```

2. **Repeat the pattern** for `Explainer`, `Game`, and any other base class that is likely to grow over time.

3. **Add this test to the existing CI workflow** (`.github/workflows/`) so it runs on every pull request. If a `tests_unit/` vs `tests_integration/` split exists (per the CHANGELOG it does), place this in `tests_unit/`.

### Acceptance criteria
- `pytest tests/test_public_api.py` passes on the current codebase.
- Manually adding a new class that subclasses `Approximator` without adding it to `__all__` causes the test to fail.

---

## Stage 3 — Restructure the Sphinx configuration

**Summary:** The core Sphinx infrastructure is already in place (Furo theme, `sphinx-gallery`, `sphinx-autodoc-typehints`, `autodoc`, `autosummary`, `napoleon`, `doctest`, `intersphinx`, `viewcode`, `nbsphinx`, `myst_parser` all configured in `docs/source/conf.py`). This stage only addresses the gaps: missing `autosummary_ignore_module_all`, missing `autodoc_member_order`, and ensuring intersphinx targets are correct.

> **Critical:** The codebase uses **Google-style docstrings** (per project convention). Do NOT change napoleon to NumPy mode. Do NOT add `numpydoc`.

### Tasks

1. **No new dependencies needed.** The existing `docs` dependency group in `pyproject.toml` already includes all necessary packages (`sphinx`, `furo`, `sphinx-gallery`, `sphinx-autodoc-typehints`, `sphinxcontrib-bibtex`, `nbsphinx`, `myst-parser`, `sphinx-copybutton`).

2. **Audit `docs/source/conf.py`** for these specific settings and add any that are missing:
   ```python
   # autosummary — ensure __all__ is respected
   autosummary_ignore_module_all = False

   # autodoc
   autodoc_member_order = "groupwise"
   autoclass_content = "both"

   # napoleon — Google-style (already set, just verify)
   napoleon_google_docstring = True
   napoleon_numpy_docstring = False

   # intersphinx — verify these targets are present
   intersphinx_mapping = {
       "python": ("https://docs.python.org/3", None),
       "numpy": ("https://numpy.org/doc/stable/", None),
       "sklearn": ("https://scikit-learn.org/stable", None),
       "matplotlib": ("https://matplotlib.org/stable", None),
   }
   ```

3. **Create `docs/source/_templates/autosummary/`** with a `class.rst` stub template to control how class pages are rendered (show `__init__`, list methods in a summary table).

### Acceptance criteria
- `cd docs && make html` completes without error.
- No warnings about missing extensions or unresolved references (or a clearly defined list of known acceptable warnings is documented).

---

## Stage 4 — Rebuild the API reference pages

**Summary:** Replace the current single recursive `autosummary` block with a structured API reference that organises classes by conceptual category, driven by `automodule` so that new additions flow through automatically via `__all__`.

### Tasks

1. **Create or rewrite `docs/source/api/index.rst`** with the following structure (adapt exact class names to what exists in the codebase):

   ```rst
   API Reference
   =============

   This page documents all public classes and functions in ``shapiq``.
   New additions appear here automatically once registered in the submodule's ``__all__``.

   Explainers
   ----------

   High-level interfaces for explaining ML model predictions.

   .. currentmodule:: shapiq

   .. autosummary::
      :toctree: generated/
      :nosignatures:
      :template: class.rst

      Explainer
      TabularExplainer
      TreeExplainer
      TabPFNExplainer

   Approximators
   -------------

   Algorithms for approximating Shapley values and interaction indices.

   .. automodule:: shapiq.approximator
      :members:
      :no-private-members:
      :show-inheritance:

   Exact Computation
   -----------------

   .. automodule:: shapiq.exact
      :members:
      :no-private-members:
      :show-inheritance:

   Interaction Values
   ------------------

   .. automodule:: shapiq.interaction_values
      :members:
      :no-private-members:

   Games
   -----

   .. automodule:: shapiq.games
      :members:
      :no-private-members:
      :show-inheritance:

   Plotting
   --------

   .. automodule:: shapiq.plot
      :members:
      :no-private-members:
   ```

   > **Note on the Approximators section:** Because approximators are the most likely category to grow, use `automodule` rather than a hand-curated `autosummary` list. This means new approximators added to `__all__` appear automatically. For Explainers, a short curated list is fine since there are few of them and they rarely change.

2. **Ensure `docs/source/index.rst`** (or the top-level toctree) includes `api/index` in its toctree.

3. **Delete or archive the old API reference file** if it exists (e.g. the file containing the current recursive `autosummary shapiq` block).

### Acceptance criteria
- All sections render without `WARNING: autodoc: failed to import` errors.
- Each public class listed in the relevant `__all__` has its own generated stub page.
- Clicking a class in the Approximators section opens a page with its docstring, parameters, and methods.

---

## Stage 5 — Migrate and restructure examples to sphinx-gallery

**Summary:** Convert the existing example notebooks/scripts into sphinx-gallery compatible `.py` files, organised into topic-based subdirectories. Each file must be executable at build time and produce real output.

### Tasks

1. **Create the top-level examples directory structure**:
   ```
   examples/
   ├── README.rst                         # gallery landing page text
   ├── basic/
   │   ├── README.rst
   │   ├── plot_shapley_values.py
   │   └── plot_interaction_indices.py
   ├── explainers/
   │   ├── README.rst
   │   ├── plot_tabular_explainer.py
   │   └── plot_tree_explainer.py
   ├── interaction_values/
   │   ├── README.rst
   │   ├── plot_network_plot.py
   │   └── plot_force_plot.py
   └── games/
       ├── README.rst
       └── plot_custom_game.py
   ```

2. **sphinx-gallery script format:** Each `.py` file must follow this structure:
   ```python
   """
   Title of the Example
   ====================

   A paragraph describing what this example demonstrates and what the reader
   will learn. This text appears as the example's description in the gallery.
   """
   # %%
   # Section heading (rendered as RST)
   # ----------------------------------
   # Explanatory prose goes here as a comment block.

   import shapiq
   # ... actual runnable code

   # %%
   # Next section
   # ------------
   # More explanation.

   # ... more code
   ```

3. **Migrate existing examples.** For each existing notebook or script in `examples/api_examples/`:
   - Convert to the sphinx-gallery `.py` format.
   - Ensure it runs end-to-end in under ~60 seconds with a small dataset (use `shapiq.datasets` loaders or synthetic data).
   - Rename to `plot_<descriptive_name>.py`.

4. **Each example file must:**
   - Import only from `shapiq` (no relative imports).
   - Use small data — if the original used a large dataset, replace with a smaller synthetic equivalent or a small built-in dataset from `shapiq.datasets`.
   - Produce at least one matplotlib figure (this becomes the gallery thumbnail).
   - End with `plt.show()` or `plt.tight_layout()` as appropriate.

5. **Write a `README.rst` for each subdirectory.** This text appears as the section header in the gallery. Example:
   ```rst
   Explainers
   ==========

   Examples showing how to use ``shapiq`` explainers to compute Shapley values
   and interaction indices for machine learning model predictions.
   ```

### Acceptance criteria
- `cd docs && make html` runs all gallery examples without error.
- The built docs contain a gallery page at `auto_examples/index.html` with thumbnails grouped by subdirectory.
- Each example page shows rendered output (figures and printed values) inline.

---

## Stage 6 — Add doctests to docstrings

**Summary:** Add `doctest`-style examples to the docstrings of the most important public classes and functions. These serve as both documentation and lightweight regression tests.

### Tasks

1. **Prioritise these classes/functions** for doctest coverage (add more as time allows):
   - `shapiq.Explainer.__init__` and `shapiq.Explainer.explain`
   - `shapiq.InteractionValues` — arithmetic operations, `__repr__`, `.plot_*` methods (use `# doctest: +ELLIPSIS` for output that varies)
   - `shapiq.ExactComputer`
   - At least 2–3 approximators (e.g. `SPEX`, `KernelSHAPIQ`)

2. **Doctest format** (use Google-style docstrings — the project convention):
   ```python
   def explain(self, x):
       """Explain a single prediction.

       Args:
           x: A single data point of shape (n_features,).

       Returns:
           The computed interaction values.

       Examples:
           >>> import numpy as np
           >>> import shapiq
           >>> rng = np.random.default_rng(42)
           >>> data = rng.random((100, 4))
           >>> model = lambda x: x.sum(axis=-1)
           >>> explainer = shapiq.TabularExplainer(model=model, data=data, index="SV")
           >>> iv = explainer.explain(data[0])
           >>> len(iv)  # number of interaction terms
           4
       """
   ```

3. **Configure doctest in `conf.py`** to allow ellipsis and normalise whitespace:
   ```python
   doctest_default_flags = (
       doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
   )
   ```

4. **Run doctests** as part of the Sphinx build by adding to the `Makefile` or CI:
   ```bash
   sphinx-build -b doctest docs/source docs/_build/doctest
   ```
   Or via pytest: `pytest --doctest-modules src/shapiq/`

5. **Do not write doctests for:**
   - Methods with non-deterministic output (unless seeded and output is fixed).
   - Plot-only methods (these cannot produce text output for doctest).
   - Private methods.

### Acceptance criteria
- `pytest --doctest-modules src/shapiq/` passes for all files that have doctests.
- Alternatively, `make doctest` in the docs directory passes.
- At least the classes listed in Task 1 have at least one doctest example each.

---

## Stage 7 — Write the User Guide

**Summary:** Create a prose-first user guide that explains *why* and *when*, not just *how*. This is the layer that makes shapiq accessible to users who are not already experts in game theory. Each page should be standalone readable RST, not a rehash of the API reference.

### Tasks

1. **Create the directory and toctree entry:**
   ```
   docs/source/user_guide/
   ├── index.rst
   ├── concepts.rst
   ├── choosing_an_index.rst
   ├── choosing_an_approximator.rst
   ├── explaining_models.rst
   ├── interaction_values.rst
   ├── custom_games.rst
   └── comparison_with_shap.rst
   ```
   Add `user_guide/index` to the main `docs/source/index.rst` toctree.

2. **Write each page** at the level of a well-informed user who knows ML but may not know game theory. Suggested content per page:

   - **`concepts.rst`** — What are cooperative games? What are Shapley values? What are interaction indices? Why does order matter? Keep math minimal; link to the NeurIPS paper for depth.
   - **`choosing_an_index.rst`** — Decision guide: SV vs SII vs k-SII vs FBII vs STI. When to use each. A comparison table.
   - **`choosing_an_approximator.rst`** — Budget vs accuracy tradeoffs. When to use SPEX vs KernelSHAPIQ vs PermutationSampling. Practical guidance on `budget` parameter.
   - **`explaining_models.rst`** — End-to-end walkthrough: load data, fit a sklearn model, use `TabularExplainer`, use `TreeExplainer`, visualise with `plot_network` and `plot_force`. Include code blocks (not gallery examples — these are illustrative inline snippets).
   - **`interaction_values.rst`** — What is the `InteractionValues` object? How to index into it, do arithmetic, convert between indices, subset players.
   - **`custom_games.rst`** — How to subclass `shapiq.Game` to use shapiq with a non-ML application (e.g. coalition games, data valuation).
   - **`comparison_with_shap.rst`** — Side-by-side: how to replicate common `shap` workflows in `shapiq`. Useful for migration.

3. **Cross-link liberally** between user guide pages and API reference pages using `:class:`, `:func:`, `:meth:` roles so users can jump from explanation to reference and back.

4. **Link user guide pages to relevant gallery examples** using the sphinx-gallery `.. minigallery::` directive at the bottom of each page where examples exist.

### Acceptance criteria
- All 7 user guide pages exist, build without warnings, and are reachable from the docs landing page.
- Each page contains at least one cross-reference to an API page and one link to a related gallery example (where one exists).
- `make html` produces no broken references within the user guide.

---

## Stage 8 — Polish the landing page and navigation

**Summary:** The `index.rst` landing page is the first thing users see. It should orient three different audiences — practitioners, researchers, and contributors — and route them to the right place immediately.

### Tasks

1. **Rewrite `docs/source/index.rst`** to include:
   - A one-paragraph description of what shapiq does and who it is for.
   - A `.. grid::` or simple toctree-based navigation block with three sections clearly labelled:
     - **Getting Started** → installation + quick example
     - **User Guide** → link to `user_guide/index`
     - **Examples** → link to `auto_examples/index`
     - **API Reference** → link to `api/index`
   - A short "If you are coming from SHAP" callout box pointing to `comparison_with_shap.rst`.
   - The citation block (BibTeX for the NeurIPS paper).

2. **Create or update `docs/source/getting_started.rst`** with:
   - Installation instructions (`pip install shapiq`, `uv add shapiq`).
   - A complete, copy-pasteable 10-line example that goes from data → explainer → plot.
   - Links onward to the user guide and examples gallery.

3. **Verify the sidebar / navigation** in the chosen Sphinx theme correctly shows all four top-level sections. If using Furo or PyData Sphinx Theme, this may require explicit `html_theme_options` configuration.

### Acceptance criteria
- The built docs landing page clearly shows the four navigation sections.
- A new user landing on the page can reach a working code example within two clicks.
- The getting started page example runs correctly with the current installed version of shapiq.

---

## Stage 9 — CI integration and ReadTheDocs configuration

**Summary:** Both the CI (`doc_build` job in `.github/workflows/ci.yml`) and `.readthedocs.yaml` already exist and are well-configured. `.readthedocs.yaml` uses Python 3.12, `uv sync --group docs`, pandoc, and a custom `copy_notebooks.py` pre-build step. The only remaining gap is the `plot_gallery=False` optimisation for PR builds (to keep CI fast) and uploading a docs artifact for review.

### Tasks

1. **`.readthedocs.yaml` is already complete.** No changes needed. It uses `uv`, Python 3.12, installs the `docs` group, and builds with `sphinx-build`.

2. **Update the existing `doc_build` job in `.github/workflows/ci.yml`** (do not create a new file) to add the `plot_gallery=False` flag for PR speed and an artifact upload:
   ```yaml
   name: Docs
   on:
     pull_request:
       paths:
         - "docs/**"
         - "src/shapiq/**"
         - "examples/**"
   jobs:
     build-docs:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v4
         - uses: actions/setup-python@v5
           with:
             python-version: "3.11"
         - run: pip install ".[docs]"
         - name: Build docs (no gallery execution on PRs)
           run: |
             cd docs
             sphinx-build -b html source _build/html \
               -D sphinx_gallery_conf.plot_gallery=False
         - name: Upload docs artifact
           uses: actions/upload-artifact@v4
           with:
             name: docs-html
             path: docs/_build/html/
   ```
   The key setting is `plot_gallery=False` for PR builds — this makes the check fast. The full gallery only runs on ReadTheDocs for the main branch and release tags.

3. **Add `[docs]` to `pyproject.toml` optional dependencies:**
   ```toml
   [project.optional-dependencies]
   docs = [
     "sphinx>=7.0",
     "sphinx-gallery>=0.16",
     "furo",                         # or current theme
     "sphinx-autodoc-typehints",
     "numpydoc",
   ]
   ```

4. **Add a `CONTRIBUTING.md` section** (or update the existing one) explaining how contributors should:
   - Add their class to the relevant `__all__`.
   - Write a docstring with at least a one-line summary and a `Parameters` section.
   - Optionally add a gallery example under `examples/`.

### Acceptance criteria
- The GitHub Actions docs workflow passes on a clean PR.
- ReadTheDocs successfully builds the full docs including gallery on push to `main`.
- `pyproject.toml` has a `[docs]` extras group that installs all necessary dependencies.

---

## Summary table

| Stage | What it does | Status | Touches |
|-------|-------------|--------|---------|
| 1 | Enforce `__all__` in all submodules | ✅ Done | — |
| 2 | CI test guarding `__all__` completeness | 🔲 To do | `tests/test_public_api.py`, CI workflow |
| 3 | Sphinx configuration (gaps only) | 🔲 To do (minor) | `docs/source/conf.py` |
| 4 | Structured API reference | 🔲 To do | `docs/source/api/` |
| 5 | sphinx-gallery examples | 🔲 To do (only 1 example exists) | `docs/source/examples/` |
| 6 | Doctests in docstrings (Google-style) | 🔲 To do | `src/shapiq/**/*.py` |
| 7 | User guide prose | 🔲 To do (highest value) | `docs/source/user_guide/` |
| 8 | Landing page and navigation | 🔲 To do | `docs/source/index.rst` |
| 9 | CI + ReadTheDocs wiring | 🔲 To do (minor — CI + `.readthedocs.yaml` exist; just add `plot_gallery=False` + artifact upload) | `.github/workflows/ci.yml` |

Stage 1 is already done. Stage 2 is the new prerequisite guard. Stages 3–4 can be done in parallel with 5–6. Stage 7 is independent and can be written in parallel with any other stage. Stages 8–9 should be done last.
