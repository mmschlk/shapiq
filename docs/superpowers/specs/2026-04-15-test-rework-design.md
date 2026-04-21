# Test Rework Design Spec

**Date:** 2026-04-15
**Branch:** `rework-tests`
**Scope:** `tests/shapiq/` only (`shapiq_games` excluded)

## Problem

The current test suite has 321 tests across 75 files (~10k lines of test code) testing ~20.6k lines
of source code. Key issues:

1. **Noise:** 23 nearly identical approximator test files. Tree explainer tests repeat the same
   pattern per model type. Plot tests are 13 files for smoke tests.
2. **Slow:** 3-5 minutes. Parametrization explosion (budgets x orders x imputers), repeated model
   fitting, no tiering between fast and slow tests.
3. **Low confidence:** Approximator tests use 40% tolerances (`pytest.approx(1.0, 0.4)`). Many
   tests only check `isinstance(result, InteractionValues)` — smoke tests dressed as unit tests.
4. **Development friction:** Adding a new approximator requires creating a new test file and
   copy-pasting the same pattern. No shared infrastructure enforcing correctness contracts.

## Goals

- **Correctness:** Tests verify contracts and mathematical properties, not rough estimates.
- **Development velocity:** Adding a new approximator = adding one fixture/dict entry. Protocol
  tests pick it up automatically.
- **Speed:** ~1 minute default, ~2-3 minutes with slow/CI markers.
- **Readability:** Every test is easy to understand. No navigating 75 files.

## Design

### File Structure

```
tests/shapiq/
├── conftest.py                     # shared fixtures: games, tiny datasets, model factories
├── test_approximators.py           # protocol + TestClass special cases
├── test_explainers.py              # protocol + TestClass special cases
├── test_tree.py                    # protocol + TestClass special cases
├── test_imputers.py                # protocol + TestClass special cases
├── test_interaction_values.py      # data structure correctness
├── test_game_theory.py             # exact computer, indices, moebius converter
├── test_plots.py                   # smoke tests
└── test_public_api.py              # import surface checks
```

**8 files.** Down from 75. Split a file only if it exceeds ~400 lines.

### Tiering

Two tiers via pytest markers:

- **Default (no marker):** Runs in ~1 minute. Small games (3-7 players), tiny datasets (30 samples,
  5 features), no optional dependencies. This is the "I changed a line" feedback loop.
- **`@pytest.mark.slow`:** CI only, adds optional-dependency tests (TabPFN, heavy models, SHAP
  comparison). Total ~2-3 minutes.

Configuration in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
markers = ["slow: marks tests that require > 5s or optional deps (deselect with '-m not slow')"]
addopts = "-m 'not slow'"
```

`pytest` runs the fast suite. `pytest -m ''` runs everything.

### Fixtures & Shared Infrastructure

All fixtures live in `conftest.py`. Key principles:

- **No files on disk.** Everything computed or constructed in fixtures.
- **Small games for protocol tests.** 3-7 players so ExactComputer runs in microseconds.
- **Tiny datasets.** 30 samples, 5 features — enough to fit a real model, fast enough not to matter.
- **Lazy model creation.** Models only instantiated when a test requests them.
- **`module` or `session` scope** for expensive fixtures (model fitting).
- **Dependency on `shapiq_games`** for `DummyGame`, `SOUM`, and other test games.

#### Example fixtures

```python
@pytest.fixture
def dummy_game():
    """3-player DummyGame -- fast, deterministic."""
    return DummyGame(n=3, interaction=(0, 1))

@pytest.fixture
def exact_values(dummy_game):
    """Ground truth via ExactComputer."""
    return ExactComputer(dummy_game).compute(index="SII", order=2)

@pytest.fixture
def tiny_data():
    """30 samples, 5 features."""
    rng = np.random.default_rng(42)
    return rng.normal(size=(30, 5))

@pytest.fixture(scope="module")
def dt_reg_model(tiny_data):
    """Tiny DecisionTreeRegressor. Fits in <10ms."""
    from sklearn.tree import DecisionTreeRegressor
    model = DecisionTreeRegressor(max_depth=3, random_state=42)
    model.fit(tiny_data, tiny_data[:, 0])
    return model
```

#### Approximator registry

```python
ALL_APPROXIMATORS = [
    {"cls": SHAPIQ, "n": 7, "max_order": 2, "index": "SII", "budget": 100},
    {"cls": SHAPIQ, "n": 7, "max_order": 2, "index": "FSII", "budget": 100},
    {"cls": KernelSHAPIQ, "n": 7, "max_order": 2, "index": "k-SII", "budget": 100},
    {"cls": PermutationSV, "n": 7, "max_order": 1, "index": "SV", "budget": 50},
    # ... one entry per (approximator, index) combo worth testing
]
```

Adding a new approximator = append one dict. Protocol tests pick it up automatically.

### Protocol Tests

Protocols test **contracts**, not quality. Every component of a given type must pass its protocol.

#### Approximator protocol

```python
@pytest.mark.parametrize("config", ALL_APPROXIMATORS,
                         ids=lambda c: f"{c['cls'].__name__}-{c['index']}")
class TestApproximatorProtocol:

    def test_returns_interaction_values(self, config, dummy_game):
        """Result is an InteractionValues with correct metadata."""
        approx = config["cls"](n=config["n"], max_order=config["max_order"],
                               index=config["index"])
        result = approx.approximate(config["budget"], dummy_game)
        assert isinstance(result, InteractionValues)
        assert result.index == config["index"]
        assert result.max_order == config["max_order"]

    def test_respects_budget(self, config, dummy_game):
        """Never exceeds declared budget (+ small epsilon for setup calls)."""
        approx = config["cls"](n=config["n"], max_order=config["max_order"],
                               index=config["index"])
        approx.approximate(config["budget"], dummy_game)
        assert dummy_game.access_counter <= config["budget"] + 2

    def test_reproducible(self, config, dummy_game):
        """Same random_state produces identical results."""
        a1 = config["cls"](n=config["n"], max_order=config["max_order"],
                           index=config["index"], random_state=42)
        a2 = config["cls"](n=config["n"], max_order=config["max_order"],
                           index=config["index"], random_state=42)
        r1 = a1.approximate(config["budget"], dummy_game)
        r2 = a2.approximate(config["budget"], dummy_game)
        assert np.allclose(r1.values, r2.values)

    def test_rejects_invalid_index(self, config):
        """Raises ValueError for indices not in valid_indices."""
        with pytest.raises((ValueError, TypeError)):
            config["cls"](n=config["n"], max_order=config["max_order"],
                          index="INVALID_INDEX")

    def test_efficiency(self, config, dummy_game):
        """For efficient indices, sum(values) ~ game(N) - game(empty)."""
        if config["index"] not in EFFICIENT_INDICES:
            pytest.skip("Efficiency doesn't apply to this index")
        approx = config["cls"](n=config["n"], max_order=config["max_order"],
                               index=config["index"], random_state=42)
        result = approx.approximate(config["budget"], dummy_game)
        expected_sum = dummy_game({0, 1, 2}) - dummy_game(set())
        assert sum(result.values) == pytest.approx(expected_sum, abs=0.3)
```

#### Explainer protocol

Same pattern: returns `InteractionValues`, efficiency holds, reproducible, handles single instance
and batch (`explain` and `explain_X`).

#### Imputer protocol

`fit` + `impute` work, output shape is correct, handles coalitions properly.

#### Tree protocol

Conversion produces valid `TreeModel`, SV matches ExactComputer, efficiency holds, baseline matches
empty prediction.

### Special Cases

Special cases live as `TestClass` groups below the protocol in the same file. Only add one when a
component has genuinely unique behavior the protocol can't cover.

```python
# In test_approximators.py, below protocol:

class TestProxySHAP:
    """ProxySHAP-specific: MSRBiased variant, coalition-to-tree-path conversion."""
    def test_msr_biased_coalitions_to_paths(self): ...

class TestSPEX:
    """SPEX requires larger budgets."""
    def test_warns_on_small_budget(self): ...
```

Most approximators won't have a `TestClass`.

### Tree Special Cases

`test_tree.py` is the densest domain:

```python
# Protocol
class TestTreeProtocol:
    def test_conversion_produces_valid_tree_model(self, tree_model): ...
    def test_sv_matches_exact(self, tree_model, x_explain): ...
    def test_efficiency(self, tree_model, x_explain): ...
    def test_baseline_matches_empty_prediction(self, tree_model): ...

# SHAP comparison -- slow, CI-only
@pytest.mark.slow
class TestSHAPComparison:
    """Expected values as in-file constants, not loaded from disk."""
    def test_xgb_regression_matches_shap(self, xgb_reg_model, background_reg_data): ...
    def test_rf_classification_matches_shap(self, rf_clf_model, background_clf_data): ...

# Edge cases
class TestTreeEdgeCases:
    def test_decision_stumps(self): ...
    def test_high_dimensional_no_overflow(self): ...
    def test_repeated_calls_no_segfault(self): ...
```

`test_tree.py` will likely be the largest file (~300-400 lines). That's acceptable given domain
complexity.

### What Gets Deleted

**Entire directories:**
- `tests/shapiq/tests_unit/` (all 68 test files)
- `tests/shapiq/tests_integration_tests/` (California Housing tests)
- `tests/shapiq/tests_deprecation/`
- `tests/shapiq/fixtures/` (replaced by `conftest.py`)
- `tests/shapiq/data/` (pre-computed JSON, model files, test images)

**Files:**
- `tests/shapiq/markers.py` (skip markers move into `conftest.py`)
- `tests/shapiq/utils.py` (absorbed into protocols or removed)

**What we preserve (migrated into new files):**
- Segfault regression tests (tree overflow, refcount corruption) -> `test_tree.py`
- SHAP comparison values (as in-file constants) -> `test_tree.py` slow suite
- `InteractionValues` tests (condensed from 896 lines to ~200-300) -> `test_interaction_values.py`
- ExactComputer tests -> `test_game_theory.py`
- Skip markers for optional deps -> `conftest.py`

### Net Effect

| Metric | Before | After |
|--------|--------|-------|
| Test files | 75 | 8 |
| Lines of test code | ~10,000 | ~1,500-2,000 |
| Test functions | 321 | ~80-120 |
| Default runtime | 3-5 min | ~1 min |
| CI runtime | 3-5 min | ~2-3 min |
| Files to touch for new approximator | 1 new file | 1 dict entry |

### Migration Approach

We write the new suite fresh based on protocols, then verify coverage is adequate before deleting
the old tests. The old tests exist on `main` as reference if we need to check what a specific test
was validating. This is not a port -- it's a rewrite.
