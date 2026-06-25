"""Regression tests for native-extension / OpenMP-runtime isolation.

Background: on macOS, shapiq's compiled extensions used to each vendor a separate
``libomp.dylib``. macOS coalesces *weak* symbols across the whole process, so a
second libomp image (e.g. xgboost's Homebrew libomp) could cross into shapiq's
copy mid-barrier and crash (``EXC_BAD_ACCESS``). The fixes guarded here:

* Only the ``interventional`` extension uses OpenMP, and it is imported lazily so
  ``import shapiq`` pulls no OpenMP runtime into the process.
* On macOS that extension links libomp statically with hidden symbols, so it
  exports only ``_PyInit_cext`` and carries no dynamic ``libomp`` load command.
* The ``conversion`` and ``linear`` extensions link no libomp at all.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys

import pytest

# Extensions that must never carry a dynamic libomp dependency.
_ALL_CEXT = (
    "shapiq.tree.interventional.cext",
    "shapiq.tree.conversion.cext",
    "shapiq.tree.linear.cext",
)

# Symbol-name fragments that must never be exported from the interventional
# extension (statically-linked OpenMP runtime + our own internal symbols).
_FORBIDDEN_EXPORT_FRAGMENTS = ("kmp", "omp", "GOMP", "algorithms")


def _cext_path(module_name: str) -> str:
    """Return the on-disk path of a compiled extension without importing it."""
    spec = importlib.util.find_spec(module_name)
    assert spec is not None and spec.origin is not None, f"{module_name} not found"
    return spec.origin


def _run_python(code: str, env_extra: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    """Run ``code`` in a fresh interpreter and capture the result.

    Used for checks that must observe a clean process (import order) or that may
    hard-crash via a native segfault, which cannot be caught in-process.
    """
    import os

    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
        check=False,
    )


def test_import_shapiq_does_not_load_interventional_cext():
    """``import shapiq`` must not eagerly load the OpenMP-linked interventional cext."""
    result = _run_python(
        "import shapiq, sys;"
        "assert 'shapiq.tree.interventional.cext' not in sys.modules,"
        " 'interventional cext loaded eagerly';"
        "print('OK')"
    )
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert "OK" in result.stdout


def test_interventional_cext_loads_only_on_use():
    """Running the interventional explainer triggers the lazy cext import."""
    result = _run_python(
        "import sys;"
        "import numpy as np;"
        "from sklearn.tree import DecisionTreeRegressor;"
        "import shapiq;"
        "from shapiq.tree import InterventionalTreeExplainer;"
        "assert 'shapiq.tree.interventional.cext' not in sys.modules;"
        "X = np.random.RandomState(0).rand(40, 4);"
        "y = X[:, 0] + X[:, 1];"
        "m = DecisionTreeRegressor(max_depth=3).fit(X, y);"
        "expl = InterventionalTreeExplainer(m, X, index='SV', max_order=1, debug=False);"
        "expl.explain_function(X[0]);"
        "assert 'shapiq.tree.interventional.cext' in sys.modules, 'cext never loaded';"
        "print('OK')"
    )
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert "OK" in result.stdout


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS static+hidden linking check")
def test_interventional_cext_exports_only_module_init():
    """The interventional cext must export only ``_PyInit_cext`` (no OpenMP symbols)."""
    path = _cext_path("shapiq.tree.interventional.cext")
    out = subprocess.run(["nm", "-gU", path], capture_output=True, text=True, check=True).stdout
    exported = [line.split()[-1] for line in out.splitlines() if line.strip()]
    assert "_PyInit_cext" in exported, f"module init not exported; got {exported}"
    leaked = [sym for sym in exported if any(frag in sym for frag in _FORBIDDEN_EXPORT_FRAGMENTS)]
    assert not leaked, f"forbidden symbols exported (coalescing surface): {leaked}"


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS dynamic-linkage check")
@pytest.mark.parametrize("module_name", _ALL_CEXT)
def test_cext_has_no_dynamic_libomp(module_name):
    """No shapiq extension may carry a dynamic ``libomp`` load command on macOS."""
    path = _cext_path(module_name)
    out = subprocess.run(["otool", "-L", path], capture_output=True, text=True, check=True).stdout
    assert "libomp" not in out, f"{module_name} links a dynamic libomp:\n{out}"


@pytest.mark.parametrize("omp_threads", [None, "1"])
def test_shapiq_xgboost_openmp_coexistence(omp_threads):
    """shapiq + xgboost must not crash when both OpenMP-using paths run.

    Imports shapiq *first* (the historically crashing order), then fits a
    multi-threaded XGBoost model and runs an interventional explanation. A libomp
    coalescing failure manifests as a hard segfault, so this runs in a subprocess
    and asserts a clean exit.
    """
    pytest.importorskip("xgboost")
    code = (
        "import shapiq;"  # import shapiq (and its libomp) first
        "import numpy as np;"
        "import xgboost as xgb;"
        "from shapiq.tree import InterventionalTreeExplainer;"
        "rng = np.random.RandomState(0);"
        "X = rng.rand(128, 6); y = X[:, 0] + X[:, 1] * X[:, 2];"
        "m = xgb.XGBRegressor(n_estimators=8, max_depth=3, n_jobs=4).fit(X, y);"
        "expl = InterventionalTreeExplainer(m, X, index='SV', max_order=1, debug=False);"
        "expl.explain_function(X[0]);"
        "print('OK')"
    )
    env_extra = {} if omp_threads is None else {"OMP_NUM_THREADS": omp_threads}
    result = _run_python(code, env_extra=env_extra)
    assert result.returncode == 0, (
        f"coexistence crashed (rc={result.returncode}); "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert "OK" in result.stdout
    assert "Error #15" not in result.stderr, f"OpenMP duplicate-runtime abort:\n{result.stderr}"
