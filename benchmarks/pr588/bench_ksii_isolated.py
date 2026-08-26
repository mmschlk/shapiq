"""Time ``aggregate_base_attributions`` alone, on inputs frozen across shapiq versions.

The end-to-end ``explain()`` numbers cannot isolate PR #590's second improvement, because
after it the aggregation is no longer the whole call. This freezes real order-2 SII inputs to
a pickle on first run and then times the aggregation against them, so the identical inputs can
be replayed against a different checkout::

    python bench_ksii_isolated.py main      # on one checkout
    python bench_ksii_isolated.py pr590     # on the other

Writes ``results/ksii_<label>.json``. ``results/ksii_isolated.json`` is the combined
three-variant comparison that figure 4 reads (main, PR 590, and PR 590 with the subset codes
built over compacted feature ids).
"""

from __future__ import annotations

import json
import pickle
import statistics
import sys
import time
import warnings
from typing import TYPE_CHECKING

from bench_common import RESULTS, load_bioresponse, load_heloc, load_superconductivity, quiet
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from shapiq.game_theory.aggregation import aggregate_base_attributions
from shapiq.tree import QuadratureTreeSHAP

if TYPE_CHECKING:
    from collections.abc import Callable

CACHE = RESULTS / "ksii_inputs.pkl"

CASES = (
    ("superconductivity d8", load_superconductivity, 8, "regression"),
    ("superconductivity d12", load_superconductivity, 12, "regression"),
    ("bioresponse d16", load_bioresponse, 16, "classification"),
    ("heloc d16", load_heloc, 16, "classification"),
)


def median_ms(fn: Callable[[], object], *, runs: int = 9, warmup: int = 2) -> float:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples) * 1e3


def freeze_inputs() -> dict:
    """Compute the order-2 SII base values once and pin them, so both versions see one input."""
    cases = {}
    for name, loader, depth, task in CASES:
        data = loader()
        cls = DecisionTreeClassifier if task == "classification" else DecisionTreeRegressor
        tree = cls(max_depth=depth, random_state=0).fit(data["X_train"], data["y_train"])
        with quiet():
            explainer = QuadratureTreeSHAP(
                tree,
                index="SII",
                max_order=2,
                class_index=1 if task == "classification" else None,
            )
            values = explainer.explain(data["X_test"][0])
        cases[name] = (
            {k: float(values[k]) for k in values.interaction_lookup},
            float(values.baseline_value),
        )
    CACHE.write_bytes(pickle.dumps(cases))
    return cases


def main() -> None:
    warnings.simplefilter("ignore")
    label = sys.argv[1] if len(sys.argv) > 1 else "current"
    cases = pickle.loads(CACHE.read_bytes()) if CACHE.exists() else freeze_inputs()  # noqa: S301

    out = {}
    for name, (interactions, baseline) in cases.items():
        kwargs = {"index": "SII", "order": 2, "min_order": 1, "baseline_value": baseline}
        elapsed = median_ms(
            lambda i=interactions, k=kwargs: aggregate_base_attributions(interactions=i, **k)
        )
        result, _, _ = aggregate_base_attributions(interactions=interactions, **kwargs)
        ids = {f for key in interactions for f in key}
        out[name] = {
            "ms": elapsed,
            "n_base": len(interactions),
            "n_out": len(result),
            "max_feature_id": max(ids),
            "n_distinct_features": len(ids),
            # a scale-invariant fingerprint of the output, to confirm nothing changed
            "checksum": float(sum(abs(v) for v in result.values())),
        }
        print(
            f"  {name:22s} {len(interactions):6d} base -> {len(result):6d} k-SII "
            f"{elapsed:9.3f} ms   (max feature id {max(ids)}, {len(ids)} distinct)"
        )
    path = RESULTS / f"ksii_{label}.json"
    json.dump(out, path.open("w"), indent=2)
    print(f"saved results/{path.name}")


if __name__ == "__main__":
    main()
