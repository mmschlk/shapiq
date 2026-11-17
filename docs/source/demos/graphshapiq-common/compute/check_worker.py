"""Check the explorer's in-browser GraphSHAP-IQ against shapiq's explanations in its data.

The explorer (``../../graphshapiq-explorer/assets/demo.js``) explains the molecule a reader builds
by itself, in a Web Worker. This script runs that worker's code with Node on the explorer's own
molecules, for every question, and compares its numbers with the explanations in the explorer's
``data.js``, which explain.py computed with shapiq's ``GraphSHAPIQ``:

- the number of maskings, which must be GraphSHAP-IQ's (the worker evaluates the GNN only on the
  maskings inside a receptive field);
- the GNN's log-odds for the molecule and for the molecule with every atom masked;
- every interaction the data holds: the k-SII of orders 1 to 4 and the Möbius interactions.

Only the molecules within the worker's limit of atoms are compared. The script needs Node, but
neither a GNN nor ``results/``. Both sides run the GNN in float32 and the data is rounded to four
decimals, so the values agree to a few ``1e-3`` of the largest value of a list.

Usage::

    python check_worker.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPLORER = HERE.parent.parent / "graphshapiq-explorer" / "assets"
TOLERANCE = 5e-3  # of a list's largest value (or of 1, if that is larger)
# runs the explorer's worker on every molecule within its limit, for every task of the data
# (argv[3]), and prints what it posts when it is done; it is asked for all interactions
NODE = r"""
const fs = require("fs");
const src = fs.readFileSync(process.argv[2], "utf8");
const worker = src.match(/const WORKER = `([\s\S]*?)`;/)[1];
const limit = Number(src.match(/const MAX_ATOMS = (\d+);/)[1]);
const data = JSON.parse(fs.readFileSync(process.argv[3], "utf8"));
const atob = (s) => Buffer.from(s, "base64").toString("binary");
const out = {};
for (const m of data.molecules) {
  if (m.atoms.length > limit) continue;
  for (const t of data.tasks) {
    let result;
    const run = new Function("postMessage", "atob", worker + "; return onmessage;")((x) => { if (x.done) result = x; }, atob);
    run({ data: { id: 1, symbols: data.symbols, m: { atoms: m.atoms, bonds: m.bonds }, t, overall: Infinity, perAtom: 0 } });
    out[`${m.id}/${t.id}`] = result;
  }
}
console.log(JSON.stringify(out));
"""


def explorer_data() -> dict:
    """Return the explorer's data.js as a dictionary."""
    text = (EXPLORER / "data.js").read_text(encoding="utf-8")
    return json.loads(text[text.index(" = ") + 3 :].rstrip().removesuffix(";"))


def in_browser(data: dict) -> dict[str, dict]:
    """Return the worker's results for the molecules within its limit, run with Node."""
    with tempfile.TemporaryDirectory() as tmp:
        script, payload = Path(tmp) / "run.js", Path(tmp) / "data.json"
        script.write_text(NODE)
        keys = ("symbols", "tasks", "molecules")
        payload.write_text(json.dumps({key: data[key] for key in keys}))
        command = ["node", str(script), str(EXPLORER / "demo.js"), str(payload)]
        run = subprocess.run(command, capture_output=True, text=True, check=True)  # noqa: S603
    return json.loads(run.stdout)


def difference(ours: list[list[float]], theirs: list[list[float]]) -> float:
    """Return the largest difference of two lists of interactions, relative to the largest value.

    Every interaction of ``theirs`` (coalitions as bitmasks) must be among ``ours``.
    """
    values = dict(map(tuple, ours))
    scale = max(1.0, *(abs(value) for _, value in theirs))
    return (
        max(abs(values.get(coalition, float("inf")) - value) for coalition, value in theirs) / scale
    )


def main() -> None:
    """Compare the worker with shapiq on every molecule and question, and fail if they differ."""
    if shutil.which("node") is None:
        sys.exit("check_worker.py needs Node (https://nodejs.org) to run the explorer's worker.")
    data = explorer_data()
    worst = 0.0
    failed = False
    for key, browser in in_browser(data).items():
        shapiq = data["explanations"][key]
        errors = {
            "log-odds": abs(browser["logOdds"] - shapiq["logOdds"]),
            "masked": abs(browser["empty"] - shapiq["empty"]),
            "Möbius": difference(browser["moebius"], shapiq["moebius"]),
        } | {
            f"{order}-SII": difference(browser["kSII"][order], values)
            for order, values in shapiq["kSII"].items()
        }
        same_budget = browser["budget"] == shapiq["budget"]
        failed |= not same_budget
        worst = max(worst, *errors.values())
        budget = f"{browser['budget']:,} of {2 ** browser['nAtoms']:,} maskings"
        if not same_budget:
            budget += f" (shapiq: {shapiq['budget']:,})"
        print(f"{key}: {budget}, largest difference {max(errors.values()):.1e}")
    if failed:
        sys.exit("the worker does not evaluate the maskings GraphSHAP-IQ evaluates")
    if worst > TOLERANCE:
        sys.exit(f"the worker differs from shapiq by {worst:.1e}, more than {TOLERANCE:.0e}")
    print(f"the worker agrees with shapiq to {worst:.1e}, on GraphSHAP-IQ's maskings")


if __name__ == "__main__":
    main()
