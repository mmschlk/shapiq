"""Check the lab's in-browser GraphSHAP-IQ against shapiq, on a few molecules.

The lab (``../../graphshapiq-lab/assets/demo.js``) computes the Shapley values and the 2-SII of the
molecules a reader builds, in a Web Worker. This script runs that worker's code with Node on a few
molecules, and compares its numbers with those of shapiq's ``GraphSHAPIQ`` for the same molecule,
GNN and baseline. The lab builds its molecules itself, so their atoms come in another order than
RDKit's: the values are compared sorted.

Usage::

    python check_lab.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from molecules import GNN, LogOdds, to_graph
from rdkit import Chem

from shapiq.graph import (
    GraphBaselineImputer,
    GraphSHAPIQ,
    convert_graph,
    get_graph_predict_function,
)

HERE = Path(__file__).resolve().parent
LAB = HERE.parent.parent / "graphshapiq-lab" / "assets"
TOLERANCE = 1e-3  # both sides run the GNN in float32
NONE = ["none"] * 5
CASES = [  # a name, the molecule as SMILES and as the lab builds it, and the question
    ("catechol", "Oc1ccccc1O", ["OH", "OH", *NONE[1:]], "PAINS"),
    ("resorcinol", "Oc1cccc(O)c1", ["OH", "none", "OH", *NONE[2:]], "PAINS"),
    ("nitrobenzene", "O=[N+]([O-])c1ccccc1", ["NO2", *NONE], "mutagenic"),
    (
        "1,3-dinitrobenzene",
        "O=[N+]([O-])c1cccc([N+](=O)[O-])c1",
        ["NO2", "none", "NO2", *NONE[2:]],
        "mutagenic",
    ),
]
# runs the lab's worker and its assemble() on the cases (argv[2]), and prints their results
NODE = r"""
const fs = require("fs");
const src = fs.readFileSync(process.argv[2], "utf8");
const worker = src.match(/const WORKER = `([\s\S]*?)`;/)[1];
const GROUPS = eval("(" + src.match(/const GROUPS = (\{[\s\S]*?\n  \});/)[1] + ")");
const assemble = new Function("GROUPS", src.match(/(function assemble\(slots\) \{[\s\S]*?\n  \})/)[1] + "; return assemble;")(GROUPS);
const data = JSON.parse(fs.readFileSync(process.argv[3], "utf8"));
global.atob = (s) => Buffer.from(s, "base64").toString("binary");
const out = data.cases.map(([slots, task]) => {
  let result;
  const run = new Function("postMessage", "atob", worker + "; return onmessage;")((m) => { if (m.done) result = m; }, global.atob);
  run({ data: { id: 1, symbols: data.symbols, m: assemble(slots), t: data.tasks.find((t) => t.id === task) } });
  return result;
});
console.log(JSON.stringify(out));
"""


def lab_data() -> dict:
    """Return the lab's data.js as a dictionary."""
    text = (LAB / "data.js").read_text(encoding="utf-8")
    return json.loads(text[text.index("=") + 1 :].strip().rstrip(";"))


def in_browser(data: dict) -> list[dict]:
    """Return the lab worker's results for the cases, run with Node."""
    with tempfile.TemporaryDirectory() as tmp:
        script, payload = Path(tmp) / "run.js", Path(tmp) / "data.json"
        script.write_text(NODE)
        cases = [[slots, task] for _, _, slots, task in CASES]
        payload.write_text(
            json.dumps({"cases": cases, "symbols": data["symbols"], "tasks": data["tasks"]})
        )
        command = ["node", str(script), str(LAB / "demo.js"), str(payload)]
        run = subprocess.run(command, capture_output=True, text=True, check=True)  # noqa: S603
    return json.loads(run.stdout)


def in_python(smiles: str, task: dict) -> dict:
    """Return shapiq's log-odds, Shapley values and 2-SII of a molecule, with the lab's baseline."""
    gnn = GNN()
    gnn.load_state_dict(torch.load(HERE / "results" / "models" / f"{task['id']}.pt"))
    model = LogOdds(gnn).eval()
    graph = convert_graph(to_graph(Chem.MolFromSmiles(smiles)), n_hops=model.num_layers)
    predict = get_graph_predict_function(model, graph, batch_size=512)
    game = GraphBaselineImputer(predict, np.array([task["baseline"]]), x=graph, normalize=False)
    graphshapiq = GraphSHAPIQ(game, graph)
    shapley = graphshapiq("k-SII", 1)
    k_sii = graphshapiq("k-SII", 2)
    alone = [v for s, v in k_sii.dict_values.items() if len(s) == 1]
    return {
        "logOdds": float(game(np.ones((1, game.n_players), dtype=bool))[0]),
        "empty": float(graphshapiq.moebius[()]),
        "shapley": [v for s, v in shapley.dict_values.items() if len(s) == 1],
        "own": alone,
        "pairs": [v for s, v in k_sii.dict_values.items() if len(s) == 2],
    }


def main() -> None:
    """Compare the lab with shapiq on every case, and fail if they differ by more than 1e-3."""
    if shutil.which("node") is None:
        sys.exit("check_lab.py needs Node (https://nodejs.org) to run the lab's worker.")
    data = lab_data()
    tasks = {task["id"]: task for task in data["tasks"]}
    worst = 0.0
    for (name, smiles, _, task), lab in zip(CASES, in_browser(data), strict=True):
        ours = in_python(smiles, tasks[task])
        # the lab sends its strongest pairs only: compare them with as many of shapiq's strongest
        lab_pairs = np.sort([value for _, value in lab["pairs"]])
        strongest = sorted(ours["pairs"], key=abs, reverse=True)[: len(lab_pairs)]
        errors = {
            "log-odds": abs(lab["logOdds"] - ours["logOdds"]),
            "empty": abs(lab["empty"] - ours["empty"]),
            "Shapley values": np.max(np.abs(np.sort(lab["shapley"]) - np.sort(ours["shapley"]))),
            "atoms alone (2-SII)": np.max(np.abs(np.sort(lab["own"]) - np.sort(ours["own"]))),
            "pairs (2-SII)": np.max(np.abs(lab_pairs - np.sort(strongest))),
        }
        worst = max(worst, *errors.values())
        print(
            f"{name} ({task}): " + ", ".join(f"{key} {error:.1e}" for key, error in errors.items())
        )
    if worst > TOLERANCE:
        sys.exit(f"the lab differs from shapiq by {worst:.1e}, more than {TOLERANCE:.0e}")
    print(f"the lab agrees with shapiq to {worst:.1e}")


if __name__ == "__main__":
    main()
