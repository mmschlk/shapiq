"""Interactive jailbreak attribution on a single user-typed prompt.

Loads Meta's Llama Prompt Guard 2 (86M) once, then repeatedly reads a prompt
from stdin and reports:

1. classification result   -- predicted label + probability for every class
2. Shapley attribution      -- sentence-level SV (first order) and SII (second order)
3. plots                    -- force plot (SV) and network plot (SII)

Each analysed prompt is also written to outputs/ as a timestamped JSON.

Run:
    python interactive_jailbreak.py

Type a prompt and press Enter. Type 'q' (or just Enter on an empty line) to quit.

Note on attribution: Prompt Guard 2 targets the "malicious" class. Attribution
needs at least 2 sentences to be meaningful (one sentence => nothing to attribute
between). Multi-sentence prompts give the most informative output.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from shapiq.game_theory.exact import ExactComputer
from shapiq.imputer.text_imputer import TextImputer

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase


MODEL_NAME = "meta-llama/Llama-Prompt-Guard-2-86M"
# This model uses non-semantic labels LABEL_0 (benign) / LABEL_1 (malicious).
TARGET_LABEL_KEYWORDS = ("malicious", "label_1")
PLAYER_LEVEL = "sentence"
PERTURBATION_TYPE = "mask"

# Sessions are always saved next to THIS script (…/jailbreak_demo/outputs/),
# regardless of the current working directory the demo is launched from.
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


class JailbreakTextImputer(TextImputer):
    """TextImputer that targets the detector's 'attack' class by label keyword."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        prompt: str,
        *,
        player_level: str = "sentence",
        perturbation_type: str = "mask",
        output_type: str = "probability",
        target_label_keywords: tuple[str, ...] = ("malicious",),
    ) -> None:
        self.label_mapping = model.config.id2label
        target_class_idx = self._find_target_class_idx(target_label_keywords)

        super().__init__(
            model=model,
            tokenizer=tokenizer,
            text=prompt,
            player_level=player_level,
            perturbation_type=perturbation_type,
            model_type="encoder_classifier",
            class_idx=target_class_idx,
            output_type=output_type,
        )
        self.target_class_idx = target_class_idx

    def _find_target_class_idx(
        self,
        target_label_keywords: tuple[str, ...],
    ) -> int:
        for class_idx, label in self.label_mapping.items():
            if any(kw in str(label).lower() for kw in target_label_keywords):
                return int(class_idx)
        msg = (
            f"Could not find a target label matching {target_label_keywords} "
            f"in {self.label_mapping}. Edit TARGET_LABEL_KEYWORDS to match one "
            f"of the labels above."
        )
        raise ValueError(msg)


def classify(
    prompt: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
) -> dict:
    """Run the raw classifier and return label + per-class probabilities."""
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
    )
    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.softmax(logits, dim=-1)[0]
    id2label = model.config.id2label

    per_class = {
        id2label[i]: float(probs[i]) for i in range(len(probs))
    }
    pred_idx = int(torch.argmax(probs))
    return {
        "predicted_label": id2label[pred_idx],
        "predicted_index": pred_idx,
        "probabilities": per_class,
    }


def print_classification(clf: dict) -> None:
    print("\n" + "-" * 70)
    print("CLASSIFICATION")
    print("-" * 70)
    print(f"predicted label : {clf['predicted_label']}")
    print("probabilities   :")
    for label, p in clf["probabilities"].items():
        bar = "#" * int(round(p * 30))
        print(f"  {label:>12s} : {p:6.4f}  {bar}")


def target_prob(clf: dict) -> float:
    """Probability of the target (malicious) class from a classify() result."""
    for label, p in clf["probabilities"].items():
        if any(kw in label.lower() for kw in TARGET_LABEL_KEYWORDS):
            return p
    return clf["probabilities"][clf["predicted_label"]]


def attribute(
    prompt: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
) -> dict:
    """Compute sentence-level SV and SII for the target (malicious) class.

    Always returns players + count. sv/sii/full_score are None when there is
    only one sentence (nothing to attribute between).
    """
    imputer = JailbreakTextImputer(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        player_level=PLAYER_LEVEL,
        perturbation_type=PERTURBATION_TYPE,
        output_type="probability",
        target_label_keywords=TARGET_LABEL_KEYWORDS,
    )

    players = imputer.player_strategy.get_players()
    n = len(players)

    print("\n" + "-" * 70)
    print("SENTENCES (players)")
    print("-" * 70)
    for idx, player in enumerate(players):
        print(f"  S{idx}: {player!r}")

    if n < 2:
        print(
            "\n[note] only one sentence so far. Add another sentence to see "
            "attribution and synergy."
        )
        return {"players": players, "n": n, "sv": None, "sii": None, "full_score": None}

    full_score = imputer.full_prediction()
    print(f"\nfull malicious probability (target class): {full_score:.6f}")

    exact = ExactComputer(n_players=n, game=imputer)
    sv = exact(index="SV", order=1)
    sii = exact(index="SII", order=2)

    print("\nfirst-order Shapley values (per sentence):")
    for idx, player in enumerate(players):
        val = float(sv[(idx,)])
        arrow = "raises" if val > 0 else "lowers"
        print(f"  S{idx}: {val:+.6f}  ({arrow} malicious prob)   {player!r}")

    print("\nsecond-order interactions (per sentence pair):")
    for i in range(n):
        for j in range(i + 1, n):
            val = float(sii[(i, j)])
            print(f"  S{i} x S{j}: {val:+.6f}")

    return {
        "players": players,
        "n": n,
        "sv": sv,
        "sii": sii,
        "full_score": full_score,
    }


def print_incremental_effect(
    attr: dict,
    n_before: int,
    prob_before: float | None,
    prob_after: float,
) -> None:
    """Highlight the effect of newly added sentences on top of existing ones."""
    n = attr["n"]
    if n_before <= 0 or n_before >= n:
        return

    print("\n" + "=" * 70)
    print(f"INCREMENTAL EFFECT — added S{n_before}..S{n - 1} on top of S0..S{n_before - 1}")
    print("=" * 70)

    if prob_before is not None:
        delta = prob_after - prob_before
        direction = "more malicious" if delta > 0 else "less malicious"
        print(
            f"malicious prob: {prob_before:.6f} -> {prob_after:.6f}  "
            f"(delta {delta:+.6f}, {direction})"
        )

    sv = attr["sv"]
    sii = attr["sii"]
    players = attr["players"]

    if sv is not None:
        print("\nstandalone contribution of each added sentence (SV):")
        for idx in range(n_before, n):
            val = float(sv[(idx,)])
            arrow = "raises" if val > 0 else "lowers"
            print(f"  S{idx}: {val:+.6f}  ({arrow})   {players[idx]!r}")

    if sii is not None:
        print("\nsynergy between existing and added sentences (cross SII):")
        print("  positive = added sentence amplifies an existing one")
        print("  negative = redundant / overlapping signal")
        for old in range(n_before):
            for new in range(n_before, n):
                val = float(sii[(old, new)])
                print(f"  S{old} x S{new}: {val:+.6f}")


def build_record(
    prompt: str,
    clf: dict,
    attr: dict | None,
) -> dict:
    """Build a JSON-serializable record for one analysis step."""
    record: dict = {
        "prompt": prompt,
        "classification": clf,
    }

    if attr is not None and attr.get("sv") is not None:
        players = attr["players"]
        record["n_players"] = len(players)
        record["players"] = players
        record["full_score"] = float(attr["full_score"])
        record["sv_by_player"] = [
            {"index": i, "player": players[i], "shapley_value": float(attr["sv"][(i,)])}
            for i in range(len(players))
        ]
        record["sii_by_pair"] = [
            {
                "i": i,
                "j": j,
                "player_i": players[i],
                "player_j": players[j],
                "interaction_value": float(attr["sii"][(i, j)]),
            }
            for i in range(len(players))
            for j in range(i + 1, len(players))
        ]

    return record


class Session:
    """One prompt lifecycle: initial prompt plus every incremental addition.

    Creates outputs/session_<timestamp>/ and writes, per analysis step:
      - step_NN.json        (prompt + classification + SV/SII)
      - step_NN_force.png   (first-order force plot)
      - step_NN_network.png (second-order network plot)
    On close(), writes session.json summarizing all steps and the final state.
    """

    def __init__(self, out_dir: str | Path = OUTPUT_DIR) -> None:
        self.started = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.dir = Path(out_dir) / f"session_{self.started}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.steps: list[dict] = []
        print(f"[session] {self.dir.resolve()}")

    def add_step(
        self,
        prompt: str,
        clf: dict,
        attr: dict | None,
    ) -> None:
        """Save one step's JSON + PNGs and remember it for the summary."""
        idx = len(self.steps)
        record = build_record(prompt, clf, attr)
        record["step"] = idx
        record["timestamp_utc"] = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        step_json = self.dir / f"step_{idx:02d}.json"
        with step_json.open("w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

        saved = {"json": step_json.name}
        if attr is not None and attr.get("sv") is not None:
            saved.update(self._save_plots(attr, idx))

        self.steps.append({"record": record, "files": saved})
        print(f"[saved step {idx}] {step_json.name}"
              + (f" + {saved.get('force')} + {saved.get('network')}"
                 if "force" in saved else ""))

    def _save_plots(self, attr: dict, idx: int) -> dict:
        """Render force + network plots to PNG (show=False) and save them."""
        players = attr["players"]
        feature_names = [f"S{i}" for i in range(len(players))]

        force_path = self.dir / f"step_{idx:02d}_force.png"
        network_path = self.dir / f"step_{idx:02d}_network.png"

        fig_force = attr["sv"].plot_force(feature_names=feature_names, show=False)
        if fig_force is not None:
            fig_force.savefig(force_path, dpi=150, bbox_inches="tight")
            plt.close(fig_force)

        net = attr["sii"].plot_network(feature_names=players, show=False)
        fig_net = net[0] if isinstance(net, tuple) else net
        if fig_net is not None:
            fig_net.savefig(network_path, dpi=150, bbox_inches="tight")
            plt.close(fig_net)

        return {"force": force_path.name, "network": network_path.name}

    def close(self) -> None:
        """Write the session summary (all steps + final state)."""
        if not self.steps:
            # nothing analysed; drop the empty folder
            try:
                self.dir.rmdir()
            except OSError:
                pass
            return

        summary = {
            "metadata": {
                "model_name": MODEL_NAME,
                "player_level": PLAYER_LEVEL,
                "perturbation_type": PERTURBATION_TYPE,
                "session_started_utc": self.started,
                "n_steps": len(self.steps),
            },
            "steps": [s["record"] for s in self.steps],
            "files": [s["files"] for s in self.steps],
            "final_state": self.steps[-1]["record"],
        }
        summary_path = self.dir / "session.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"[session saved] {summary_path.resolve()}  ({len(self.steps)} steps)")


def show_plots_on_screen(attr: dict) -> None:
    """Display force + network plots interactively (saving is done by Session)."""
    players = attr["players"]
    feature_names = [f"S{i}" for i in range(len(players))]

    print("\nfeature mapping:")
    for short, player in zip(feature_names, players, strict=True):
        print(f"  {short}: {player}")

    attr["sv"].plot_force(feature_names=feature_names, show=True)
    attr["sii"].plot_network(feature_names=players, show=True)
    plt.show()


def analyze_and_report(
    prompt: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    session: Session,
    *,
    n_before: int,
    prob_before: float | None,
) -> tuple[dict, float]:
    """Run classification + attribution, print, save the step, show plots.

    Returns (attr, target_probability) so the caller can track state across
    incremental additions.
    """
    clf = classify(prompt, model, tokenizer)
    print_classification(clf)
    prob_after = target_prob(clf)

    attr = attribute(prompt, model, tokenizer)

    print_incremental_effect(
        attr,
        n_before=n_before,
        prob_before=prob_before,
        prob_after=prob_after,
    )

    # persist this step (JSON + PNGs) before showing, so nothing is lost
    session.add_step(prompt, clf, attr)

    if attr.get("sv") is not None:
        show_plots_on_screen(attr)

    return attr, prob_after


def main() -> None:
    print(f"loading model: {MODEL_NAME}")
    print("(first run downloads the model; needs a gated-model HF login)")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()

    print("\nmodel labels:", model.config.id2label)
    print(
        "\nready.\n"
        "  - type a prompt and press Enter to start\n"
        "  - after that, type more sentence(s) to ADD them on top and see the synergy\n"
        "  - 'reset' clears and starts a fresh prompt\n"
        "  - empty line or 'q' quits\n"
    )

    current_prompt = ""
    prob_before: float | None = None
    n_before = 0
    session: Session | None = None

    while True:
        try:
            if not current_prompt:
                raw = input("prompt> ").strip()
            else:
                raw = input("add sentence(s) (or 'reset' / 'q')> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye.")
            break

        if not raw or raw.lower() in {"q", "quit", "exit"}:
            if session is not None:
                session.close()
            print("bye.")
            break

        if raw.lower() == "reset":
            # finalize the previous session before starting fresh
            if session is not None:
                session.close()
                session = None
            current_prompt = ""
            prob_before = None
            n_before = 0
            print("[reset] cleared. type a new prompt.\n")
            continue

        try:
            if not current_prompt:
                current_prompt = raw
                n_before = 0
                session = Session()  # new prompt lifecycle -> new session folder
            else:
                # remember how many sentences existed BEFORE this addition,
                # so cross-interactions (old x new) can be highlighted.
                n_before = count_sentences(current_prompt, model, tokenizer)
                # append the new text to the running prompt
                current_prompt = f"{current_prompt} {raw}"

            attr, prob_before = analyze_and_report(
                current_prompt,
                model,
                tokenizer,
                session,
                n_before=n_before,
                prob_before=prob_before,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[error] failed to analyse: {exc}")

        print()


def count_sentences(
    prompt: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
) -> int:
    """Number of sentence-players the current prompt splits into."""
    imputer = JailbreakTextImputer(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        player_level=PLAYER_LEVEL,
        perturbation_type=PERTURBATION_TYPE,
        output_type="probability",
        target_label_keywords=TARGET_LABEL_KEYWORDS,
    )
    return len(imputer.player_strategy.get_players())


if __name__ == "__main__":
    main()
