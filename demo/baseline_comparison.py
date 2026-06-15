"""Baseline comparison scaffold for the text/LLM shapiq demos.

This script creates a lightweight comparison report for shapiq vs. common text
explanation baselines. It intentionally does not fail when optional libraries
are missing, because SHAP, Captum, and Inseq are not part of the core shapiq
environment.

这个脚本会生成一个轻量 baseline comparison report，用来比较 shapiq 和常见
文本解释 baseline。它不会因为 SHAP / Captum / Inseq 没安装就失败，因为这些
库不是 shapiq 核心环境的一部分。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


REPORT_PATH = Path(__file__).resolve().parent / "figures" / "baseline_comparison_summary.md"


BASELINES = {
    "shapiq": {
        "import_name": "shapiq",
        "method": "ExactComputer + k-SII",
        "supports_interactions": "yes, any-order interactions",
        "notes": "Used in the context and jailbreak demos.",
    },
    "shap": {
        "import_name": "shap",
        "method": "shap.Explainer + shap.maskers.Text",
        "supports_interactions": "no, mainly first-order text attributions",
        "notes": "Useful de-facto baseline for token/text attributions.",
    },
    "captum": {
        "import_name": "captum",
        "method": "Captum ShapleyValueSampling",
        "supports_interactions": "no, mainly feature attribution",
        "notes": "Can be applied to token or embedding features in PyTorch models.",
    },
    "inseq": {
        "import_name": "inseq",
        "method": "Inseq sequence attribution",
        "supports_interactions": "no, sequence attribution rather than Shapley interactions",
        "notes": "Most relevant for generation-focused explanations.",
    },
}


def is_installed(import_name: str) -> bool:
    """Return whether a Python package can be imported.

    English:
        We use importlib metadata instead of importing packages, so this check is
        cheap and has no model-loading side effects.

    中文:
        这里用 importlib 检查包是否存在，不真正 import 包，因此速度快，也不会
        触发模型加载等副作用。
    """
    return importlib.util.find_spec(import_name) is not None


def build_report() -> str:
    """Build the baseline comparison Markdown report."""
    lines: list[str] = [
        "# Baseline Comparison Summary",
        "",
        "This section compares the current shapiq text demos with common text-explanation baselines.",
        "",
        "## Shared Setup",
        "",
        "- **Model:** Gemma causal LM (`google/gemma-4-E2B-it`)",
        "- **Context demo value function:** `v(S) = log P_Gemma(\"Paris\" | question + selected context chunks)`",
        "- **Jailbreak demo value function:** `v(S) = log P_Gemma(\"Sure\" | selected jailbreak prompt segments)`",
        "- **shapiq index:** `k-SII`, order 2",
        "",
        "## Environment Availability",
        "",
        "| Library | Installed? | Method | Interaction Support | Notes |",
        "| --- | --- | --- | --- | --- |",
    ]

    for library, metadata in BASELINES.items():
        installed = is_installed(metadata["import_name"])
        lines.append(
            "| "
            f"{library} | "
            f"{'yes' if installed else 'no'} | "
            f"{metadata['method']} | "
            f"{metadata['supports_interactions']} | "
            f"{metadata['notes']} |"
        )

    lines.extend(
        [
            "",
            "## Current Findings",
            "",
            "### shapiq",
            "",
            "- The context demo identifies a positive `k-SII` interaction between two supporting chunks.",
            "- The context demo identifies a negative `k-SII` interaction between a supporting chunk and a misleading chunk.",
            "- The jailbreak demo identifies the strongest positive interaction between `ignore safety` and `unsafe request`.",
            "- These are pairwise interaction effects, not only first-order token or segment attributions.",
            "",
            "### SHAP / Captum / Inseq",
            "",
            "- These packages are not installed in the current environment, so a live head-to-head runtime/agreement experiment has not been executed yet.",
            "- Once installed, the comparison should use the same model, inputs, and target scores where possible.",
            "- The key expected difference is that shapiq directly reports interaction indices such as `k-SII`, while the baselines primarily provide first-order attributions or generation attributions.",
            "",
            "## Next Step To Run Live Baselines",
            "",
            "Install optional baseline packages in a separate environment if needed:",
            "",
            "```powershell",
            "python -m pip install shap captum inseq",
            "```",
            "",
            "Then add runtime and attribution-agreement measurements for the same context and jailbreak examples.",
            "",
        ]
    )

    return "\n".join(lines)


def main() -> None:
    """Write the comparison report."""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report = build_report()
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"Saved baseline comparison report to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
