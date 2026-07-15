"""Streamlit UI for the context attribution demo.
Streamlit 界面：用于运行 context attribution demo。

Run from the repository root with:
在项目根目录运行：
    streamlit run demo/context_app.py
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import streamlit as st


# Repository paths used by the Streamlit wrapper.
# Streamlit 外层界面需要用到的项目路径。
REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_SCRIPT = REPO_ROOT / "demo" / "context_attribution.py"
FIGURE_PATTERN = re.compile(r"^[A-Za-z]:\\.*\.png$|^/.*\.png$")


# Short UI explanations for the two supported demo modes.
# 两种 demo 模式在界面上展示的简短说明。
MODE_EXPLANATIONS = {
    "retrieval": (
        "Retrieval mode treats each evidence chunk as one player. "
        "It explains which chunks support, distract, or contradict the target answer."
    ),
    "few_shot": (
        "Few-shot mode treats each demonstration example as one player. "
        "It explains which examples help Gemma choose the target multiple-choice answer."
    ),
}


# Short explanations shown under each generated figure.
# 每张生成图片下方展示的简短解释。
FIGURE_EXPLANATIONS = {
    "single_chunk_effects": (
        "Single-player effects: positive bars increase Gemma's score for the target answer; "
        "negative bars reduce it. This is the quickest view of which individual player helps or hurts."
    ),
    "pairwise_interactions": (
        "shapiq matrix-style plot: this is the built-in shapiq visualization. "
        "The upper bars rank interaction values, and the lower dot matrix shows which players form each interaction."
    ),
    "pairwise_heatmap": (
        "Pairwise heatmap: the same order-2 k-SII interaction matrix in a more readable pairwise layout. "
        "Red means positive cooperation; blue means negative interference."
    ),
}


# One-sentence reading guide for the selected mode.
# 根据当前模式给出一句读图提示。
RUN_TIPS = {
    "retrieval": (
        "Read this as evidence attribution: which retrieved chunk changes Gemma's confidence in the final answer?"
    ),
    "few_shot": (
        "Read this as demonstration attribution: which few-shot examples guide Gemma toward the target option?"
    ),
}


def run_demo(
    demo_mode: str,
    case_source: str,
    device: str,
    generated_retrieval_topic: str,
    generated_few_shot_topic: str,
    mmlu_subject: str,
) -> tuple[int, str, list[Path]]:
    """Run context_attribution.py with environment-variable controls.
通过环境变量控制并运行 context_attribution.py。
"""
    # Build a clean environment for the subprocess.
    # 为子进程构造运行环境。
    env = os.environ.copy()
    old_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    if old_pythonpath:
        env["PYTHONPATH"] += os.pathsep + old_pythonpath
    # Avoid duplicate OpenMP runtime crashes on Windows and save plots without popup windows.
    # 避免 Windows 上 OpenMP 重复加载报错，并禁止图片窗口自动弹出。
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    env["HF_ENABLE_PARALLEL_LOADING"] = "false"
    env["SHAPIQ_SHOW_PLOTS"] = "0"
    env["SHAPIQ_DEMO_MODE"] = demo_mode
    env["SHAPIQ_CASE_SOURCE"] = case_source
    env["SHAPIQ_DEVICE"] = device
    env["SHAPIQ_GENERATED_RETRIEVAL_TOPIC"] = generated_retrieval_topic
    env["SHAPIQ_GENERATED_FEW_SHOT_TOPIC"] = generated_few_shot_topic
    env["SHAPIQ_MMLU_SUBJECT"] = mmlu_subject

    # Run the actual attribution script and capture its console output.
    # 运行真正的 attribution 脚本，并捕获终端输出。
    result = subprocess.run(
        [sys.executable, str(DEMO_SCRIPT)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    output = result.stdout
    if result.stderr:
        output += "\n\n[stderr]\n" + result.stderr

    # Parse saved figure paths from the script output.
    # 从脚本输出中提取已保存的图片路径。
    image_paths: list[Path] = []
    for line in output.splitlines():
        candidate = line.strip()
        if candidate.endswith(".png") and FIGURE_PATTERN.match(candidate):
            path = Path(candidate)
            if path.exists():
                image_paths.append(path)

    return result.returncode, output, image_paths


def output_value(output: str, prefix: str) -> str | None:
    """Extract the value after a console-output prefix.
提取某个终端输出前缀后面的值。
"""
    for line in output.splitlines():
        if line.startswith(prefix):
            return line.split(":", 1)[1].strip()
    return None


def output_line(output: str, prefix: str) -> str | None:
    """Extract a whole console-output line by prefix.
根据前缀提取一整行终端输出。
"""
    for line in output.splitlines():
        if line.startswith(prefix):
            return line.strip()
    return None


def output_block(output: str, heading: str) -> list[str]:
    """Extract non-empty lines after a console heading until the next blank section.
提取某个标题后的非空行，直到下一个空行段落为止。
"""
    lines = output.splitlines()
    for index, line in enumerate(lines):
        if line.strip() == heading:
            block: list[str] = []
            for next_line in lines[index + 1:]:
                stripped = next_line.strip()
                if not stripped:
                    if block:
                        break
                    continue
                block.append(stripped)
            return block
    return []


def parse_output_summary(output: str) -> dict[str, str]:
    """Collect the most important console lines for a compact UI summary.
收集最重要的终端输出，用于界面上的简洁摘要。
"""
    return {
        "mode": output_value(output, "Demo mode") or "-",
        "source": output_value(output, "Case source") or "-",
        "device": output_value(output, "Runtime device") or "-",
        "case": output_value(output, "Selected case") or "-",
        "target": (output_value(output, "Target answer") or "-").strip("'"),
        "question": "\n".join(output_block(output, "Question:")) or "-",
        "chunks": "\n".join(output_block(output, "Context chunks:")) or "-",
        "full_score": output_line(output, "Full context score") or "-",
        "top_positive": output_line(output, "Top positive") or "-",
        "top_negative": output_line(output, "Top negative") or "-",
        "best_interaction": output_line(output, "Strongest positive interaction") or "-",
        "worst_interaction": output_line(output, "Strongest negative interaction") or "-",
    }


def figure_kind(path: Path) -> str:
    """Classify a generated figure by filename.
根据文件名判断生成图片的类型。
"""
    name = path.name
    if "single_chunk_effects" in name:
        return "Single effects"
    if "pairwise_heatmap" in name:
        return "Pairwise heatmap"
    if "pairwise_interactions" in name:
        return "shapiq matrix plot"
    return "Other"


def figure_caption(path: Path) -> str:
    """Return a short interpretation for a saved figure path.
根据图片路径返回一句简短解释。
"""
    name = path.name
    for key, explanation in FIGURE_EXPLANATIONS.items():
        if key in name:
            return explanation
    return "Generated figure from the context attribution demo."


def render_output_summary(output: str) -> None:
    """Render the compact summary before the full console log.
在完整终端 log 之前先展示简洁摘要。
"""
    summary = parse_output_summary(output)

    st.subheader("Key results")
    cols = st.columns(5)
    cols[0].metric("Mode", summary["mode"])
    cols[1].metric("Case source", summary["source"])
    cols[2].metric("Device", summary["device"])
    cols[3].metric("Selected case", summary["case"])
    cols[4].metric("Target answer", summary["target"])

    st.markdown("#### Case overview")
    st.markdown("**Question**")
    st.code(summary["question"], language="text")

    st.markdown("**Context chunks / players**")
    st.code(summary["chunks"], language="text")

    st.markdown("#### Main findings")
    findings = [
        ("Full prompt score", summary["full_score"]),
        ("Most helpful player", summary["top_positive"]),
        ("Most harmful player", summary["top_negative"]),
        ("Strongest cooperation", summary["best_interaction"]),
        ("Strongest conflict", summary["worst_interaction"]),
    ]
    for title, text in findings:
        if text != "-":
            st.markdown(f"**{title}:** {text}")


def render_figures(image_paths: list[Path]) -> None:
    """Render generated figures in readable groups.
按图片类型分组展示生成结果。
"""
    st.subheader("Figures and interpretation")
    if not image_paths:
        st.warning("No saved figure paths were found in the output.")
        return

    grouped: dict[str, list[Path]] = {
        "Single effects": [],
        "shapiq matrix plot": [],
        "Pairwise heatmap": [],
        "Other": [],
    }
    for path in image_paths:
        grouped[figure_kind(path)].append(path)

    tab_labels = ["Single effects", "shapiq matrix plot", "Pairwise heatmap", "Other"]
    tabs = st.tabs(tab_labels)
    for tab, label in zip(tabs, tab_labels, strict=False):
        with tab:
            if not grouped[label]:
                st.info(f"No {label.lower()} figure was generated for this run.")
                continue
            for path in grouped[label]:
                st.markdown(f"**{path.name}**")
                st.image(str(path), use_container_width=True)
                st.caption(figure_caption(path))


# Page layout and sidebar controls.
# 页面布局和侧边栏控件。
st.set_page_config(page_title="Context Attribution Demo", layout="wide")
st.title("Context Attribution Demo")
st.caption("Gemma + TextImputer + shapiq")
st.markdown("**Value function**")
st.code(
    "v(S) = log P_Gemma(target answer | target question + selected players in S)",
    language="text",
)
st.caption(
    "For each coalition S, the selected context chunks or demonstrations are kept, "
    "and Gemma scores how likely the target answer is."
)

with st.sidebar:
    st.header("Run settings")
    demo_mode = st.radio("Demo mode", ["retrieval", "few_shot"], index=1)
    case_source_labels = {
        "manual": "Manual case",
        "gemma_generated": "Gemma-generated case",
        "mmlu_dataset": "MMLU dataset case",
    }
    case_source = st.radio(
        "Case source",
        ["manual", "gemma_generated", "mmlu_dataset"],
        format_func=case_source_labels.get,
        index=0,
    )
    device = st.radio("Device", ["auto", "cpu", "cuda"], index=0)

    st.divider()
    st.subheader("Gemma-generated case")
    generated_retrieval_topic = st.text_input(
        "Retrieval generation topic",
        "simple general-knowledge question with short evidence chunks",
    )
    generated_few_shot_topic = st.text_input(
        "Few-shot generation topic",
        "simple MMLU-style multiple-choice question with five short demonstrations",
    )

    st.divider()
    st.subheader("MMLU dataset")
    mmlu_subject = st.text_input("MMLU subject", "college_computer_science")

    run_clicked = st.button("Run demo", type="primary")

st.markdown(f"**Current mode:** `{demo_mode}`")
st.write(MODE_EXPLANATIONS[demo_mode])
st.info(RUN_TIPS[demo_mode])

if case_source == "gemma_generated":
    st.warning(
        "Gemma-generated case means Gemma first creates the demo case, then Gemma is used again as the scoring model."
    )

if case_source == "mmlu_dataset":
    st.info(
        "MMLU dataset case means the case comes from real MMLU questions; Gemma is only used as the scoring model."
    )
    if demo_mode != "few_shot":
        st.warning("MMLU dataset source only works with few_shot mode.")

if run_clicked:
    with st.spinner("Running Gemma/TextImputer/shapiq pipeline..."):
        returncode, output, image_paths = run_demo(
            demo_mode=demo_mode,
            case_source=case_source,
            device=device,
            generated_retrieval_topic=generated_retrieval_topic,
            generated_few_shot_topic=generated_few_shot_topic,
            mmlu_subject=mmlu_subject,
        )

    if returncode == 0:
        st.success("Run completed.")
    else:
        st.error(f"Run finished with exit code {returncode}.")

    if "falling back to manual case" in output:
        st.warning("Gemma case generation failed, so the demo used the manual fallback case.")

    render_output_summary(output)
    render_figures(image_paths)

    with st.expander("Full console log", expanded=False):
        st.code(output, language="text")
else:
    st.info("Choose settings in the sidebar, then click Run demo.")