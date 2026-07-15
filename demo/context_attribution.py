"""Context attribution demo for LLM explanations with Gemma, TextImputer, and shapiq.
用 Gemma、TextImputer 和 shapiq 做 LLM context attribution demo。

This file supports two demo modes.
本文件支持两种 demo 模式。

1. retrieval mode: each retrieved evidence chunk is one player.
1. retrieval 模式：每个检索到的 evidence chunk 是一个 player。

2. few_shot mode: each question-choice-answer demonstration is one player.
2. few_shot 模式：每个 question-choice-answer demonstration 是一个 player。

For each coalition S, selected players are kept and missing players are removed.
对每个 coalition S，被选中的 players 会保留，没被选中的 players 会被移除。

Value function:
价值函数：
    v(S) = log P_Gemma(target_answer | target question + selected players in S)

A higher score means Gemma is more likely to produce the target answer.
分数越高，表示 Gemma 越倾向生成目标答案。

The demo can use manual cases or Gemma-generated cases, and can run on CPU or GPU.
本 demo 可以使用手写案例或 Gemma 自动生成案例，也可以切换 CPU 或 GPU 运行。
"""

from __future__ import annotations

import itertools
import json
import os
import re
from pathlib import Path
from typing import TypedDict

# Disable threaded HuggingFace weight loading on Windows to avoid torch access-violation crashes.
# 在 Windows 上关闭 HuggingFace 并行加载，避免 torch 访问冲突崩溃。
os.environ.setdefault("HF_ENABLE_PARALLEL_LOADING", "false")

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from datasets import load_dataset
except ImportError:  # pragma: no cover - optional dependency for the MMLU demo path
    load_dataset = None

import shapiq
from shapiq.imputer.text_imputer import TextImputer


# Avoid tokenizer parallelism warnings during repeated demo runs.
# 避免多次运行 demo 时出现 tokenizer 并行 warning。
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Model setup.
# 模型设置。
# ---------------------------------------------------------------------------

MODEL_NAME = "google/gemma-4-E2B-it"

# Formal shapiq interaction index used for the main heatmap.
# 主 heatmap 使用的正式 shapiq interaction index。
INTERACTION_INDEX = "k-SII"
INTERACTION_ORDER = 2

# Prefer cached HuggingFace files to avoid network calls during demos.
# 优先使用 HuggingFace 本地缓存，避免 demo 时重新联网下载模型。
LOCAL_FILES_ONLY = True

# CPU mode is slower but more stable on this machine.
# CPU 模式慢一些，但在这台电脑上更稳定。
DEVICE = os.getenv("SHAPIQ_DEVICE", "auto")

# Keep the causal LM prompt exactly as build_text() creates it.
# 保持 causal LM 的 prompt 就是 build_text() 构造出来的内容。
PROMPT_TEMPLATE = "{text}"

# Show matplotlib windows when running from the command line.
# 命令行运行时是否弹出 matplotlib 图片窗口。
SHOW_PLOTS = os.getenv("SHAPIQ_SHOW_PLOTS", "1") == "1"


class ContextCase(TypedDict):
    """Reusable context attribution case.
    可复用的 context attribution 案例。

    question: the question asked after the retrieved chunks.
    question：放在 retrieved chunks 后面的具体问题。

    target_answer: the answer scored by the value function.
    target_answer：value function 要打分的目标答案。

    chunk_names: short labels used in plots.
    chunk_names：画图时使用的短标签。

    chunk_kinds: supporting, irrelevant, misleading, or contradictory labels.
    chunk_kinds：标注每个 chunk 是 supporting、irrelevant、misleading 还是 contradictory。

    context_chunks: retrieved chunks used as players.
    context_chunks：作为 players 的 retrieved context chunks。
    """

    question: str
    target_answer: str
    chunk_names: list[str]
    chunk_kinds: list[str]
    context_chunks: list[str]


# Each case mixes supporting, irrelevant, misleading, and contradictory context.
# 每个 case 都混合了支持、无关、误导和矛盾性的 context。
CASES: dict[str, ContextCase] = {
    "eiffel_tower": {
        "question": "Where is the Eiffel Tower located?",
        "target_answer": "Paris",
        "chunk_names": [
            "support: tower",
            "mislead: vegas",
            "irrelevant: berlin",
            "contradict: rome",
            "support: paris",
        ],
        "chunk_kinds": [
            "supporting",
            "misleading",
            "irrelevant",
            "contradictory",
            "supporting",
        ],
        "context_chunks": [
            "The Eiffel Tower is located in Paris, France.",
            "A replica of the Eiffel Tower can be found in Las Vegas.",
            "Berlin is the capital of Germany.",
            "The Eiffel Tower is located in Rome, Italy.",
            "Paris is the capital of France and is known for many landmarks.",
        ],
    },
    "penicillin": {
        "question": "Who discovered penicillin?",
        "target_answer": "Fleming",
        "chunk_names": [
            "support: fleming",
            "irrelevant: curie",
            "mislead: pasteur",
            "support: mold",
            "irrelevant: einstein",
        ],
        "chunk_kinds": [
            "supporting",
            "irrelevant",
            "misleading",
            "supporting",
            "irrelevant",
        ],
        "context_chunks": [
            "Alexander Fleming discovered penicillin in 1928.",
            "Marie Curie conducted pioneering research on radioactivity.",
            "Louis Pasteur developed the process of pasteurization.",
            "Penicillin was discovered after Fleming noticed mold inhibiting bacteria.",
            "Albert Einstein developed the theory of relativity.",
        ],
    },
    "moon_landing": {
        "question": "In what year did Apollo 11 land on the Moon?",
        "target_answer": "1969",
        "chunk_names": [
            "support: apollo",
            "irrelevant: mars",
            "mislead: 1972",
            "support: armstrong",
            "irrelevant: hubble",
        ],
        "chunk_kinds": [
            "supporting",
            "irrelevant",
            "misleading",
            "supporting",
            "irrelevant",
        ],
        "context_chunks": [
            "Apollo 11 landed on the Moon in 1969.",
            "Mars is the fourth planet from the Sun.",
            "Apollo 17 was the final Apollo Moon landing mission in 1972.",
            "Neil Armstrong became the first person to walk on the Moon during Apollo 11.",
            "The Hubble Space Telescope was launched in 1990.",
        ],
    },
}




# Few-shot cases treat each demonstration as one player.
# few-shot case 会把每个 demonstration 当作一个 player。
FEW_SHOT_CASES: dict[str, ContextCase] = {
    "mmlu_electrical": {
        "question": (
            "Question: A three-phase induction motor does not require which component for starting?\n"
            "A. Stator\n"
            "B. Rotor\n"
            "C. Commutator\n"
            "D. Capacitor"
        ),
        "target_answer": "C",
        "chunk_names": [
            "demo 1: latch",
            "demo 2: circuit",
            "demo 3: power",
            "demo 4: motor",
            "demo 5: distractor",
        ],
        "chunk_kinds": [
            "helpful",
            "helpful",
            "irrelevant",
            "helpful",
            "misleading",
        ],
        "context_chunks": [
            (
                "Demo 1:\n"
                "Question: In an SR latch built from NOR gates, which condition is not allowed?\n"
                "A. S=0, R=0\n"
                "B. S=0, R=1\n"
                "C. S=1, R=0\n"
                "D. S=1, R=1\n"
                "Answer: D"
            ),
            (
                "Demo 2:\n"
                "Question: The Thevenin equivalent circuit contains which elements?\n"
                "A. Voltage source and resistor\n"
                "B. Current source and resistor\n"
                "C. Two resistors\n"
                "D. Inductor and capacitor\n"
                "Answer: A"
            ),
            (
                "Demo 3:\n"
                "Question: Which of the following is not a unit of power?\n"
                "A. Watt\n"
                "B. Volt-Ampere\n"
                "C. Joule/Second\n"
                "D. Ohm\n"
                "Answer: D"
            ),
            (
                "Demo 4:\n"
                "Question: Which part of a DC motor reverses current direction in the armature?\n"
                "A. Brush\n"
                "B. Bearing\n"
                "C. Commutator\n"
                "D. Capacitor\n"
                "Answer: C"
            ),
            (
                "Demo 5:\n"
                "Question: Which component is commonly used to start a single-phase induction motor?\n"
                "A. Transformer\n"
                "B. Capacitor\n"
                "C. Diode\n"
                "D. Fuse\n"
                "Answer: B"
            ),
        ],
    },
}

# Choose the demo format: "retrieval" or "few_shot".
# 选择 demo 格式："retrieval" 是证据检索场景，"few_shot" 是 in-context learning 示例场景。
DEMO_MODE = os.getenv("SHAPIQ_DEMO_MODE", "few_shot")

# Change this value to run another test case.
# 改这里就可以切换测试案例。
CASE_NAME = os.getenv("SHAPIQ_CASE_NAME", "eiffel_tower")

# Change this value to run another few-shot test case.
# 改这里可以切换 few-shot 测试案例。
FEW_SHOT_CASE_NAME = os.getenv("SHAPIQ_FEW_SHOT_CASE_NAME", "mmlu_electrical")

# Choose where the case comes from: "manual" or "gemma_generated".
# 选择 case 来源："manual" 使用手写案例，"gemma_generated" 让 Gemma 自动生成一个新案例。
CASE_SOURCE = os.getenv("SHAPIQ_CASE_SOURCE", "manual")

# Topic used only when CASE_SOURCE is "gemma_generated".
# 只有 CASE_SOURCE 是 "gemma_generated" 时才会使用这个主题。
GENERATED_RETRIEVAL_TOPIC = os.getenv("SHAPIQ_GENERATED_RETRIEVAL_TOPIC", "simple general-knowledge question with short evidence chunks")
GENERATED_FEW_SHOT_TOPIC = os.getenv("SHAPIQ_GENERATED_FEW_SHOT_TOPIC", "simple MMLU-style multiple-choice question with five short demonstrations")

# MMLU dataset controls used when CASE_SOURCE is "mmlu_dataset".
# CASE_SOURCE ? "mmlu_dataset" ???? MMLU ??????
MMLU_DATASET_NAME = os.getenv("SHAPIQ_MMLU_DATASET_NAME", "cais/mmlu")
MMLU_SUBJECT = os.getenv("SHAPIQ_MMLU_SUBJECT", "college_computer_science")
MMLU_SPLIT = os.getenv("SHAPIQ_MMLU_SPLIT", "test")
MMLU_NUM_DEMOS = int(os.getenv("SHAPIQ_MMLU_NUM_DEMOS", "5"))
MMLU_TARGET_OFFSET = int(os.getenv("SHAPIQ_MMLU_TARGET_OFFSET", str(MMLU_NUM_DEMOS)))

# Limit the length of Gemma's generated JSON case.
# 限制 Gemma 生成 JSON case 的长度，避免 demo 等太久。
GENERATED_CASE_MAX_NEW_TOKENS = 512

# Figures are saved here for reports and slides.
# 图片会保存到这里，方便放进汇报和 slides。
FIGURE_DIR = Path(__file__).resolve().parent / "figures"


def get_case(case_name: str) -> ContextCase:
    """Return the selected retrieval-style context attribution case.
    返回当前选择的 retrieval-style context attribution 案例。
    """
    if case_name not in CASES:
        available = ", ".join(CASES)
        msg = f"Unknown case {case_name!r}. Available cases: {available}"
        raise ValueError(msg)
    return CASES[case_name]


def get_few_shot_case(case_name: str) -> ContextCase:
    """Return the selected few-shot attribution case.
    返回当前选择的 few-shot attribution 案例。
    """
    if case_name not in FEW_SHOT_CASES:
        available = ", ".join(FEW_SHOT_CASES)
        msg = f"Unknown few-shot case {case_name!r}. Available cases: {available}"
        raise ValueError(msg)
    return FEW_SHOT_CASES[case_name]



def _answer_to_letter(answer: object) -> str:
    """Convert an MMLU answer field into an A/B/C/D letter.
    ? MMLU ? answer ????? A/B/C/D ???
    """
    letters = "ABCD"
    if isinstance(answer, int):
        return letters[answer]
    if isinstance(answer, str):
        stripped = answer.strip()
        if stripped in letters:
            return stripped
        if stripped.isdigit():
            return letters[int(stripped)]
    msg = f"Unsupported MMLU answer format: {answer!r}"
    raise ValueError(msg)


def _mmlu_choices(row: dict[str, object]) -> list[str]:
    """Return the four multiple-choice options from an MMLU row.
    ??? MMLU ??????????
    """
    choices = row.get("choices")
    if isinstance(choices, list) and len(choices) >= 4:
        return [str(choice) for choice in choices[:4]]

    option_keys = ["A", "B", "C", "D"]
    if all(key in row for key in option_keys):
        return [str(row[key]) for key in option_keys]

    msg = f"Could not find four MMLU choices in row keys: {sorted(row)}"
    raise ValueError(msg)


def _format_mmlu_question(row: dict[str, object], include_answer: bool) -> str:
    """Format one MMLU item as prompt text.
    ??? MMLU ?????? prompt ???
    """
    question = str(row["question"]).strip()
    choices = _mmlu_choices(row)
    lines = [
        f"Question: {question}",
        f"A. {choices[0]}",
        f"B. {choices[1]}",
        f"C. {choices[2]}",
        f"D. {choices[3]}",
    ]
    if include_answer:
        lines.append(f"Answer: {_answer_to_letter(row['answer'])}")
    return "\n".join(lines)


def load_mmlu_few_shot_case(subject: str) -> ContextCase:
    """Download/cache MMLU and convert one subject into a few-shot attribution case.
    ???????? MMLU????? subject ?? few-shot attribution case?
    """
    if load_dataset is None:
        msg = (
            "MMLU dataset support requires the optional `datasets` package. "
            "Install it with: python -m pip install datasets"
        )
        raise ImportError(msg)

    dataset = load_dataset(MMLU_DATASET_NAME, subject, split=MMLU_SPLIT)
    needed = MMLU_NUM_DEMOS + 1
    if len(dataset) <= MMLU_TARGET_OFFSET:
        msg = (
            f"MMLU split {MMLU_SPLIT!r} for subject {subject!r} has only {len(dataset)} rows; "
            f"need target offset {MMLU_TARGET_OFFSET}."
        )
        raise ValueError(msg)
    if len(dataset) < needed:
        msg = f"MMLU split {MMLU_SPLIT!r} for subject {subject!r} needs at least {needed} rows."
        raise ValueError(msg)

    demo_rows = [dict(dataset[index]) for index in range(MMLU_NUM_DEMOS)]
    target_row = dict(dataset[MMLU_TARGET_OFFSET])

    context_chunks = [
        f"Demo {index + 1}:\n{_format_mmlu_question(row, include_answer=True)}"
        for index, row in enumerate(demo_rows)
    ]
    chunk_names = [f"demo {index + 1}" for index in range(MMLU_NUM_DEMOS)]
    chunk_kinds = ["mmlu_demo" for _ in range(MMLU_NUM_DEMOS)]

    return {
        "question": _format_mmlu_question(target_row, include_answer=False),
        "target_answer": _answer_to_letter(target_row["answer"]),
        "chunk_names": chunk_names,
        "chunk_kinds": chunk_kinds,
        "context_chunks": context_chunks,
    }


def build_text(case: ContextCase) -> str:
    """Build the full prompt from players, target question, and answer prefix.
    把 players、目标问题和 answer prefix 拼成完整 prompt。
    """
    if DEMO_MODE == "few_shot":
        demonstrations = "\n\n".join(case["context_chunks"])
        return f"{demonstrations}\n\nNow answer the target question:\n{case['question']}\nAnswer:"

    context = "\n".join(
        f"Context {idx + 1}: {chunk}"
        for idx, chunk in enumerate(case["context_chunks"])
    )
    return f"{context}\n\nQuestion: {case['question']}\nAnswer:"


class ContextChunkPlayerStrategy:
    """Player strategy that only treats retrieved context chunks as players.
    这个 player strategy 只把 retrieved context chunks 当成 players。

    TextImputer has built-in sentence-level strategies, but here the question should stay fixed.
    TextImputer 有内置的 sentence-level strategy，但这里 question 应该保持固定。

    Therefore, only retrieved chunks are explained as players.
    因此，这里只解释 retrieved chunks 对答案的影响。
    """

    def __init__(self, case: ContextCase) -> None:
        """Store the current context attribution case.
        保存当前 context attribution 案例。
        """
        self.case = case
        self.context_chunks = case["context_chunks"]

    def get_players(self) -> list[str]:
        """Return context chunks as players.
        返回作为 players 的 context chunks。
        """
        return self.context_chunks

    @property
    def n_players(self) -> int:
        """Return the number of context chunk players.
        返回 context chunk players 的数量。
        """
        return len(self.context_chunks)

    def coalition_to_text(
        self,
        coalition: np.ndarray,
        perturbation_strategy: object,  # noqa: ARG002
    ) -> str:
        """Build a prompt from selected context chunks.
        根据选中的 context chunks 构造 prompt。

        Missing chunks are removed from the prompt.
        没有被选中的 chunks 会从 prompt 里移除。

        The question and answer prefix are always kept because they are not players.
        question 和 answer prefix 始终保留，因为它们不是要解释的 players。
        """
        if len(coalition) != self.n_players:
            msg = (
                f"Coalition length {len(coalition)} does not match "
                f"n_players={self.n_players}"
            )
            raise ValueError(msg)

        selected_chunks = [
            chunk
            for keep, chunk in zip(coalition, self.context_chunks, strict=False)
            if keep
        ]

        if DEMO_MODE == "few_shot":
            selected_demos = "\n\n".join(selected_chunks)
            target = f"Now answer the target question:\n{self.case['question']}\nAnswer:"
            return f"{selected_demos}\n\n{target}" if selected_demos else target

        selected_context = "\n".join(
            f"Context {idx + 1}: {chunk}"
            for idx, chunk in enumerate(selected_chunks)
        )

        if selected_context:
            return f"{selected_context}\n\nQuestion: {self.case['question']}\nAnswer:"

        return f"Question: {self.case['question']}\nAnswer:"


def _slugify(text: str) -> str:
    """Create a short file-safe name.
    创建一个适合文件名使用的短名称。
    """
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text.lower()).strip("_")
    return slug[:40] or "generated_case"


def _extract_json_object(text: str) -> dict[str, object]:
    """Extract the first JSON object from a model response.
    从模型回复中提取第一个 JSON object。
    """
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        msg = f"Could not find a JSON object in Gemma output:\n{text}"
        raise ValueError(msg)
    return json.loads(text[start : end + 1])


def _short_label(label: str, max_words: int = 4) -> str:
    """Shorten generated labels so plots stay readable.
    缩短自动生成的标签，避免图上文字重叠。
    """
    words = label.replace(":", ": ").split()
    return " ".join(words[:max_words])


def _validate_generated_case(data: dict[str, object]) -> ContextCase:
    """Validate and normalize a Gemma-generated context case.
    校验并标准化 Gemma 自动生成的 context case。
    """
    required = ["question", "target_answer", "context_chunks", "chunk_names", "chunk_kinds"]
    missing = [key for key in required if key not in data]
    if missing:
        msg = f"Generated case is missing keys: {missing}"
        raise ValueError(msg)

    question = str(data["question"]).strip()
    target_answer = str(data["target_answer"]).strip()
    context_chunks = [str(chunk).strip() for chunk in data["context_chunks"]]  # type: ignore[index]
    chunk_names = [_short_label(str(name).strip()) for name in data["chunk_names"]]  # type: ignore[index]
    chunk_kinds = [str(kind).strip().lower() for kind in data["chunk_kinds"]]  # type: ignore[index]

    lengths = {len(context_chunks), len(chunk_names), len(chunk_kinds)}
    if len(lengths) != 1:
        msg = "Generated chunk_names, chunk_kinds, and context_chunks must have the same length."
        raise ValueError(msg)
    if not 4 <= len(context_chunks) <= 6:
        msg = "Generated case should contain 4 to 6 context chunks."
        raise ValueError(msg)
    if not question or not target_answer:
        msg = "Generated question and target_answer must not be empty."
        raise ValueError(msg)

    allowed_kinds = {"supporting", "irrelevant", "misleading", "contradictory"}
    bad_kinds = sorted(set(chunk_kinds) - allowed_kinds)
    if bad_kinds:
        msg = f"Generated chunk_kinds contain unsupported labels: {bad_kinds}"
        raise ValueError(msg)
    if "supporting" not in chunk_kinds or "contradictory" not in chunk_kinds:
        msg = "Generated case should include at least one supporting and one contradictory chunk."
        raise ValueError(msg)

    return {
        "question": question,
        "target_answer": target_answer,
        "chunk_names": chunk_names,
        "chunk_kinds": chunk_kinds,
        "context_chunks": context_chunks,
    }


def build_case_generation_prompt(topic: str) -> str:
    """Build the instruction prompt that asks Gemma to create a case.
    构造让 Gemma 自动生成 context attribution case 的指令 prompt。
    """
    return f"""
Create one context attribution demo case for an LLM explanation experiment.
The topic should be: {topic}.

The case should look like a small retrieval setting: one question, one short target answer,
and 4 or 5 short context chunks.

Hard constraints:
- The target_answer must be 1 to 3 words only.
- Each chunk_name must be 2 to 4 words only, for plotting.
- Each context chunk must be one short sentence, under 16 words.
- Prefer everyday facts over technical biology, medicine, or law.

Include these evidence types:
- supporting: directly supports the target answer
- irrelevant: related-looking but not useful
- misleading: distracts toward another answer without directly contradicting the target
- contradictory: directly states an answer that conflicts with the target answer

Return JSON only. Use exactly this schema and level of detail:
{{
  "question": "Where is the Eiffel Tower located?",
  "target_answer": "Paris",
  "chunk_names": ["support: tower", "mislead: vegas", "irrelevant: berlin", "contradict: rome", "support: paris"],
  "chunk_kinds": ["supporting", "misleading", "irrelevant", "contradictory", "supporting"],
  "context_chunks": [
    "The Eiffel Tower is located in Paris, France.",
    "A replica of the Eiffel Tower can be found in Las Vegas.",
    "Berlin is the capital of Germany.",
    "The Eiffel Tower is located in Rome, Italy.",
    "Paris is the capital of France and is known for many landmarks."
  ]
}}
""".strip()


def build_few_shot_generation_prompt(topic: str) -> str:
    """Build the instruction prompt that asks Gemma to create a few-shot case.
    构造让 Gemma 自动生成 few-shot attribution case 的指令 prompt。
    """
    return f"""
Create one few-shot in-context learning attribution case for an LLM explanation experiment.
The topic should be: {topic}.

Hard constraints:
- Use exactly 5 demonstrations.
- Each context_chunks item must be a complete demonstration, not a short label.
- Each demonstration must contain these exact lines: Demo N, Question, A., B., C., D., and Answer: <letter>.
- The target question must contain Question, A., B., C., and D., but must not contain Answer.
- The target_answer must be one letter only: A, B, C, or D.
- Do not use placeholder text such as "...".
- Keep every question and choice short and easy to read.
- Keep chunk_names short, like "demo 1: helpful".

Return JSON only. Use exactly this schema and level of detail:
{{
  "question": "Question: Which device stores data permanently?\nA. CPU\nB. RAM\nC. Hard drive\nD. Monitor",
  "target_answer": "C",
  "chunk_names": ["demo 1: helpful", "demo 2: similar", "demo 3: irrelevant", "demo 4: helpful", "demo 5: misleading"],
  "chunk_kinds": ["helpful", "helpful", "irrelevant", "helpful", "misleading"],
  "context_chunks": [
    "Demo 1:\nQuestion: Which device is used for long-term file storage?\nA. Keyboard\nB. Hard drive\nC. Mouse\nD. Speaker\nAnswer: B",
    "Demo 2:\nQuestion: Which memory is temporary and loses data when power is off?\nA. RAM\nB. Hard drive\nC. SSD\nD. DVD\nAnswer: A",
    "Demo 3:\nQuestion: Which planet is known as the Red Planet?\nA. Earth\nB. Venus\nC. Mars\nD. Jupiter\nAnswer: C",
    "Demo 4:\nQuestion: Which component performs most computer calculations?\nA. Monitor\nB. CPU\nC. Printer\nD. Router\nAnswer: B",
    "Demo 5:\nQuestion: Which device displays images to the user?\nA. Monitor\nB. RAM\nC. Hard drive\nD. Battery\nAnswer: A"
  ]
}}
""".strip()


def format_case_generation_prompt(tokenizer: AutoTokenizer, prompt: str) -> str:
    """Format the case-generation prompt for an instruction-tuned model.
    将 case 生成 prompt 格式化成 instruction model 更容易响应的 chat 格式。
    """
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:  # noqa: BLE001
        return prompt



def _has_choice_lines(text: str) -> bool:
    """Return whether text contains A-D multiple-choice lines.
    判断文本里是否包含完整的 A-D 选项行。
    """
    return all(
        re.search(rf"(?im)^\s*{letter}\.\s+\S", text)
        for letter in ("A", "B", "C", "D")
    )


def _has_answer_line(text: str) -> bool:
    """Return whether text contains an answer letter line.
    判断文本里是否包含 Answer: A/B/C/D 这一行。
    """
    return re.search(r"(?im)^\s*Answer\s*:\s*[ABCD]\b", text) is not None


def _validate_few_shot_demo_text(chunk: str, index: int) -> None:
    """Require one generated demonstration to include question, choices, and answer.
    要求每个自动生成的 demonstration 都包含题目、选项和答案。
    """
    if "..." in chunk:
        msg = f"Generated demo {index} still contains placeholder text '...'."
        raise ValueError(msg)
    if not re.search(r"(?im)^\s*Question\s*:", chunk):
        msg = f"Generated demo {index} is missing a Question line."
        raise ValueError(msg)
    if not _has_choice_lines(chunk):
        msg = f"Generated demo {index} is missing complete A-D choices."
        raise ValueError(msg)
    if not _has_answer_line(chunk):
        msg = f"Generated demo {index} is missing an Answer: A/B/C/D line."
        raise ValueError(msg)

def _validate_generated_few_shot_case(data: dict[str, object]) -> ContextCase:
    """Validate and normalize a Gemma-generated few-shot case.
    校验并标准化 Gemma 自动生成的 few-shot case。
    """
    required = ["question", "target_answer", "context_chunks", "chunk_names", "chunk_kinds"]
    missing = [key for key in required if key not in data]
    if missing:
        msg = f"Generated few-shot case is missing keys: {missing}"
        raise ValueError(msg)

    question = str(data["question"]).strip()
    target_answer = str(data["target_answer"]).strip().upper()
    context_chunks = [str(chunk).strip() for chunk in data["context_chunks"]]  # type: ignore[index]
    chunk_names = [_short_label(str(name).strip()) for name in data["chunk_names"]]  # type: ignore[index]
    chunk_kinds = [str(kind).strip().lower() for kind in data["chunk_kinds"]]  # type: ignore[index]

    lengths = {len(context_chunks), len(chunk_names), len(chunk_kinds)}
    if len(lengths) != 1:
        msg = "Generated few-shot chunk_names, chunk_kinds, and context_chunks must have the same length."
        raise ValueError(msg)
    if len(context_chunks) != 5:
        msg = "Generated few-shot case must contain exactly 5 demonstrations."
        raise ValueError(msg)
    if target_answer not in {"A", "B", "C", "D"}:
        msg = "Generated few-shot target_answer must be one answer letter: A, B, C, or D."
        raise ValueError(msg)
    if not question:
        msg = "Generated few-shot question must not be empty."
        raise ValueError(msg)
    if "..." in question:
        msg = "Generated few-shot target question still contains placeholder text '...'."
        raise ValueError(msg)
    if not re.search(r"(?im)^\s*Question\s*:", question):
        msg = "Generated few-shot target question is missing a Question line."
        raise ValueError(msg)
    if not _has_choice_lines(question):
        msg = "Generated few-shot target question is missing complete A-D choices."
        raise ValueError(msg)
    if _has_answer_line(question):
        msg = "Generated few-shot target question must not include an Answer line."
        raise ValueError(msg)

    for index, chunk in enumerate(context_chunks, start=1):
        _validate_few_shot_demo_text(chunk, index)

    allowed_kinds = {"helpful", "irrelevant", "misleading", "harmful", "similar", "distractor"}
    bad_kinds = sorted(set(chunk_kinds) - allowed_kinds)
    if bad_kinds:
        msg = f"Generated few-shot chunk_kinds contain unsupported labels: {bad_kinds}"
        raise ValueError(msg)

    return {
        "question": question,
        "target_answer": target_answer,
        "chunk_names": chunk_names,
        "chunk_kinds": chunk_kinds,
        "context_chunks": context_chunks,
    }


def generate_case_with_gemma(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    prompt: str,
    few_shot: bool = False,
) -> ContextCase:
    """Ask Gemma to generate either a retrieval or few-shot attribution case.
    调用 Gemma 自动生成 retrieval 或 few-shot attribution case。
    """
    formatted_prompt = format_case_generation_prompt(tokenizer, prompt)
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=GENERATED_CASE_MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = generated[0, inputs["input_ids"].shape[-1] :]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    if not generated_text:
        full_output = tokenizer.decode(generated[0], skip_special_tokens=False)
        msg = "Gemma returned empty generated text. Last decoded output:\n" + full_output[-1000:]
        raise ValueError(msg)

    data = _extract_json_object(generated_text)
    if few_shot:
        return _validate_generated_few_shot_case(data)
    return _validate_generated_case(data)


def resolve_device() -> str:
    """Resolve DEVICE into the actual runtime device.
    将 DEVICE 解析成实际运行设备。
    """
    if DEVICE == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if DEVICE in {"cpu", "cuda"}:
        if DEVICE == "cuda" and not torch.cuda.is_available():
            print("CUDA was requested but is not available; falling back to CPU.")
            return "cpu"
        return DEVICE
    msg = 'DEVICE must be "auto", "cpu", or "cuda".'
    raise ValueError(msg)


def select_case(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
) -> tuple[str, ContextCase]:
    """Select the case for the current demo mode and case source.
    根据当前 demo mode 和 case source 选择 case。
    """
    if DEMO_MODE not in {"retrieval", "few_shot"}:
        msg = 'DEMO_MODE must be either "retrieval" or "few_shot".'
        raise ValueError(msg)

    if CASE_SOURCE not in {"manual", "gemma_generated", "mmlu_dataset"}:
        msg = 'CASE_SOURCE must be "manual", "gemma_generated", or "mmlu_dataset".'
        raise ValueError(msg)

    if CASE_SOURCE == "mmlu_dataset" and DEMO_MODE != "few_shot":
        msg = 'CASE_SOURCE="mmlu_dataset" is only supported when DEMO_MODE="few_shot".'
        raise ValueError(msg)

    if DEMO_MODE == "few_shot" and CASE_SOURCE == "manual":
        return FEW_SHOT_CASE_NAME, get_few_shot_case(FEW_SHOT_CASE_NAME)

    if DEMO_MODE == "few_shot" and CASE_SOURCE == "mmlu_dataset":
        mmlu_case = load_mmlu_few_shot_case(MMLU_SUBJECT)
        return f"mmlu_{_slugify(MMLU_SUBJECT)}", mmlu_case

    if DEMO_MODE == "retrieval" and CASE_SOURCE == "manual":
        return CASE_NAME, get_case(CASE_NAME)

    try:
        if DEMO_MODE == "few_shot":
            generated_case = generate_case_with_gemma(
                tokenizer=tokenizer,
                model=model,
                prompt=build_few_shot_generation_prompt(GENERATED_FEW_SHOT_TOPIC),
                few_shot=True,
            )
            return f"generated_few_shot_{_slugify(generated_case['target_answer'])}", generated_case

        generated_case = generate_case_with_gemma(
            tokenizer=tokenizer,
            model=model,
            prompt=build_case_generation_prompt(GENERATED_RETRIEVAL_TOPIC),
            few_shot=False,
        )
        return f"generated_retrieval_{_slugify(generated_case['target_answer'])}", generated_case
    except Exception as error:  # noqa: BLE001
        print("\nGemma case generation failed; falling back to manual case.")
        print(f"Generation error: {error}")
        if DEMO_MODE == "few_shot":
            return FEW_SHOT_CASE_NAME, get_few_shot_case(FEW_SHOT_CASE_NAME)
        return CASE_NAME, get_case(CASE_NAME)


def all_coalitions(n_players: int) -> np.ndarray:
    """Enumerate all binary coalitions.
    枚举所有二进制 coalitions。
    """
    return np.array(
        list(itertools.product([0, 1], repeat=n_players)),
        dtype=bool,
    )


def _score_lookup(coalitions: np.ndarray, scores: np.ndarray) -> dict[tuple[int, ...], float]:
    """Create a lookup table from coalition to value-function score.
    创建 coalition 到 value function 分数的查询表。
    """
    return {
        tuple(coalition.astype(int)): float(score)
        for coalition, score in zip(coalitions, scores, strict=False)
    }


def _as_float(value: object) -> float:
    """Convert scalar or one-element array-like values to float.
    将标量或单元素 array-like 值转成 float。
    """
    return float(np.asarray(value).reshape(-1)[0])


def estimate_single_chunk_effects(
    coalitions: np.ndarray,
    scores: np.ndarray,
) -> np.ndarray:
    """Estimate the individual effect of each context chunk.
    估计每个 context chunk 的单独影响。

    This prototype score uses v({i}) - v({}).
    这里使用 v({i}) - v({}) 作为直观的 first-order prototype 指标。
    """
    n_players = coalitions.shape[1]
    lookup = _score_lookup(coalitions, scores)

    empty = tuple([0] * n_players)
    empty_score = lookup[empty]

    effects = np.zeros(n_players, dtype=float)

    for i in range(n_players):
        only_i = [0] * n_players
        only_i[i] = 1
        effects[i] = lookup[tuple(only_i)] - empty_score

    return effects


def estimate_pairwise_interactions(
    coalitions: np.ndarray,
    scores: np.ndarray,
) -> np.ndarray:
    """Estimate simple prototype pairwise interactions.
    估计简单的 prototype 两两交互。

    The prototype formula is v({i,j}) - v({i}) - v({j}) + v({}).
    prototype 公式是 v({i,j}) - v({i}) - v({j}) + v({})。

    The main reported result still uses formal shapiq k-SII.
    主要汇报结果仍然使用正式的 shapiq k-SII。
    """
    n_players = coalitions.shape[1]
    interactions = np.zeros((n_players, n_players), dtype=float)
    lookup = _score_lookup(coalitions, scores)

    empty = tuple([0] * n_players)
    empty_score = lookup[empty]

    for i in range(n_players):
        for j in range(i + 1, n_players):
            only_i = [0] * n_players
            only_j = [0] * n_players
            both = [0] * n_players

            only_i[i] = 1
            only_j[j] = 1
            both[i] = 1
            both[j] = 1

            interaction = (
                lookup[tuple(both)]
                - lookup[tuple(only_i)]
                - lookup[tuple(only_j)]
                + empty_score
            )

            interactions[i, j] = interaction
            interactions[j, i] = interaction

    return interactions


def compute_shapiq_pairwise_interactions(
    imputer: TextImputer,
) -> tuple[np.ndarray, shapiq.InteractionValues]:
    """Compute formal second-order interaction values with shapiq.
    用 shapiq 计算正式的二阶 interaction values。
    """
    computer = shapiq.ExactComputer(
        n_players=imputer.n_players,
        game=imputer,
    )
    interaction_values = computer(
        index=INTERACTION_INDEX,
        order=INTERACTION_ORDER,
    )
    interaction_matrix = interaction_values.get_n_order_values(order=INTERACTION_ORDER)
    return interaction_matrix, interaction_values


def build_display_interaction_values(
    effects: np.ndarray,
    interactions: np.ndarray,
) -> shapiq.InteractionValues:
    """Build a mixed first- and second-order object for force/waterfall plots.
    为 force/waterfall 图构造一个同时包含一阶和二阶结果的展示对象。
    """
    n_players = len(effects)
    display_values: dict[tuple[int, ...], float] = {}

    for i, effect in enumerate(effects):
        display_values[(i,)] = float(effect)

    for i in range(n_players):
        for j in range(i + 1, n_players):
            display_values[(i, j)] = float(interactions[i, j])

    return shapiq.InteractionValues(
        values=display_values,
        index=INTERACTION_INDEX,
        max_order=INTERACTION_ORDER,
        min_order=1,
        n_players=n_players,
        estimated=False,
        baseline_value=0.0,
    )


def plot_single_chunk_effects(
    case_name: str,
    case: ContextCase,
    effects: np.ndarray,
) -> Path:
    """Plot and save individual context chunk effects with shapiq.
    """
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIGURE_DIR / f"context_{case_name}_single_chunk_effects.png"

    first_order_values = shapiq.InteractionValues.from_first_order_array(
        first_order_values=effects,
        index="SV",
    )
    ax = shapiq.bar_plot(
        [first_order_values],
        feature_names=case["chunk_names"],
        show=False,
        abbreviate=False,
        max_display=None,
        plot_base_value=True,
    )
    if ax is None:
        msg = "shapiq.bar_plot returned None although show=False."
        raise RuntimeError(msg)

    fig = ax.figure
    ax.set_title(f"Effect on Gemma answer score: {case['target_answer']!r}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    if SHOW_PLOTS:
        plt.show()

    return output_path


def plot_pairwise_interactions(
    case_name: str,
    case: ContextCase,
    interaction_values: shapiq.InteractionValues,
) -> Path:
    """Plot and save second-order interactions with shapiq's matrix-style UpSet plot.
    使用 shapiq 自带的 matrix-style UpSet plot 绘制并保存二阶 interaction 图。
    """
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIGURE_DIR / f"context_{case_name}_pairwise_interactions.png"

    fig = interaction_values.plot_upset(
        feature_names=case["chunk_names"],
        n_interactions=20,
        color_matrix=True,
        all_features=True,
        show=False,
    )
    if fig is None:
        msg = "interaction_values.plot_upset returned None although show=False."
        raise RuntimeError(msg)

    fig.suptitle(
        f"{INTERACTION_INDEX} context interactions for answer {case['target_answer']!r}",
        y=0.98,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    if SHOW_PLOTS:
        plt.show()

    return output_path


def plot_pairwise_heatmap(
    case_name: str,
    case: ContextCase,
    interactions: np.ndarray,
) -> Path:
    """Plot and save a readable heatmap for pairwise interactions.
    绘制并保存更便于解释的两两 interaction heatmap。
    """
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIGURE_DIR / f"context_{case_name}_pairwise_heatmap.png"

    fig, ax = plt.subplots(figsize=(8, 6))
    max_abs = max(abs(float(np.min(interactions))), abs(float(np.max(interactions))), 1e-6)
    image = ax.imshow(interactions, cmap="coolwarm", vmin=-max_abs, vmax=max_abs)

    ax.set_xticks(range(len(case["chunk_names"])))
    ax.set_yticks(range(len(case["chunk_names"])))
    ax.set_xticklabels(case["chunk_names"], rotation=35, ha="right")
    ax.set_yticklabels(case["chunk_names"])

    for i in range(interactions.shape[0]):
        for j in range(interactions.shape[1]):
            if i == j:
                continue
            ax.text(j, i, f"{interactions[i, j]:.2f}", ha="center", va="center", fontsize=8)

    fig.colorbar(image, ax=ax, label="Pairwise interaction")
    ax.set_title(
        f"{INTERACTION_INDEX} pairwise heatmap for answer {case['target_answer']!r}"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    if SHOW_PLOTS:
        plt.show()

    return output_path


def plot_network_interactions(
    case_name: str,
    case: ContextCase,
    interaction_values: shapiq.InteractionValues,
) -> Path | None:
    """Plot and save shapiq's built-in network interaction visualization.
    使用 shapiq 自带的 network plot 保存 interaction 关系图。
    """
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIGURE_DIR / f"context_{case_name}_pairwise_network.png"

    try:
        result = interaction_values.plot_network(
            feature_names=case["chunk_names"],
            show=False,
        )
    except Exception as error:  # noqa: BLE001
        print(f"Warning: could not create shapiq network plot: {error}")
        return None

    if result is None:
        print("Warning: interaction_values.plot_network returned None although show=False.")
        return None

    fig, _ax = result
    fig.suptitle(
        f"{INTERACTION_INDEX} network interactions for answer {case['target_answer']!r}",
        y=0.98,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    if SHOW_PLOTS:
        plt.show()

    return output_path


def plot_force_interactions(
    case_name: str,
    case: ContextCase,
    interaction_values: shapiq.InteractionValues,
) -> Path | None:
    """Plot and save shapiq's built-in force plot if available.
    如果当前环境支持，则保存 shapiq 自带的 force plot。
    """
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIGURE_DIR / f"context_{case_name}_force.png"

    try:
        fig = interaction_values.plot_force(
            feature_names=np.asarray(case["chunk_names"], dtype=object),
            show=False,
            abbreviate=False,
        )
    except Exception as error:  # noqa: BLE001
        print(f"Warning: could not create shapiq force plot: {error}")
        return None

    if fig is None:
        print("Warning: interaction_values.plot_force returned None although show=False.")
        return None

    fig.suptitle(
        f"{INTERACTION_INDEX} force plot for answer {case['target_answer']!r}",
        y=0.98,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    if SHOW_PLOTS:
        plt.show()

    return output_path


def plot_waterfall_interactions(
    case_name: str,
    case: ContextCase,
    interaction_values: shapiq.InteractionValues,
) -> Path | None:
    """Plot and save shapiq's built-in waterfall plot if available.
    如果当前环境支持，则保存 shapiq 自带的 waterfall plot。
    """
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIGURE_DIR / f"context_{case_name}_waterfall.png"

    try:
        plt.figure()
        ax = interaction_values.plot_waterfall(
            feature_names=np.asarray(case["chunk_names"], dtype=object),
            show=False,
            abbreviate=False,
            max_display=20,
        )
    except Exception as error:  # noqa: BLE001
        print(f"Warning: could not create shapiq waterfall plot: {error}")
        return None

    if ax is None:
        print("Warning: interaction_values.plot_waterfall returned None although show=False.")
        return None

    fig = ax.figure
    ax.set_title(f"{INTERACTION_INDEX} waterfall for answer {case['target_answer']!r}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    if SHOW_PLOTS:
        plt.show()

    return output_path


def print_automatic_summary(
    case: ContextCase,
    effects: np.ndarray,
    interactions: np.ndarray,
) -> None:
    """Print a short automatic summary of the most important results.
    打印最重要结果的简短自动总结。
    """
    top_positive_idx = int(np.argmax(effects))
    top_negative_idx = int(np.argmin(effects))

    # Only summarize real pairs; ignore the diagonal.
    # 只总结真正的两两组合，不看对角线。
    pair_indices = np.triu_indices_from(interactions, k=1)
    pair_values = interactions[pair_indices]

    strongest_positive_position = int(np.argmax(pair_values))
    strongest_positive = (
        int(pair_indices[0][strongest_positive_position]),
        int(pair_indices[1][strongest_positive_position]),
    )

    strongest_negative_position = int(np.argmin(pair_values))
    strongest_negative = (
        int(pair_indices[0][strongest_negative_position]),
        int(pair_indices[1][strongest_negative_position]),
    )

    print("\nAutomatic summary:")
    print(
        "Top positive chunk: "
        f"{case['chunk_names'][top_positive_idx]} "
        f"({case['chunk_kinds'][top_positive_idx]}), "
        f"effect={effects[top_positive_idx]:.4f}"
    )
    print(
        "Top negative chunk: "
        f"{case['chunk_names'][top_negative_idx]} "
        f"({case['chunk_kinds'][top_negative_idx]}), "
        f"effect={effects[top_negative_idx]:.4f}"
    )
    print(
        "Strongest positive interaction: "
        f"{case['chunk_names'][strongest_positive[0]]} + "
        f"{case['chunk_names'][strongest_positive[1]]}, "
        f"value={interactions[strongest_positive]:.4f}"
    )
    print(
        "Strongest negative interaction: "
        + (
            f"{case['chunk_names'][strongest_negative[0]]} + "
            f"{case['chunk_names'][strongest_negative[1]]}, "
            f"value={interactions[strongest_negative]:.4f}"
            if interactions[strongest_negative] < 0
            else "none below 0.0000"
        )
    )


def main() -> None:
    """Run the context attribution demo.
    运行 context attribution demo。

    The workflow loads the model, creates TextImputer, enumerates coalitions,
    computes the value function, computes k-SII interactions, and saves figures.
    流程包括加载模型、创建 TextImputer、枚举 coalitions、计算 value function、计算 k-SII interaction、画图并保存结果。
    """
    runtime_device = resolve_device()

    print(f"Demo mode: {DEMO_MODE}")
    print(f"Case source: {CASE_SOURCE}")
    print(f"Configured device: {DEVICE}")
    print(f"Runtime device: {runtime_device}")
    if DEMO_MODE == "few_shot":
        if CASE_SOURCE == "manual":
            print(f"Few-shot case name: {FEW_SHOT_CASE_NAME}")
        elif CASE_SOURCE == "mmlu_dataset":
            print(f"MMLU dataset: {MMLU_DATASET_NAME}")
            print(f"MMLU subject: {MMLU_SUBJECT}")
            print(f"MMLU split: {MMLU_SPLIT}")
        else:
            print(f"Generated few-shot topic: {GENERATED_FEW_SHOT_TOPIC}")
    else:
        if CASE_SOURCE == "manual":
            print(f"Manual retrieval case name: {CASE_NAME}")
        else:
            print(f"Generated retrieval topic: {GENERATED_RETRIEVAL_TOPIC}")

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        local_files_only=LOCAL_FILES_ONLY,
    )

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        local_files_only=LOCAL_FILES_ONLY,
    ).to(runtime_device)

    selected_case_name, case = select_case(tokenizer, model)
    text = build_text(case)

    print(f"\nSelected case: {selected_case_name}")
    print(f"Target answer: {case['target_answer']!r}")
    print("Value function:")
    print("v(S) = log P_Gemma(target_answer | question + selected context chunks)")

    print("\nCreating TextImputer...")
    imputer = TextImputer(
        model=model,
        tokenizer=tokenizer,
        text=text,
        model_type="causal_lm",
        target_label=case["target_answer"],
        prompt_template=PROMPT_TEMPLATE,
        player_level="sentence",
        player_strategy=ContextChunkPlayerStrategy(case),
        perturbation_type="removal",
        device=runtime_device,
    )

    print("\nQuestion:")
    print(case["question"])

    print("\nContext chunks:")
    for idx, (name, kind, chunk) in enumerate(
        zip(case["chunk_names"], case["chunk_kinds"], case["context_chunks"], strict=False)
    ):
        print(f"[{idx}] {name} ({kind}): {chunk}")

    print("\nTextImputer players:")
    players = imputer.player_strategy.get_players()
    for idx, player in enumerate(players):
        print(f"[{idx}] {player}")

    n_players = imputer.n_players
    coalitions = all_coalitions(n_players)

    print("\nNumber of players:", n_players)
    print("Number of coalitions:", len(coalitions))

    full_prediction = imputer.full_prediction
    full_score = _as_float(full_prediction() if callable(full_prediction) else full_prediction)
    print(f"\nFull context score for {case['target_answer']!r}: {full_score:.4f}")

    print("\nEvaluating all coalitions...")
    scores = np.asarray(imputer.value_function(coalitions), dtype=float).reshape(-1)

    effects = estimate_single_chunk_effects(coalitions, scores)
    prototype_interactions = estimate_pairwise_interactions(coalitions, scores)

    print(f"\nComputing formal shapiq {INTERACTION_INDEX} interactions...")
    interactions, interaction_values = compute_shapiq_pairwise_interactions(imputer)

    print("\nSingle chunk effects:")
    for name, kind, effect in zip(
        case["chunk_names"], case["chunk_kinds"], effects, strict=False
    ):
        print(f"{name:20s} ({kind:10s}): {effect:.4f}")

    print("\nPrototype pairwise interactions:")
    print(prototype_interactions)

    print(f"\nFormal shapiq {INTERACTION_INDEX} pairwise interactions:")
    print(interactions)
    print("\nFormal interaction values object:")
    print(interaction_values.get_n_order(order=INTERACTION_ORDER))

    print_automatic_summary(case, effects, interactions)

    figure_case_name = f"{DEMO_MODE}_{selected_case_name}"
    display_interaction_values = build_display_interaction_values(effects, interactions)
    single_effects_path = plot_single_chunk_effects(figure_case_name, case, effects)
    interactions_path = plot_pairwise_interactions(figure_case_name, case, interaction_values)
    heatmap_path = plot_pairwise_heatmap(figure_case_name, case, interactions)
    network_path = plot_network_interactions(figure_case_name, case, interaction_values)
    force_path = plot_force_interactions(figure_case_name, case, display_interaction_values)
    waterfall_path = plot_waterfall_interactions(
        figure_case_name,
        case,
        display_interaction_values,
    )

    print("\nSaved figures:")
    for path in (
        single_effects_path,
        interactions_path,
        heatmap_path,
        network_path,
        force_path,
        waterfall_path,
    ):
        if path is not None:
            print(path)


if __name__ == "__main__":
    main()

