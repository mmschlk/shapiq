"""Context attribution demo: analyze retrieved context chunks with Gemma and TextImputer.
Context attribution demo：用 Gemma 和 TextImputer 分析检索到的上下文片段。

Each context chunk is treated as one span-level player.
每个 context chunk 被当作一个 span-level player。

For each coalition, selected chunks are kept and missing chunks are removed.
对每个 coalition，被选中的 chunks 会保留，没有被选中的 chunks 会被移除。

Value function:
价值函数：
    v(S) = log P_Gemma(target_answer | question + context chunks in S)

A higher score means Gemma is more likely to produce the target answer.
分数越高，表示 Gemma 越倾向生成目标答案。
"""

from __future__ import annotations

import itertools
import os
from pathlib import Path
from typing import TypedDict

import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

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
DEVICE = "cpu"

# Keep the causal LM prompt exactly as build_text() creates it.
# 保持 causal LM 的 prompt 就是 build_text() 构造出来的内容。
PROMPT_TEMPLATE = "{text}"


class ContextCase(TypedDict):
    """Reusable context attribution case.
    可复用的 context attribution 案例。

    question: the question asked after the retrieved chunks.
    question：放在 retrieved chunks 后面的具体问题。

    target_answer: the answer scored by the value function.
    target_answer：value function 要打分的目标答案。

    chunk_names: short labels used in plots.
    chunk_names：画图时使用的短标签。

    chunk_kinds: supporting, irrelevant, or misleading labels.
    chunk_kinds：标注每个 chunk 是 supporting、irrelevant 还是 misleading。

    context_chunks: retrieved chunks used as players.
    context_chunks：作为 players 的 retrieved context chunks。
    """

    question: str
    target_answer: str
    chunk_names: list[str]
    chunk_kinds: list[str]
    context_chunks: list[str]


# Each case mixes supporting, irrelevant, and misleading context.
# 每个 case 都混合了支持、无关和误导性的 context。
CASES: dict[str, ContextCase] = {
    "eiffel_tower": {
        "question": "Where is the Eiffel Tower located?",
        "target_answer": "Paris",
        "chunk_names": [
            "support: tower",
            "mislead: vegas",
            "irrelevant: berlin",
            "support: paris",
        ],
        "chunk_kinds": [
            "supporting",
            "misleading",
            "irrelevant",
            "supporting",
        ],
        "context_chunks": [
            "The Eiffel Tower is located in Paris, France.",
            "A replica of the Eiffel Tower can be found in Las Vegas.",
            "Berlin is the capital of Germany.",
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


# Change this value to run another test case.
# 改这里就可以切换测试案例。
CASE_NAME = "eiffel_tower"

# Figures are saved here for reports and slides.
# 图片会保存到这里，方便放进汇报和 slides。
FIGURE_DIR = Path(__file__).resolve().parent / "figures"


def get_case(case_name: str) -> ContextCase:
    """Return the selected context attribution case.
    返回当前选择的 context attribution 案例。

    To test another example, change CASE_NAME.
    如果想测试另一个例子，只需要修改 CASE_NAME。
    """
    if case_name not in CASES:
        available = ", ".join(CASES)
        msg = f"Unknown case {case_name!r}. Available cases: {available}"
        raise ValueError(msg)
    return CASES[case_name]


def build_text(case: ContextCase) -> str:
    """Build the full prompt from context chunks, question, and answer prefix.
    把 context chunks、question 和 answer prefix 拼成完整 prompt。
    """
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

        selected_context = "\n".join(
            f"Context {idx + 1}: {chunk}"
            for idx, (keep, chunk) in enumerate(
                zip(coalition, self.context_chunks, strict=False)
            )
            if keep
        )

        if selected_context:
            return f"{selected_context}\n\nQuestion: {self.case['question']}\nAnswer:"

        return f"Question: {self.case['question']}\nAnswer:"


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


def plot_single_chunk_effects(
    case_name: str,
    case: ContextCase,
    effects: np.ndarray,
) -> Path:
    """Plot and save individual context chunk effects.
    绘制并保存每个 context chunk 的单独影响图。
    """
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIGURE_DIR / f"context_{case_name}_single_chunk_effects.png"

    fig, ax = plt.subplots(figsize=(9, 4))

    colors = ["tab:red" if value > 0 else "tab:blue" for value in effects]
    ax.bar(case["chunk_names"], effects, color=colors)

    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_ylabel("Single chunk effect")
    ax.set_title(f"Effect on Gemma answer score: {case['target_answer']!r}")
    ax.tick_params(axis="x", rotation=35)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.show()

    return output_path


def plot_pairwise_interactions(
    case_name: str,
    case: ContextCase,
    interactions: np.ndarray,
) -> Path:
    """Plot and save the second-order interaction heatmap.
    绘制并保存二阶 interaction heatmap。
    """
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIGURE_DIR / f"context_{case_name}_pairwise_interactions.png"

    fig, ax = plt.subplots(figsize=(8, 6))

    max_abs = max(abs(interactions.min()), abs(interactions.max()), 1e-6)

    image = ax.imshow(
        interactions,
        cmap="coolwarm",
        vmin=-max_abs,
        vmax=max_abs,
    )

    ax.set_xticks(range(len(case["chunk_names"])))
    ax.set_yticks(range(len(case["chunk_names"])))
    ax.set_xticklabels(case["chunk_names"], rotation=35, ha="right")
    ax.set_yticklabels(case["chunk_names"])

    fig.colorbar(image, ax=ax, label="Pairwise interaction")
    ax.set_title(
        f"{INTERACTION_INDEX} context interactions for answer {case['target_answer']!r}"
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
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
    case = get_case(CASE_NAME)
    text = build_text(case)

    print(f"Selected case: {CASE_NAME}")
    print(f"Target answer: {case['target_answer']!r}")
    print("Value function:")
    print("v(S) = log P_Gemma(target_answer | question + selected context chunks)")

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        local_files_only=LOCAL_FILES_ONLY,
    )

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        local_files_only=LOCAL_FILES_ONLY,
    )

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
        device=DEVICE,
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

    full_score = imputer.full_prediction()
    print(f"\nFull context score for {case['target_answer']!r}: {full_score:.4f}")

    print("\nEvaluating all coalitions...")
    scores = imputer.value_function(coalitions)

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

    single_effects_path = plot_single_chunk_effects(CASE_NAME, case, effects)
    interactions_path = plot_pairwise_interactions(CASE_NAME, case, interactions)

    print("\nSaved figures:")
    print(single_effects_path)
    print(interactions_path)


if __name__ == "__main__":
    main()

