"""Context attribution demo：用 Gemma 和 TextImputer 分析检索上下文。

这个脚本研究哪些 context chunks 支持模型生成目标答案。每个 context chunk 被当作一个 player，coalition 决定哪些 chunks 会提供给模型。

价值函数：
    v(S) = log P_Gemma(target_answer | question + context chunks in S)

分数越高，表示 Gemma 在看到所选 context chunks 后，越倾向生成目标答案。"""

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


# 避免 tokenizer 并行 warning。
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# 模型设置
# ---------------------------------------------------------------------------

MODEL_NAME = "google/gemma-4-E2B-it"

# 主 heatmap 使用的正式 shapiq interaction index。
INTERACTION_INDEX = "k-SII"
INTERACTION_ORDER = 2

# 优先使用已经下载到 HuggingFace cache 里的模型文件，避免每次运行都访问网络。
LOCAL_FILES_ONLY = True

# CPU 模式更慢，但可以避免小机器上 accelerate 磁盘 offload 不稳定导致的崩溃。
DEVICE = "cpu"


class ContextCase(TypedDict):
    """可复用的 context attribution 案例。

    question 是 retrieved chunks 后面的问题。target_answer 是 value function 里要打分的目标答案。chunk_names 是画图使用的短标签。chunk_kinds 表示 supporting、irrelevant 或 misleading 类型。context_chunks 是作为 players 的 retrieved context chunks。"""

    question: str
    target_answer: str
    chunk_names: list[str]
    chunk_kinds: list[str]
    context_chunks: list[str]


# 每个 case 都混合了有用、无关和误导性的 context。
CASES: dict[str, ContextCase] = {
    "eiffel_tower": {
        "question": "Where is the Eiffel Tower located?",
        "target_answer": "Paris",
        "chunk_names": [
            "support: tower",
            "mislead: vegas",
            "support: paris",
        ],
        "chunk_kinds": [
            "supporting",
            "misleading",
            "supporting",
        ],
        "context_chunks": [
            "The Eiffel Tower is located in Paris, France.",
            "A replica of the Eiffel Tower can be found in Las Vegas.",
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


# 改这里可以切换测试案例。
CASE_NAME = "eiffel_tower"

# 图片保存到这里，方便汇报使用。
FIGURE_DIR = Path(__file__).resolve().parent / "figures"


def get_case(case_name: str) -> ContextCase:
    """返回当前选择的 context attribution 案例。想测试另一个例子，只需要修改 CASE_NAME。"""
    if case_name not in CASES:
        available = ", ".join(CASES)
        msg = f"Unknown case {case_name!r}. Available cases: {available}"
        raise ValueError(msg)
    return CASES[case_name]


def build_text(case: ContextCase) -> str:
    """把 context chunks、question 和 answer prefix 拼成完整 prompt。"""
    context = "\n".join(
        f"Context {idx + 1}: {chunk}"
        for idx, chunk in enumerate(case["context_chunks"])
    )
    return f"{context}\n\nQuestion: {case['question']}\nAnswer:"


class ContextChunkPlayerStrategy:
    """只把 retrieved context chunks 当成 players 的策略。

    TextImputer 内置的 sentence strategy 会把 question 和 answer prefix 也当成 players。在 RAG attribution 里，我们希望 question 固定，只解释 retrieved chunks，所以这里自定义一个 player strategy。"""

    def __init__(self, case: ContextCase) -> None:
        """保存当前 context attribution 案例。"""
        self.case = case
        self.context_chunks = case["context_chunks"]

    def get_players(self) -> list[str]:
        """返回作为 players 的 context chunks。"""
        return self.context_chunks

    @property
    def n_players(self) -> int:
        """返回 context chunk players 的数量。"""
        return len(self.context_chunks)

    def coalition_to_text(
        self,
        coalition: np.ndarray,
        perturbation_strategy: object,  # noqa: ARG002
    ) -> str:
        """根据选中的 context chunks 构造 prompt。没有选中的 chunks 会被删除。question 和 answer prefix 始终保留，因为它们不是要解释的 players。"""
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
    """枚举所有二进制 coalitions。"""
    return np.array(
        list(itertools.product([0, 1], repeat=n_players)),
        dtype=bool,
    )


def _score_lookup(coalitions: np.ndarray, scores: np.ndarray) -> dict[tuple[int, ...], float]:
    """创建 coalition 到 value function 分数的查询表。"""
    return {
        tuple(coalition.astype(int)): float(score)
        for coalition, score in zip(coalitions, scores, strict=False)
    }


def estimate_single_chunk_effects(
    coalitions: np.ndarray,
    scores: np.ndarray,
) -> np.ndarray:
    """估计每个 context chunk 的单独影响。这里使用 v({i}) - v({}) 作为直观的 first-order prototype 指标。"""
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
    """估计简单的两两交互。这个 prototype 公式是 v({i,j}) - v({i}) - v({j}) + v({})。主结果仍然使用正式的 shapiq k-SII。"""
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
    """用 shapiq 计算正式的二阶 interaction values。"""
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
    """绘制并保存每个 context chunk 的单独影响图。"""
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
    """绘制并保存二阶 interaction heatmap。"""
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
    """打印最重要结果的简短自动总结。"""
    top_positive_idx = int(np.argmax(effects))
    top_negative_idx = int(np.argmin(effects))

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
    """运行 context attribution demo。

    流程包括加载模型、创建 TextImputer、枚举 coalitions、计算 value function、计算 k-SII interaction、画图并保存结果。"""
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
        player_level="sentence",
        player_strategy=ContextChunkPlayerStrategy(case),
        perturbation_type="mask",
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
