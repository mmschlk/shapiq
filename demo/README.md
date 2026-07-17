# Context Attribution Demo

This demo explains which parts of a prompt influence a Gemma model's target answer. It uses `TextImputer` to remove prompt parts, Gemma to score the target answer, and `shapiq` to compute interaction values.

## Quick Start

Run all commands from the repository root.

First make the local `src` package importable:

```powershell
$env:PYTHONPATH="$PWD\src;$env:PYTHONPATH"
```

Start the Streamlit app:

```powershell
streamlit run demo\context_app.py
```

The app lets you choose:

- demo mode: `retrieval` or `few_shot`
- case source: manual, Gemma-generated, or MMLU dataset
- device: `auto`, `cpu`, or `cuda`
- MMLU subject, for example `college_computer_science`

You can also run the backend script directly:

```powershell
$env:SHAPIQ_DEMO_MODE="retrieval"
$env:SHAPIQ_CASE_SOURCE="manual"
$env:SHAPIQ_DEVICE="auto"
$env:SHAPIQ_SHOW_PLOTS="0"
python demo\context_attribution.py
```

For MMLU few-shot mode:

```powershell
$env:SHAPIQ_DEMO_MODE="few_shot"
$env:SHAPIQ_CASE_SOURCE="mmlu_dataset"
$env:SHAPIQ_MMLU_SUBJECT="college_computer_science"
$env:SHAPIQ_DEVICE="auto"
$env:SHAPIQ_SHOW_PLOTS="0"
python demo\context_attribution.py
```

## Requirements

Install the local project and the demo dependencies:

```powershell
python -m pip install -e .
python -m pip install streamlit transformers torch accelerate matplotlib numpy
```

Optional MMLU support requires:

```powershell
python -m pip install datasets
```

The demo uses the Hugging Face model:

```text
google/gemma-4-E2B-it
```

The model may be loaded from the local Hugging Face cache or downloaded automatically depending on the user's environment.

## What This Demo Does

The demo asks:

> When an LLM receives a prompt with extra context or few-shot examples, which prompt parts actually influence the final answer?

It treats prompt parts as players in a cooperative game. For each coalition `S`, selected players are kept and missing players are removed. Gemma then scores the target answer:

```text
v(S) = log P_Gemma(target answer | target question + selected players in S)
```

A higher value means the selected players make Gemma more likely to produce the target answer.

## Pipeline

The attribution pipeline is:

```text
case -> players -> coalitions -> masked prompts -> Gemma scores -> shapiq interactions -> figures
```

In more detail:

1. Select a manual, Gemma-generated, or MMLU case.
2. Treat context chunks or few-shot demonstrations as players.
3. Enumerate coalitions of selected players.
4. Use `TextImputer` to keep selected players and remove missing players.
5. Use Gemma to score the target answer for each coalition.
6. Use `shapiq` to compute interaction values and visualizations.

## Demo Modes

### Retrieval Mode

Each retrieved context chunk is one player. A retrieval case can include supporting, misleading, irrelevant, and contradictory evidence. This mode explains which retrieved chunks support, distract from, or contradict the target answer.

### Few-Shot Mode

Each question-choice-answer demonstration is one player. This mode explains which few-shot examples help Gemma answer the target question and which examples may confuse it.

A demonstration is formatted like:

```text
Question: ...
A. ...
B. ...
C. ...
D. ...
Answer: ...
```

The target question is placed after the demonstrations and does not include the answer.

## Case Sources

- **Manual case:** hand-written and stable, best for live demos.
- **Gemma-generated case:** Gemma creates a case first, then the same attribution pipeline is applied. This is useful for exploration but can fail validation.
- **MMLU dataset case:** MMLU provides real multiple-choice questions; Gemma is still the scoring model.

In short: Gemma is the model being explained, and MMLU is only a data source.

## Outputs

The Streamlit app shows the value function, run settings, question, players, key findings, generated figures, and an expandable console log.

Figures are saved under:

```text
demo/figures/
```

Main figure types:

- **Single-player effect plot:** shows whether each player increases or decreases the target-answer score.
- **shapiq matrix-style plot:** built-in shapiq interaction visualization.
- **shapiq network plot:** players are nodes and pairwise interactions are edges.
- **shapiq force plot:** shows effects pushing the score up or down.
- **shapiq waterfall plot:** ranks contribution values as an additive explanation.
- **Pairwise heatmap:** a readable matrix view of order-2 pairwise interactions.

Positive values mean helpful or supportive effects. Negative values mean distracting, confusing, or interfering effects. Positive pairwise interactions mean two players work better together than expected; negative pairwise interactions mean they interfere with each other.

## Useful Environment Variables

| Variable | Example | Meaning |
| --- | --- | --- |
| `SHAPIQ_DEMO_MODE` | `retrieval` | Select retrieval or few-shot mode. |
| `SHAPIQ_CASE_SOURCE` | `manual` | Select manual, Gemma-generated, or MMLU source. |
| `SHAPIQ_DEVICE` | `auto` | Use `auto`, `cpu`, or `cuda`. |
| `SHAPIQ_SHOW_PLOTS` | `0` | Save figures without pop-up windows. |
| `SHAPIQ_MMLU_SUBJECT` | `college_computer_science` | MMLU subject for few-shot MMLU mode. |
| `SHAPIQ_GENERATED_RETRIEVAL_TOPIC` | `simple general-knowledge question` | Topic for Gemma-generated retrieval cases. |
| `SHAPIQ_GENERATED_FEW_SHOT_TOPIC` | `simple MMLU-style multiple-choice question` | Topic for Gemma-generated few-shot cases. |

## Troubleshooting

### `ModuleNotFoundError: No module named 'shapiq'`

Set `PYTHONPATH` from the repository root:

```powershell
$env:PYTHONPATH="$PWD\src;$env:PYTHONPATH"
```

Or install the local project:

```powershell
python -m pip install -e .
```

### Streamlit shows `Run finished with exit code 3221225477`

This is usually a Windows access-violation crash during model loading. The scripts set:

```text
HF_ENABLE_PARALLEL_LOADING=false
```

If it still happens, restart the terminal and Streamlit process.

### MMLU source fails

Install the optional dependency:

```powershell
python -m pip install datasets
```

Also make sure the subject name is valid, for example `college_computer_science`.

### Gemma-generated case falls back to manual case

This happens when Gemma generates text that does not match the required JSON format or labels. Use manual cases for stable live demos.
