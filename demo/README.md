# Context Attribution Demo

This folder contains the current text/LLM demo for explaining how different parts of a prompt affect a Gemma model's target answer. The demo is built around `TextImputer` and `shapiq` interaction values.

The main entry points are:

- `context_app.py`: Streamlit UI for live demonstrations.
- `context_attribution.py`: command-line pipeline for running the attribution experiment.

## What This Demo Does

The demo asks a simple question:

> When an LLM receives a prompt with extra context or few-shot examples, which parts of the prompt actually influence the final answer?

Instead of only checking whether the model answers correctly, the demo treats prompt parts as players in a cooperative game. It evaluates many coalitions of players and uses shapiq to estimate which players are helpful, harmful, or interactive.

The value function is:

```text
v(S) = log P_Gemma(target answer | target question + selected players in S)
```

For each coalition `S`, the selected players are kept in the prompt, the missing players are removed, and Gemma scores the target answer. A higher value means that the selected players make Gemma more likely to produce the target answer.

## Pipeline

The demo follows the same pipeline in both retrieval mode and few-shot mode.

1. **Select a case**

   The case can be manual, Gemma-generated, or loaded from MMLU. Each case contains a target question, a target answer, and a list of prompt parts.

2. **Define players**

   In retrieval mode, each player is one retrieved context chunk. In few-shot mode, each player is one full question-choice-answer demonstration.

3. **Build coalitions**

   The script enumerates coalitions of players. A coalition is a subset of players that will be kept in the prompt.

4. **Mask missing players with TextImputer**

   `TextImputer` keeps the selected players and removes the missing players. This creates one prompt for each coalition.

5. **Score each coalition with Gemma**

   Gemma computes the log probability of the target answer for each coalition prompt. These scores are the values used by shapiq.

6. **Compute interactions with shapiq**

   shapiq computes interaction values, mainly order-2 `k-SII` pairwise interactions. The results are summarized in text and saved as figures.

In short:

```text
case -> players -> coalitions -> masked prompts -> Gemma scores -> shapiq interactions -> figures
```

## Demo Modes

### 1. Retrieval Mode

Retrieval mode treats each retrieved evidence chunk as one player.

A retrieval case can include:

- supporting evidence
- misleading evidence
- irrelevant evidence
- contradictory evidence

This mode is useful for asking which retrieved chunks support the model's answer and which chunks distract or contradict it.

### 2. Few-Shot Mode

Few-shot mode treats each question-choice-answer demonstration as one player.

A few-shot player is one full example, usually formatted as:

```text
Question: ...
A. ...
B. ...
C. ...
D. ...
Answer: ...
```

The target question is placed after the demonstrations. This mode is useful for in-context learning attribution: it shows which examples help the model answer the target question and which examples may be confusing or off-topic.

## Case Sources

The Streamlit app supports three case sources.

### Manual Case

Manual cases are hand-written and stable. They are the best choice for live presentations because the output is predictable.

### Gemma-Generated Case

Gemma-generated cases use Gemma to create a small test case first. The generated case is then passed through the same attribution pipeline.

This source is useful for exploration, but it is less stable than a manual case because the generated content can vary or fail validation.

### MMLU Dataset Case

MMLU cases use real multiple-choice questions from the MMLU dataset. This is currently used for the few-shot mode.

Gemma and MMLU have different roles:

- Gemma is the scoring model.
- MMLU is the data source.

When MMLU is selected, MMLU provides the questions and answers, while Gemma still computes the target-answer score.

## Requirements

Run from the repository root.

The demo expects the local shapiq source tree to be importable. In PowerShell, set:

```powershell
$env:PYTHONPATH="$PWD\src;$env:PYTHONPATH"
```

The demo uses the cached Hugging Face model:

```text
google/gemma-4-E2B-it
```

By default, `LOCAL_FILES_ONLY=True` in `context_attribution.py`, so the model should already be cached locally. If the model is not cached, load it once with internet access or change the local-files setting for your own environment.

Optional MMLU support requires the `datasets` package:

```powershell
python -m pip install datasets
```

## Run the Streamlit App

From the repository root:

```powershell
$env:PYTHONPATH="$PWD\src;$env:PYTHONPATH"
streamlit run demo\context_app.py
```

The sidebar lets you choose:

- demo mode: `retrieval` or `few_shot`
- case source: manual, Gemma-generated, or MMLU dataset
- device: `auto`, `cpu`, or `cuda`
- MMLU subject, such as `college_computer_science`

The app shows:

- the value function
- current run settings
- question and players
- key findings
- saved figures
- expandable full console log

## Run from the Command Line

You can also run the attribution script directly.

### Retrieval Mode

```powershell
$env:PYTHONPATH="$PWD\src;$env:PYTHONPATH"
$env:SHAPIQ_DEMO_MODE="retrieval"
$env:SHAPIQ_CASE_SOURCE="manual"
$env:SHAPIQ_DEVICE="auto"
$env:SHAPIQ_SHOW_PLOTS="0"
python demo\context_attribution.py
```

### Few-Shot Mode with Manual Case

```powershell
$env:PYTHONPATH="$PWD\src;$env:PYTHONPATH"
$env:SHAPIQ_DEMO_MODE="few_shot"
$env:SHAPIQ_CASE_SOURCE="manual"
$env:SHAPIQ_DEVICE="auto"
$env:SHAPIQ_SHOW_PLOTS="0"
python demo\context_attribution.py
```

### Few-Shot Mode with MMLU

```powershell
$env:PYTHONPATH="$PWD\src;$env:PYTHONPATH"
$env:SHAPIQ_DEMO_MODE="few_shot"
$env:SHAPIQ_CASE_SOURCE="mmlu_dataset"
$env:SHAPIQ_MMLU_SUBJECT="college_computer_science"
$env:SHAPIQ_DEVICE="auto"
$env:SHAPIQ_SHOW_PLOTS="0"
python demo\context_attribution.py
```

## Useful Environment Variables

| Variable | Example | Meaning |
| --- | --- | --- |
| `SHAPIQ_DEMO_MODE` | `retrieval` | Select retrieval or few-shot mode. |
| `SHAPIQ_CASE_SOURCE` | `manual` | Select manual, Gemma-generated, or MMLU source. |
| `SHAPIQ_DEVICE` | `auto` | Use `auto`, `cpu`, or `cuda`. |
| `SHAPIQ_SHOW_PLOTS` | `0` | Set to `0` for saved figures only, without pop-up windows. |
| `SHAPIQ_MMLU_SUBJECT` | `college_computer_science` | MMLU subject used by the few-shot MMLU case. |
| `SHAPIQ_GENERATED_RETRIEVAL_TOPIC` | `simple general-knowledge question` | Topic prompt for Gemma-generated retrieval cases. |
| `SHAPIQ_GENERATED_FEW_SHOT_TOPIC` | `simple MMLU-style multiple-choice question` | Topic prompt for Gemma-generated few-shot cases. |

The scripts also disable Hugging Face parallel weight loading on Windows:

```text
HF_ENABLE_PARALLEL_LOADING=false
```

This avoids a known access-violation crash during Gemma weight loading in the current Windows environment.

## Output Files

Generated figures are saved under:

```text
demo/figures/
```

The main figure types are:

### Single-Player Effect Plot

This plot shows the effect of each individual player.

- Positive bars increase Gemma's score for the target answer.
- Negative bars reduce Gemma's score for the target answer.

### shapiq Matrix-Style Plot

This plot uses shapiq's built-in visualization style. It ranks interaction values and shows which players are involved in each interaction.

It is useful for consistency with the shapiq library.

### shapiq Network Plot

This plot uses shapiq's built-in network visualization. Players are shown as nodes, and pairwise interactions are shown as edges.

It is useful for explaining interaction structure in a more relational way.

### shapiq Force Plot

This plot uses shapiq's built-in force visualization on a display object that combines single-player effects and pairwise interactions.

It is useful for quickly seeing which effects push the target-answer score up or down.

### shapiq Waterfall Plot

This plot uses shapiq's built-in waterfall visualization on the same display object as the force plot.

It is useful for presenting the contribution values as a ranked additive explanation. If the local plotting backend cannot render it, the demo prints a warning and continues.

### Pairwise Heatmap

This plot shows the order-2 pairwise interaction matrix.

- Red cells indicate positive interaction.
- Blue cells indicate negative interaction.
- Each cell corresponds to a pair of players.

The heatmap is easier to explain in a live presentation because it directly maps player pairs to interaction values.

## How to Interpret the Results

In retrieval mode, a positive chunk effect means that the chunk supports the target answer. A negative chunk effect means that the chunk distracts from the target answer or introduces competing information.

In few-shot mode, demonstrations are not evidence in the same way as retrieval chunks. They are examples that teach the model a task format or reasoning pattern. A positive demonstration effect means the example helps the target question; a negative effect means it may be off-topic, confusing, or misleading for this target question.

Pairwise interactions explain non-additive behavior. A positive pairwise interaction means two players work better together than expected from their separate effects. A negative pairwise interaction means two players interfere with each other.

## Troubleshooting

### `ModuleNotFoundError: No module named 'shapiq'`

Set `PYTHONPATH` from the repository root:

```powershell
$env:PYTHONPATH="$PWD\src;$env:PYTHONPATH"
```

### Streamlit shows `Run finished with exit code 3221225477`

This is a Windows access-violation crash, usually during model weight loading. The scripts set:

```text
HF_ENABLE_PARALLEL_LOADING=false
```

If the error persists, restart the terminal and Streamlit process, then run again.

### MMLU source fails

Install the optional datasets dependency:

```powershell
python -m pip install datasets
```

Also make sure the subject name is a valid MMLU config, for example:

```text
college_computer_science
```

### Gemma-generated case falls back to manual case

This can happen if Gemma generates text that does not match the required JSON format or labels. Manual cases are recommended for stable live demos.

## Current Scope

The current maintained demo is the context attribution workflow. Older prototype demos may exist in local history, but the main presentation path is:

1. launch `context_app.py`
2. choose retrieval or few-shot mode
3. choose manual, Gemma-generated, or MMLU case source
4. run the attribution pipeline
5. explain key findings and figures

