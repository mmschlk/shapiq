# Gemma Few-Shot Generation Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reject malformed generated multiple-choice cases and retry Gemma generation up to three times with corrective feedback.

**Architecture:** Keep case-shape validation in `demo/context_attribution.py`, adding one focused helper for A-D option and answer-line checks. Wrap the existing generation, JSON extraction, and validation pipeline in a bounded retry loop that only catches expected `ValueError` failures and feeds the reason into the next prompt.

**Tech Stack:** Python 3.12, PyTorch tensors, Hugging Face generation interface, pytest, Ruff

---

## File structure

- Modify `demo/context_attribution.py`: add multiple-choice text validation and bounded retry orchestration.
- Create `tests/shapiq/tests_unit/test_context_attribution_demo.py`: fast regression tests using tokenizer/model fakes, without loading Gemma weights.

### Task 1: Enforce complete multiple-choice case text

**Files:**
- Modify: `demo/context_attribution.py:671-713`
- Create: `tests/shapiq/tests_unit/test_context_attribution_demo.py`

- [ ] **Step 1: Write failing validation tests**

Load the demo with `importlib.util.spec_from_file_location`. Define `valid_case()` returning five demonstrations, each with ordered `A.` through `D.` lines and one `Answer: A` line. Add these tests:

```python
def test_generated_few_shot_case_rejects_target_without_choices() -> None:
    case = valid_case()
    case["question"] = "Question: Target?"
    with pytest.raises(ValueError, match="target question.*A-D choices"):
        context_demo._validate_generated_few_shot_case(case)


def test_generated_few_shot_case_accepts_complete_choices_and_answers() -> None:
    assert context_demo._validate_generated_few_shot_case(valid_case())["target_answer"] == "B"


def test_generated_few_shot_case_rejects_demo_without_answer() -> None:
    case = valid_case()
    chunks = list(case["context_chunks"])
    chunks[0] = chunks[0].replace("\nAnswer: A", "")
    case["context_chunks"] = chunks
    with pytest.raises(ValueError, match="demonstration 1.*answer"):
        context_demo._validate_generated_few_shot_case(case)
```

- [ ] **Step 2: Verify RED**

Run `uv run pytest tests/shapiq/tests_unit/test_context_attribution_demo.py -v`.

Expected: malformed target and demonstration tests fail because current validation accepts them; valid case passes.

- [ ] **Step 3: Add minimal format validation**

Add before the few-shot validator:

```python
CHOICE_LINE_PATTERN = re.compile(r"(?m)^\s*([A-D])\.\s+\S.*$")
ANSWER_LINE_PATTERN = re.compile(r"(?im)^\s*Answer:\s*([A-D])\s*$")


def _validate_multiple_choice_text(
    text: str, *, description: str, require_answer: bool
) -> None:
    choice_labels = CHOICE_LINE_PATTERN.findall(text)
    if choice_labels != ["A", "B", "C", "D"]:
        msg = f"Generated few-shot {description} must contain exactly one ordered set of A-D choices."
        raise ValueError(msg)
    answer_labels = ANSWER_LINE_PATTERN.findall(text)
    if require_answer and len(answer_labels) != 1:
        msg = f"Generated few-shot {description} must contain exactly one A-D answer."
        raise ValueError(msg)
    if not require_answer and answer_labels:
        msg = f"Generated few-shot {description} must not contain an answer."
        raise ValueError(msg)
```

Call it once for the target with `require_answer=False`, then for each demonstration with `require_answer=True`.

- [ ] **Step 4: Verify GREEN**

Run `uv run pytest tests/shapiq/tests_unit/test_context_attribution_demo.py -v`.

Expected: 3 passed.

### Task 2: Retry invalid Gemma generations

**Files:**
- Modify: `demo/context_attribution.py:304,716-747`
- Test: `tests/shapiq/tests_unit/test_context_attribution_demo.py`

- [ ] **Step 1: Write failing retry tests**

Add lightweight `FakeInputs`, `FakeTokenizer`, and `FakeModel` objects. The tokenizer records chat prompts and returns queued decoded strings; the model counts calls and appends one fake generated token. Add:

```python
def test_generation_retries_invalid_case_then_returns_valid_case() -> None:
    invalid = valid_case()
    invalid["question"] = "Question: Target?"
    tokenizer = FakeTokenizer([json.dumps(invalid), json.dumps(valid_case())])
    model = FakeModel()

    result = context_demo.generate_case_with_gemma(
        tokenizer, model, "create case", few_shot=True
    )

    assert result["target_answer"] == "B"
    assert model.generate_calls == 2
    assert "A-D choices" in tokenizer.prompts[1]


def test_generation_reports_all_failures_after_three_attempts() -> None:
    tokenizer = FakeTokenizer(["not json"] * 3)
    model = FakeModel()

    with pytest.raises(ValueError, match="failed after 3 attempts") as error:
        context_demo.generate_case_with_gemma(
            tokenizer, model, "create case", few_shot=True
        )

    assert model.generate_calls == 3
    assert str(error.value).count("Could not find a JSON object") == 3
```

- [ ] **Step 2: Verify RED**

Run `uv run pytest tests/shapiq/tests_unit/test_context_attribution_demo.py -k generation -v`.

Expected: both fail because generation currently attempts only once.

- [ ] **Step 3: Implement bounded retry**

Add `GENERATED_CASE_MAX_ATTEMPTS = 3`. In `generate_case_with_gemma`, loop over three attempts. On each attempt, format and tokenize the current prompt, generate, decode, extract JSON, and validate. Catch only `ValueError`, record `Attempt N: <reason>`, and construct the next prompt as:

```python
attempt_prompt = (
    f"{prompt}\n\nYour previous response was invalid: {error}\n"
    "Return corrected JSON only and satisfy every hard constraint."
)
```

Return immediately after successful validation. After the loop, raise:

```python
details = "\n".join(failures)
msg = (
    f"Gemma case generation failed after {GENERATED_CASE_MAX_ATTEMPTS} attempts:\n"
    f"{details}"
)
raise ValueError(msg)
```

Runtime errors outside JSON extraction/validation must propagate without retry.

- [ ] **Step 4: Verify GREEN**

Run `uv run pytest tests/shapiq/tests_unit/test_context_attribution_demo.py -v`.

Expected: 5 passed.

### Task 3: Verify the focused change

**Files:**
- Modify: `demo/context_attribution.py`
- Test: `tests/shapiq/tests_unit/test_context_attribution_demo.py`

- [ ] **Step 1: Run focused formatting and lint**

Run:

```bash
uv run ruff format --check demo/context_attribution.py tests/shapiq/tests_unit/test_context_attribution_demo.py
uv run ruff check demo/context_attribution.py tests/shapiq/tests_unit/test_context_attribution_demo.py
```

Expected: both exit 0. If needed, format only these two files and rerun.

- [ ] **Step 2: Run required project pre-commit**

Run exactly from the project root: `uv run pre-commit run --all-files`.

Expected: all hooks pass. Report unrelated dirty-worktree failures rather than changing unrelated files.

- [ ] **Step 3: Review the scoped diff**

Run:

```bash
git diff --check -- demo/context_attribution.py tests/shapiq/tests_unit/test_context_attribution_demo.py
git diff -- demo/context_attribution.py tests/shapiq/tests_unit/test_context_attribution_demo.py
```

Expected: no whitespace errors and only the planned validation, retry, and tests beyond the user's pre-existing demo edits.

- [ ] **Step 4: Report without committing implementation**

Do not commit implementation unless explicitly requested. Report exact verification results and changed files.

