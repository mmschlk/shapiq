# Gemma Few-Shot Generation Validation and Retry Design

## Problem

Gemma can return a syntactically valid JSON case whose target question omits the required
multiple-choice options. The current few-shot validator only checks that the question is
non-empty and that `target_answer` is an A-D letter, so the incomplete case reaches the UI.

## Desired behavior

Every generated few-shot case must have a target question with exactly one option for each
label A-D. Every demonstration must also have exactly one A-D option and one valid answer
letter. Invalid generations should be retried automatically up to three total attempts.

## Design

Add focused format validation for multiple-choice text. The validator will reject missing,
duplicated, or unexpected option labels. For demonstrations, it will additionally require one
`Answer: A`, `Answer: B`, `Answer: C`, or `Answer: D` line. The existing JSON schema, player
count, kind, and target-answer checks remain unchanged.

Wrap Gemma generation, JSON extraction, and case validation in an attempt loop with three total
attempts. After a failed attempt, append the validation error to the next generation prompt so
Gemma can correct the specific defect. Each retry performs a fresh model generation. If all
attempts fail, raise one error containing the failure reason for every attempt.

The retry behavior applies to generated retrieval and few-shot cases because malformed JSON or
schema violations can occur in either mode. The stricter multiple-choice checks apply only to
few-shot cases.

## Error handling

Only expected case-generation failures (`ValueError` from output extraction or validation) are
retried. Runtime failures such as CUDA errors are propagated immediately. After the final
invalid generation, the raised error clearly states that three attempts failed and preserves the
individual validation messages for diagnosis.

## Testing

Add regression tests that verify:

- a target question without A-D choices is rejected;
- a complete generated few-shot case is accepted;
- a demonstration without choices or a valid answer is rejected;
- generation retries after an invalid first response and accepts a valid second response;
- three invalid responses produce a final error containing the attempt failures.

Tests will use small tokenizer and model fakes so they exercise the real generation orchestration
without loading Gemma weights.

## Scope

No Streamlit layout changes, model changes, generated-case repair, or unrelated refactoring are
included. The UI will continue to render the validated `question` and separate target answer as
it does today.
