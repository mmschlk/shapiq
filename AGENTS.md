# AGENTS.md

This role of this file is to describe common mistakes and confusion points that agents might encounter as they work in this project. If you ever encounter something in the project that surprises you, please alert the developer working with you and indicate that this is the case in the Agent.md file to help prevent future agents from having the same issue.

## Commands to interact with the codebase which you should run:

## Conversion package note

Boosting converters live in separate modules such as `xgboost.py`,
`lightgbm.py`, and `catboost.py`, and are hooked up through lazy registrations in
`src/shapiq/tree/conversion/__init__.py`.

`treelite.py` is the odd one out: it registers *scikit-learn* classes
(`GradientBoosting*`, `HistGradientBoosting*`), so it is imported eagerly next to
`sklearn.py` rather than through a `delayed_register` on a class-path string. The
optional `treelite` dependency it needs is imported inside the conversion function,
not at module import time — so importing shapiq never requires treelite.

Note that the module is named `treelite.py` inside `shapiq.tree.conversion` and still
does `import treelite` to reach the third-party package; this is an absolute import
and resolves correctly (same pattern as `lightgbm.py` and `xgboost.py`).

### Build Docs (only use this command verbatim from the project root)

```bash
rm -rf docs/source/generated docs/source/auto_examples && uv run sphinx-build -b html docs/source docs/build/html
```

### Run Pre-commit (takes only 3s)

```bash
uv run pre-commit run --all-files
```
