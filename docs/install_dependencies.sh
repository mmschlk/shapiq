#!/usr/bin/env bash
# Install what the docs build needs into the project's environment (.venv, created with `uv venv`).
# Read the Docs (.readthedocs.yaml) and the docs workflow (.github/workflows/docs.yml) both run it,
# so that CI checks the docs in the environment that Read the Docs builds them in.
#
# The examples run on the CPU: the locked packages of the docs group and all extras, without the
# CUDA libraries of torch (nvidia-*, triton; about 4 GB, most of the install time), and torch from
# its CPU index. --no-deps installs exactly that list (XGBoost would pull in NCCL, a GPU library).
set -euo pipefail

locked="$(mktemp)"
requirements="$(mktemp)"
uv export --frozen --no-dev --all-extras --group docs --no-emit-project --no-hashes --no-annotate \
    --no-header --output-file "$locked"
grep -vE '^(nvidia-|triton==)' "$locked" > "$requirements"
uv pip install --torch-backend cpu --no-deps -r "$requirements"
uv pip install --no-deps -e .
rm -f "$locked" "$requirements"
