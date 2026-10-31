#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
uv run ruff check distill tests
uv run ruff format --check distill tests
uv run mypy distill tests
