#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "${1:-}" != "--web" ]]; then
    exec ./scripts/task_desktop.sh "$@"
fi
shift
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=.
exec .venv/bin/python -m scripts.task_gui "$@"
