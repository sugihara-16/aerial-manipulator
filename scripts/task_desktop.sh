#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=.
# Qt owns the desktop OpenGL context; do not inherit an offscreen backend.
unset MUJOCO_GL PYOPENGL_PLATFORM
if ! .venv/bin/python -c 'import PySide6.QtOpenGL' >/dev/null 2>&1; then
    echo 'Install the desktop UI: .venv/bin/python -m pip install -r requirements-gui.txt' >&2
    exit 1
fi
exec .venv/bin/python -m scripts.task_desktop "$@"
