#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
exec env -u LD_PRELOAD PYTHONPATH="$project_dir:$project_dir/build/body_bounds/python" \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=glfw \
  "$project_dir/.venv/bin/python" -m scripts.replay_panel "$@"
