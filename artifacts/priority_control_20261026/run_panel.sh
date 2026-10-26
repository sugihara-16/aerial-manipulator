#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
output=${1:?Usage: $0 NEW_OUTPUT_DIRECTORY [REMOTE_CONFIG_JSON]}
remote_args=()
if [[ -n "${2:-}" ]]; then remote_args=(--structure-remote "$2"); fi
exec env -u LD_PRELOAD PYTHONPATH=.:build/body_bounds/python OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
.venv/bin/python artifacts/span_load30_20261024/run_panel.py \
--panel artifacts/runs/span_load30_inputs_20261024.json \
--output "$output" --workers 3 --defer-execution --keep-going \
--execution-timeout 600 --planning-watchdog 30 \
--structure-workers 32 "${remote_args[@]}" --structure-generator task --trial-seed 2026102404 --steps 80 \
--generated-feedback --momentum-feedback --finite-slip-preload --minimize-input \
--global-structure-search --governed-motion --background-withdrawal \
--bounded-allocation --hold-withdrawal-pose --preload-floor --process-withdrawal \
--task-priority-control --observed-handoff --inertial-input-updates --nearby-handoff --contact-coupled
