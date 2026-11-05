# Airborne start, landing and walking — 2026-11-05

The final implementation completed 33/36 tasks (91.7%) from fresh structure generation, with a 39.9287% mean walking rotor proxy ratio and a 14.5279 s maximum initial preparation time. The runtime was fixed throughout `final36_v13`.

## Evaluation contract

- Fixed module counts 3–8, speeds 0.22/0.26/0.30 m/s, straight and left/right curved paths; 5 m supported walking.
- The 36 task inputs and generation seeds are in `final36_v13/panel.json` and `configs/evaluation/walking_airborne36.json`.
- Spawn with the whole collision geometry at least 0.6 m above the floor, then hold, morph, descend, transfer load and walk. No physical state or clock reset between stages.
- Measured control computation delays advance the simulation with the preceding command. Numerical integration wall time is not treated as controller latency.
- Walking success, physical constraints and initial 15 s acceptance limits remain unchanged. The <40% target applies to the arithmetic mean of successful walking cases, not to each count or the entire sequence.
- Consumption is an uncalibrated rotor thrust-to-the-power-1.5 integral, not measured electrical energy. Joint burden and positive mechanical work are reported separately.

## Primary results

`final36_v13/audit_summary.json` is the independently recomputed task, timing and energy result. `sequence_audit.json` checks the airborne initial state, continuous clock and disjoint interval integrals. Both have no issues. `engineering_review.json` records the remaining physical failures and module-specific results. `source_scope.json` checks the runtime and unchanged physical inputs.

All 35 physically executed plans passed landing acquisition. One task had no plan. Two completed 5 m but failed walking requirements: n5_v26_right had too few valid steps on one foot and excess support outside the feet; n8_v30_straight had excess support outside the feet. These remain failures.

The successful whole-sequence rotor proxy ratio is 61.1406% against hover over the same whole duration. Against hover over only the walking duration it is 97.4241%. The aggregate whole integral is 2.41935 times the aggregate walking integral. These are different denominators and weights.

Command publication: 108,763 calls, maximum 6.4576 ms, zero above 10 ms. Observation-to-new-command application: maximum 17 ms, explicitly applied to the physics. This is a PC measurement, not a hard real-time or embedded-board guarantee.

## Additional checks, kept separate

- `halfstep6_v13`: the six predeclared saved-plan tests at 0.25 ms all retained their outcomes from the primary 0.5 ms run.
- `halfstep36_v13`: these six plus the remaining 29 plans, the latter run four at a time. The original no-plan failure is retained without physical replay. Result: 31/36, mean successful ratio 39.0761%. The two new failures are 8-module straight paths; their background computation delays also increased. The lower mean includes the loss of two high-consumption successes, so it is not evidence of improved efficiency.
- `step_isolation_v13`: those two plans executed sequentially at 0.25 ms using the primary CPU configuration. Both succeeded with path errors and proxy ratios close to the primary run. See `comparison.json`. These successes do not overwrite the parallel failures and are not added to the primary score.
- The additional test changes integration timestep and execution load together. It establishes neither timestep-only causation nor strict timestep independence. Sensitivity to concurrent computation remains a limitation.
- `tests_v13.log`: 92 checks passed. `closure_checks.json`: all 1,592 runtime hashes still match and document links exist.

## Reproduction

Run from the repository root with the recorded dependencies and compiled native components. `final36_v13/environment.json` records the machine and libraries; `sources.json` records source and binary hashes. `validation_primary.tar.xz` contains the primary raw records and source snapshot; `validation_sensitivity.tar.xz` contains the full sensitivity and the two diagnostic replays. `validation_manifest.json` lists every archived file and its SHA-256, checked against all 2,469 decompressed files (2,024 primary and 445 sensitivity/diagnostic records). The first six replay directories are identical copies within the full sensitivity archive, so their raw data are not stored twice. Videos, summaries and the six-case comparison metadata are tracked separately. Earlier development runs and rejected implementations are not part of these archives. The source snapshot records what was present at execution, including generated geometry and auxiliary files; it is provenance, not a list of adopted runtime components.

The stored models contain `/home/leus/amsrr_det` paths. Restore from that repository location without overwriting existing files:

```bash
tar --skip-old-files -xJf artifacts/walking_landing_20261105/validation_primary.tar.xz
tar --skip-old-files -xJf artifacts/walking_landing_20261105/validation_sensitivity.tar.xz
mkdir -p build
cp -an artifacts/walking_landing_20261105/final36_v13/source_snapshot/build/collision_physx_docks build/
OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m scripts.report_walking_panel artifacts/walking_landing_20261105/final36_v13
OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python artifacts/walking_landing_20261105/audit_sequence.py artifacts/walking_landing_20261105/final36_v13
```

These commands recompute reports from saved records without rerunning the physical task. For a new physical experiment:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m scripts.run_walking_panel artifacts/new_airborne36 --panel configs/evaluation/walking_airborne36.json
OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m scripts.report_walking_panel artifacts/new_airborne36

# A saved-plan sensitivity run; not a new planning-time measurement.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m scripts.run_walking_task --plan artifacts/walking_landing_20261105/final36_v13/n8_v26_straight/selection.json --out artifacts/new_halfstep_case --timestep .00025 --control-cpu 0 --worker-cpu 2
```

Videos derived from actual saved states are in the primary n3_v26_straight, n4_v22_straight, n6_v26_left and n8_v30_right directories. The fixed development panel is not a guarantee for unseen tasks or seeds, actual hardware, or walking with five or more feet.
