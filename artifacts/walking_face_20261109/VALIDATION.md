# Walking Dock alignment validation

**Complete: the fixed-panel targets passed, and representative timestep/visual checks found no conflicting result.**

`full36_599` uses one frozen implementation and one fresh generation seed drawn before outcomes for each of the unchanged 36 tasks. The sequence is airborne spawn, morph, land and walk 5 m. Modules are fixed at 3–8, speeds at 0.22/0.26/0.30 m/s, with straight and radius-3.33 m left/right curves.

Female means pitch Dock; male means yaw Dock, as specified by the user. Female support faces are constrained to 1 degree in nominal planning, and checked at every physical step against 5 degrees when a foot carries more than 2% of assembly weight. Off-face impulse is limited to 1% of foot impulse. Male feet may tilt. The same role mask is used by generation, gait construction, control and physical audit. Real CAD, friction, motor limits, original distance/speed/ordinary-step criteria and failure counts are unchanged.

## Results

| Modules | Success | Walking hover proxy, successful cases | Whole-sequence hover proxy, successful cases |
|---|---:|---:|---:|
| 3 | 6/6 | 42.37% | 63.54% |
| 4 | 6/6 | 40.29% | 63.70% |
| 5 | 5/6 | 24.47% | 52.64% |
| 6 | 5/6 | 30.68% | 57.85% |
| 7 | 6/6 | 42.07% | 64.51% |
| 8 | 5/6 | 45.51% | 66.39% |

Overall **33/36 (91.7%)**. The arithmetic mean of successful walking ratios is **37.932629%**, below the 40% target. The whole-sequence mean is **61.661591%**. Summed whole-sequence accumulated rotor proxy is **2.550890 times** the sum over walking alone. These are uncalibrated thrust-derived proxies, not electrical energy or a measured hardware comparison. Individual cases and module groups need not be below 40%.

All 36 completed all 59 distinct basic generation/improvement trials. Initial time includes new structure generation, gait/direction refinement and execution preparation: maximum **14.500674 s**, mean **13.107447 s**. No planning deadline truncation was accepted.

Command issuance: **105,689 calls**, maximum **5.326787 ms**, zero over 10 ms; all observation intervals were within 10 ms. Observation-to-application latency is a different quantity: maximum **17.5 ms**, with 403 allocations at or above 10 ms. Physics continues with the previous command during measured computation delay. This is PC simulation evidence, not VIM4, Wi-Fi, hardware or an OS hard-real-time guarantee.

The generated plans use 2–4 feet. Female feet occur in three executed cases (5, 6 and 8 modules); all three pass the face criteria. Two complete the task; the 6-module case fails the route endpoint only. Maximum loaded female angle across all three is **1.360273 degrees**, and off-face impulse fraction is zero. Among successful cases the maximum loaded angle is 0.951516 degrees and the maximum landing angle is 0.201902 degrees. Male tilt is permitted and is not counted as a violation.

## Remaining failures and concentration review

- `n5_v30_straight`: no accepted candidate after all 59 trials.
- `n6_v26_straight`: 5.00109 m traveled, but endpoint route error 5.05094 cm exceeds 5 cm; maximum route error is 6.04150 cm. All other task checks, face alignment and timing pass.
- `n8_v30_right`: no accepted candidate after all 59 trials.

These failures remain in the denominator. Speeds 0.22/0.26/0.30 m/s have 12/12, 11/12 and 10/12 successes; straight/left/right have 16/18, 9/9 and 8/9. The predeclared 30 one-sided Fisher comparisons with Holm correction do not detect concentration. This does not prove equal failure rates.

The earlier development panel 592 had 3/6 successes on high-speed curves versus 30/30 elsewhere, and separate diagnostic 593 repeated 3-module route and 7-module generation failures. They motivated the changes and are not pooled or replaced. Current 599 has **5/6 high-speed curves versus 28/30 elsewhere**; the recurring 3-module and 7-module cases both succeed. The current distribution and mechanisms do not show the earlier clear recurring concentration. Finite sampling, remaining candidate failures, and the endpoint miss remain limitations.

## Adopted changes

- Task-dependent load/stroke port growth is retained. Articulated branches start from either transverse root direction using the sign of the existing heading draw; the final topology and posture are still optimized and checked. This is an empirical initialization, not a completed-structure catalogue.
- Static posture energy is weighted by support residence time, and periodic rotor-direction optimization retains continuous angle/rate bounds. Sparse local spline coordinates, redundant-row removal, scheduling, idle-core phase checks and valid common part-dimension caching reduce work without omitting the fixed comparison or acceptance checks.
- The common constrained acceleration/input allocation retains the real actuator bounds. Female foot orientation, actual material-point slip, articulated shape and whole-body route tracking enter the same allocation. Bounded accumulated corrections address lasting errors. COM gains 49/14 and the selected weights are empirical control settings, not a stability or anti-windup proof.
- No learning, complete-structure lookup, full-task rollout success selection, physical state reset on landing, or simulated reaction-force truth as a controller input is used.

## Independent checks and scope

`audit_summary.json` and `face_audit.json` contain no inconsistencies. The result audit independently reconstructs route/speed, actuator bounds, measured-delay mapping, proxy normalization and frozen source/input evidence. The face audit separately reconstructs loaded orientations from saved poses; full-rate force/angle evidence remains in the physics audit. Source snapshots and dependency/environment data accompany the panel.

98 walking/face/landing/cache tests passed after the final two-line adoption; 25 shared grasp derivative checks had passed on the unchanged shared code.

The seven available half-step replays all retained success. Publication maximum was **4.891518 ms**, with zero over 10 ms; maximum loaded female angle was **1.397456 degrees**. The independent half-step face audit also reported no inconsistencies. See `half599/comparison.json`, `half599/summary.json` and `half599/face_audit.json`. The six predefined indices remain [2, 7, 14, 21, 28, 35]. Index 35 has no accepted plan and is retained as unavailable, not replaced. The first executed female case is the predeclared feature addition. A separate post-evaluation feature check adds the first executed 8-module female case because the predefined 8-module case cannot be replayed. These checks are not a new 36-case score.

Pro244 reviewed the method changes and Pro245 confirmed that the stated fixed-panel completion judgment has no major logical inconsistency, subject to the now-completed representative checks. The statistical 59-trial explanation assumes a fixed independent sampling procedure and concerns one further trial improving the analytic cost. It is not a probability that no better unexplored structure exists, a simultaneous 36-task guarantee, or a guarantee of physical success/energy.

Saved physical states were rendered for [3-module biped curve](full36_599/n3_v30_left/motion.mp4), [4-module tripod straight](full36_599/n4_v22_straight/motion.mp4), [5-module female-foot curve](full36_599/n5_v26_right/motion.mp4) and [8-module female-foot curve](full36_599/n8_v26_left/motion.mp4). Sequence contact sheets and one-cycle frame strips were inspected for airborne descent, ground acquisition, articulated swing and forward stepping. Green feet are female; orange feet are male. Rendering changes colors and the checker floor only; it uses saved states without a new physical rollout. The underlying every-step/contact checks remain the numerical evidence.

`engineering_review.json` records ground support, actual steps, joint effort/work and mechanical stop reactions by module count. Actuator output stays within its limit; a mechanical joint stop reaction is a different quantity (maximum 66.50 Nm among successful cases) and has not been validated against hardware structural strength. No new stop/strength criterion was introduced or silently passed.

## Reproduce

```bash
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m scripts.run_walking_panel artifacts/walking_face_recheck --panel artifacts/walking_face_20261109/panel599.json
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m scripts.report_walking_panel artifacts/walking_face_recheck
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python artifacts/walking_face_20261109/audit_faces.py artifacts/walking_face_recheck
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python artifacts/walking_face_20261109/sequence_metrics.py artifacts/walking_face_recheck
```

Recorded seeds reproduce the fixed tasks/generation inputs; measured timing and delay-sensitive trajectories can vary with host load. Development results are not mixed into this report.

## Archived evidence

Final `full36_599` and `half599` raw records and the frozen source snapshot are stored in `validation_*.tar.xz`. Summaries and videos remain directly accessible. `validation_manifest.json` lists SHA-256 values for every archived file and archive; all files were verified after compression. Development implementations are excluded.

Restore from the repository root before inspecting raw records or replaying saved plans:

```bash
for archive in artifacts/walking_face_20261109/validation_*.tar.xz; do
    tar -xJf "$archive"
done
```
