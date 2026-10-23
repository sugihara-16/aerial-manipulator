# Retained evaluation evidence

This is one unchanged implementation on 120 development inputs, not an unseen holdout.
The results are 105/120 successes. Source manifests and execution settings match across all cases.

Tracked records include the panel and seeds, reports, per-case requests and results,
generated plans, source manifests, and compact selection evidence. Each selection
record includes the SHA-256 of the original local selection.json. Detailed per-start
rows, runtime source copies, binary checkpoints, and intermediate trials remain local
and are excluded from Git. The structure report retains their aggregate counts.

Recompute task scores with scripts/report_grasp_panel.py. The full structure report
requires the locally retained selection.json files. Control updates during contact
are 2.5 ms in simulated time; real-time 400 Hz is not achieved.
