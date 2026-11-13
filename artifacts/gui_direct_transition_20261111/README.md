# Simultaneous flight and morphology transition

All three requested inputs are byte-equivalent as parsed JSON to the previous successful recordings. Each final run performs fresh structure generation, individual takeoff and assembly, then the task in one physical world. All three succeed under the existing planning/control timing and physical task checks. These three representative runs do not update the prior full benchmark success rates.

| Task | Modules | Transition before | Transition now | Initial preparation | Maximum command time |
|---|---:|---:|---:|---:|---:|
| grasp_clear | 4 | 37.04 s | 19.78 s | 8.586 s | 1.504 ms |
| valve_clear | 3 | 28.87 s | 15.22 s | 9.035 s | 1.470 ms |
| walking_clear | 6 | 32.25 s | 15.52 s | 13.958 s | 2.189 ms |

Transition values are simulated trajectory durations after assembly: flight transfer for grasp/valve, flight transfer plus descent and ground-load transfer for walking. Measured preparation between stages is additionally reflected in physics under the previous command; it is not included in this trajectory-only comparison. No command exceeded 10 ms. Walking completed 5.00145 m at the requested 0.5 m/s; valve used exactly two contacts and completed 180 degrees.

## Changes

- Replace fixed ascent, morph, horizontal transit, descent and 3-second hold with overlapping translation and articulation. Height stays between the actual start and target heights; there is no fixed added ascent or intermediate stationary waypoint.
- Sample 41 configurations using actual collision geometry. If a descent sweeps an arm through an object, compare monotone timing curves that defer descent while translation/articulation continue. No physical rollout or task-success test is used to select this path. The sample check does not guarantee continuous or real-world clearance. For grasp, zero delay was rejected and a 0.2-duration descent delay passed. Valve and walking use zero delay.
- In integrated walking, fly directly to the airborne landing posture, omitting the old 0.8-second hold and 3-second morphology continuation. Preserve the screened 4.5-second descent and 3-second ground-load transfer. Physical position, velocity and time are never reset. Actual entry clearance is measured during the first control period instead of a removed hold.
- Keep physical limits, control laws, delay modeling, task inputs and success criteria. The preserved landing suffix and common GUI contracts pass 50 tests.

## Recordings

- [Ellipsoid, 4 modules](grasp_clear/gui_full.mp4)
- [Valve, 3 modules and 2 contacts](valve_clear/gui_full.mp4)
- [Walking, 6 modules at 0.5 m/s](walking_clear/gui_full.mp4)

All clips show candidate generation through success at 1440×900, 60 fps, with cursor hidden both in Qt and capture. Measured display rate is about 62.5 fps. No speed changes or phase splicing. Simulation advances more slowly than wall time; capture fps is not a claim of real-time physics. Inputs, full results, capture timing and phase telemetry are retained per case; summary.json indexes the final runs.

## Development evidence

The first direct grasp path crossed the ellipsoid during articulation and failed with extreme control instability. An independent geometry check reproduced the overlap; descent timing resolved it without reintroducing added ascent or fixed waits. The failure remains in grasp_direct and is not counted as a success. walking_direct was stopped to fix the entry-clearance audit before completing; walking_direct2 succeeded before the geometry screen was added. walking_clear reran the final implementation successfully. No final failures were relabeled or removed from their records.

## Requested walking-speed revision

[Walking at 0.4 m/s, 6 modules](walking_speed04/gui_full.mp4) replaces the walking demonstration for presentation. Only the input speed changed; all other task inputs match walking_clear. Fresh generation, takeoff, assembly, landing and 5 m walking succeeded: 5.003154 m, measured walking mean 0.395506 m/s, preparation 13.915125 s, maximum command time 2.105140 ms and no 10 ms misses. Cursor remains hidden; capture remains 1440×900/60 fps. See walking_speed04/summary.json. The earlier 0.5 m/s result is retained.
