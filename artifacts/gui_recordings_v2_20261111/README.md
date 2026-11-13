# Successful native GUI recordings

All three clips start before pressing Start, show fresh structure generation, takeoff and assembly, and end with the GUI success result. They are continuous captures, without time compression or spliced phases. These are selected demonstration successes, not a new benchmark success rate.

| Task | Video | Modules / contacts | Initial preparation | Maximum command time |
|---|---|---|---|---|
| Ellipsoid grasp and transport | [Full recording](grasp_gpu/gui_full.mp4) | 4 / 4 | 13.923759 s | 5.964443 ms |
| Valve, 24 Nm, 180 degrees | [Full recording](valve_reserved/gui_full.mp4) | 3 / 2 | 12.028731 s | 1.836541 ms |
| Walking, 0.5 m/s, 5.001276 m | [Full recording](walking_reserved/gui_full.mp4) | 6 / 4 feet | 14.552108 s | 2.550685 ms |

No command exceeded 10 ms in these three runs. Each case contains requested_task.json, recording.json and the run result. summary.json indexes inputs and results. All captures are 1440×900 at 60 fps. Measured application drawing averages 62.5 fps. Drawing and encoding rates do not imply that physical state changes 60 times per wall-clock second. Simulation is slower than wall-clock execution; the videos retain the actual GUI pacing. Control computation delays remain reflected in physics.

GPU rendering uses VirtualGL/EGL with RTX4090; capture uses NVENC. Display-only batching preserves all mesh triangles; see display_audit.json. Physics geometry and success criteria were not relaxed. The grasp video was captured before the later substep-state publication and allocation-reuse improvements, so its displayed physical state updates can be less frequent despite the same capture rate.

The other case directories contain development failures or early stops; they are not successes. Problems were checked during execution rather than waiting for the wall-clock timeout. A separate remote GPU environment was prepared and tried, but the remote valve planning exceeded its deadline. All three delivered recordings are local.
