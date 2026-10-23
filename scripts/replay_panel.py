"""Watch saved structures execute again; no recording or new structure search."""
import argparse
from contextlib import redirect_stdout
import gc
import json
import os
from pathlib import Path
import tempfile
import time

import glfw
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / 'artifacts/runs/task_pose120_v7_20261023'
CONTROL_FLAGS = '''predictive_motion condensed_solve fast_solve minimize_input
    governed_motion distance_closing generated_feedback rigid_object_feedback
    momentum_feedback finite_slip_preload bounded_allocation hold_withdrawal_pose
    preload_floor supported_load_transfer observed_handoff inertial_input_updates
    nearby_handoff contact_coupled constrained_load interval_motion interval_acquisition
    constrained_motion staged_approach planned_withdrawal hybrid_withdrawal
    background_withdrawal finite_acquisition_support process_withdrawal normal_tracking
    internal_tracking planned_preload_floor'''.split()


def replay_arguments(case, scratch, preview_seconds=None):
    settings = json.loads((case / 'experiment.json').read_text())['execution_configuration']
    # endpoint.npz is saved at table height. The existing --goal entry accepts
    # the airborne reference and lowers its free bodies by 0.15 m internally.
    model = mujoco.MjModel.from_xml_path(str(case / 'generated_plan/scene.xml'))
    with np.load(case / 'endpoint.npz') as saved:
        q = saved['q'].copy()
    free = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    q[model.jnt_qposadr[free] + 2] += .15
    goal = scratch / 'air_goal.npz'
    np.savez(goal, q=q)
    args = ['--plan', str(case / 'generated_plan'), '--goal', str(goal),
            '--output', str(scratch / 'execution'), '--steps', str(settings['steps']),
            '--duration', str(preview_seconds or settings['duration'])]
    args.extend('--' + name.replace('_', '-') for name in CONTROL_FLAGS if settings[name])
    return args


class Window:
    """A single read-only MuJoCo renderer, reused across all case models."""
    def __init__(self):
        if not glfw.init():
            raise RuntimeError('画面を開けません。デスクトップの端末から実行してください。')
        self.window = glfw.create_window(1280, 900, 'MuJoCo - saved 120 trials', None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError('MuJoCoの表示ウィンドウを作成できません。')
        glfw.make_context_current(self.window)
        glfw.swap_interval(0)
        self.context = None
        self.scene = None
        self.model = None
        self.display_data = None
        self.physics_model = None
        self.full_model = False
        self.camera = mujoco.MjvCamera()
        self.options = mujoco.MjvOption()
        self.perturb = mujoco.MjvPerturb()
        self.skip = False
        self.next_frame = 0.
        self.label = ''
        self.cursor = (0., 0.)
        glfw.set_key_callback(self.window, self.key)
        glfw.set_cursor_pos_callback(self.window, self.mouse)
        glfw.set_scroll_callback(self.window, self.scroll)
        self.message('Loading saved trial...', 'N: next trial | Esc / Q: quit')

    def key(self, window, key, scancode, action, mods):
        if action != glfw.PRESS:
            return
        if key in (glfw.KEY_ESCAPE, glfw.KEY_Q):
            glfw.set_window_should_close(window, True)
        elif key == glfw.KEY_N:
            self.skip = True

    def mouse(self, window, x, y):
        dx, dy = x - self.cursor[0], y - self.cursor[1]
        self.cursor = (x, y)
        if self.model is None:
            return
        left = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
        right = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
        if left or right:
            height = max(1, glfw.get_window_size(window)[1])
            action = mujoco.mjtMouse.mjMOUSE_MOVE_V if right else mujoco.mjtMouse.mjMOUSE_ROTATE_V
            mujoco.mjv_moveCamera(self.model, action, dx / height, dy / height, self.camera)

    def scroll(self, window, x, y):
        if self.model is not None:
            mujoco.mjv_moveCamera(self.model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0., -.05 * y, self.camera)

    @property
    def closed(self):
        return bool(glfw.window_should_close(self.window))

    def message(self, title, detail):
        glfw.set_window_title(self.window, title)
        if self.context is None:
            # A tiny display-only scene allows progress text before loading a trial.
            self.model = mujoco.MjModel.from_xml_string('<mujoco><worldbody/></mujoco>')
            self.scene = mujoco.MjvScene(self.model, maxgeom=100)
            self.context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)
        width, height = glfw.get_framebuffer_size(self.window)
        viewport = mujoco.MjrRect(0, 0, width, height)
        mujoco.mjr_render(viewport, self.scene, self.context)
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                           viewport, title, detail, self.context)
        glfw.swap_buffers(self.window)
        glfw.poll_events()

    def set_model(self, model):
        if self.context is not None:
            self.context.free()
        self.model = model
        self.display_data = mujoco.MjData(model)
        self.scene = mujoco.MjvScene(model, maxgeom=30000)
        self.context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)

    def begin(self, label, request, scene_path=None):
        self.label = label
        self.skip = False
        self.next_frame = 0.
        self.physics_model = None
        self.full_model = scene_path is not None
        self.fit_camera = True
        positions = np.array([request['source'][:3], request['destination'][:3]])
        self.travel = positions[1] - positions[0]
        self.camera.lookat[:] = positions.mean(axis=0) + [0., 0., .1]
        self.camera.distance = max(2.2, 2. * np.linalg.norm(positions[1] - positions[0]) + 1.)
        self.camera.azimuth = 135.
        self.camera.elevation = -30.
        self.message(label, 'Preparing saved structure... | N: next | Esc / Q: quit')
        if scene_path is not None:
            self.set_model(mujoco.MjModel.from_xml_path(str(scene_path)))

    def __call__(self, model, data, phase):
        glfw.poll_events()
        if self.closed or self.skip:
            return False
        now = time.monotonic()
        if now < self.next_frame:
            return True
        if self.physics_model is not model:
            if not self.full_model:
                self.set_model(model)
            # Collision/frame pruning preserves the complete joint state. Check
            # that contract before copying it into the full display-only model.
            if (self.model.nq != model.nq or self.model.nv != model.nv or
                    self.model.njnt != model.njnt or
                    any(self.model.joint(i).name != model.joint(i).name for i in range(model.njnt))):
                raise ValueError('display and physics joint layouts differ')
            for field in ('jnt_qposadr', 'jnt_dofadr', 'jnt_type'):
                if not np.array_equal(getattr(self.model, field), getattr(model, field)):
                    raise ValueError('display and physics joint layouts differ: ' + field)
            self.physics_model = model
        self.display_data.qpos[:] = data.qpos
        self.display_data.qvel[:] = data.qvel
        self.display_data.time = data.time
        # Position transforms only: never step the full display model or run its
        # much larger collision calculation. All parts follow the real joints.
        mujoco.mj_kinematics(self.model, self.display_data)
        mujoco.mj_camlight(self.model, self.display_data)
        if self.fit_camera:
            ids = [i for i in range(self.model.ngeom)
                   if self.model.geom(i).name.startswith('m') or self.model.geom(i).name == 'object']
            if ids:
                positions = self.display_data.geom_xpos[ids]
                positions = np.vstack((positions, positions + self.travel))
                self.camera.lookat[:] = (positions.min(axis=0) + positions.max(axis=0)) / 2
                self.camera.distance = max(2.2, 2. * (np.max(np.ptp(positions, axis=0)) + .3))
            self.fit_camera = False
        mujoco.mjv_updateScene(self.model, self.display_data, self.options, self.perturb, self.camera,
                              mujoco.mjtCatBit.mjCAT_ALL, self.scene)
        self.message(self.label, f'{phase} | {data.time:.2f} s | N: next | Esc / Q: quit')
        self.next_frame = time.monotonic() + 1. / 30.
        return True

    def finish(self, status):
        self.message(self.label, status)
        until = time.monotonic() + 1.5
        while time.monotonic() < until and not self.closed and not self.skip:
            glfw.poll_events()
            time.sleep(.02)

    def close(self):
        if self.context is not None:
            self.context.free()
        glfw.destroy_window(self.window)
        glfw.terminate()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--start', type=int, default=1, help='first trial number (1–120)')
    parser.add_argument('--limit', type=int, help='number of trials; default: all remaining')
    parser.add_argument('--preview-seconds', type=float, help='short display check; not a success evaluation')
    args = parser.parse_args()
    if args.start < 1 or (args.limit is not None and args.limit < 1):
        parser.error('--start and --limit must be positive')
    if args.preview_seconds is not None and args.preview_seconds < .01:
        parser.error('--preview-seconds must be at least 0.01')
    run = args.run.resolve()
    cases = json.loads((run / 'panel.json').read_text())['cases']
    if args.start > len(cases):
        parser.error('--start exceeds the number of saved trials')
    selected = cases[args.start - 1:][:args.limit]
    for item in selected:
        for name in ['generated_plan/plan.json', 'generated_plan/request.json',
                     'generated_plan/scene.xml', 'endpoint.npz', 'experiment.json']:
            path = run / item['id'] / name
            if not path.is_file():
                parser.error(f'保存データがありません: {path}')
    # Existing controller uses paths relative to this repository.
    os.chdir(ROOT)
    window = Window()
    try:
        from build.check_local_task import main as execute
        print('保存構造を順次再実行します。N: 次へ / Esc・Q・画面を閉じる: 終了。録画なし。', flush=True)
        for number, item in enumerate(selected, args.start):
            case = run / item['id']
            label = f'{number}/{len(cases)} - {item["id"]}'
            print(label, flush=True)
            window.begin(label, item, case / 'generated_plan/scene.xml')
            if window.closed:
                break
            # The evaluator's scratch files are automatically discarded; original
            # results remain untouched. Do not retain new logs, videos or results.
            with tempfile.TemporaryDirectory(prefix='amsrr-view-') as temporary:
                scratch = Path(temporary)
                command = replay_arguments(case, scratch, args.preview_seconds)
                with open(os.devnull, 'w') as quiet, redirect_stdout(quiet):
                    execute(command, observer=window)
                result = json.loads((scratch / 'execution/result.json').read_text())
            if window.closed:
                break
            status = ('SKIPPED' if window.skip else
                      ('PREVIEW' if args.preview_seconds else
                       ('SUCCESS' if result['task_success'] else 'FAILED')))
            print(f'  {status}: {result["reason"]}', flush=True)
            window.finish(f'{status}: {result["reason"]}')
            gc.collect()
    except KeyboardInterrupt:
        print('\n再実行を終了しました。', flush=True)
    finally:
        window.close()


if __name__ == '__main__':
    main()
