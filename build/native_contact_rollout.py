"""Diagnostic C loop for the same MuJoCo steps, with per-step load outputs.

The installed MuJoCo library and headers must match exactly. Compilation is
outside control timing. A separate MjData per worker is mandatory. This is
not a controller and does not initialize from actual simulation internals.
"""
import ctypes
import hashlib
import subprocess
from pathlib import Path
import numpy as np
import mujoco


class ContactRollout:
    def __init__(self, model, links):
        self.model = model
        root = Path(mujoco.__file__).parent
        source = Path(__file__).with_name('contact_rollout.cc')
        version = mujoco.mj_versionString()
        digest = hashlib.sha256(source.read_bytes() + version.encode()).hexdigest()[:16]
        binary = source.with_name('contact_rollout_' + digest + '.so')
        if not binary.exists():
            subprocess.run(['c++', '-O3', '-std=c++17', '-shared', '-fPIC',
                '-Wall', '-Wextra', '-I' + str(root / 'include'), str(source),
                str(root / ('libmujoco.so.' + version)), '-Wl,-rpath,' + str(root),
                '-o', str(binary)], check=True)
        self.library = ctypes.CDLL(str(binary))
        if self.library.rollout_version() != mujoco.mj_version():
            raise RuntimeError('MuJoCo header/library version mismatch')
        self.call = self.library.contact_rollout
        self.call.restype = ctypes.c_int
        pointer = ctypes.c_void_p
        self.call.argtypes = [pointer, pointer, ctypes.c_int, ctypes.c_int,
                              ctypes.c_int, pointer, pointer, pointer, pointer, pointer, pointer, pointer]
        names = [model.geom(i).name for i in range(model.ngeom)]
        self.groups = np.array([next((j for j, link in enumerate(links)
            if name.startswith(link + '/')), -1) for name in names], dtype=np.int32)
        self.bodies = np.array([model.body(name).id for name in links], dtype=np.int32)
        self.object = model.geom('object').id
        self.pads = len(links)

    def advance(self, data, steps, *, include_penetration=False, include_poses=False):
        m = self.model
        if data.model._address != m._address:
            raise ValueError('data/model mismatch')
        if steps < 1:
            raise ValueError('positive step count required')
        states = np.empty((steps, 1 + m.nq + m.nv))
        loads = np.empty((steps, self.pads, 7))
        motors = np.empty((steps, 2, m.nu))
        penetration = np.empty(steps)
        poses = np.empty((steps, self.pads + 1, 7)) if include_poses else None
        error = self.call(m._address, data._address, steps, self.object, self.pads,
            self.groups.ctypes.data, self.bodies.ctypes.data,
            states.ctypes.data, loads.ctypes.data, motors.ctypes.data, penetration.ctypes.data,
            None if poses is None else poses.ctypes.data)
        if error or not all(np.isfinite(x).all() for x in [states, loads, motors, penetration]):
            raise RuntimeError(f'contact rollout failed: {error}')
        if include_poses:
            if not np.isfinite(poses).all():raise RuntimeError("nonfinite forecast body poses")
            return states, loads, motors, penetration, poses
        if include_penetration:
            return states, loads, motors, penetration
        return states, loads, motors
