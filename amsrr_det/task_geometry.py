"""Geometry-derived grasp frames, opening postures and transport references.

Uses observed poses and known shapes, never simulated future motion or measured
contact forces. The opening checks are sampled geometry and static hover checks,
not a certificate that the entire motion will succeed.
"""
import ctypes
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
import time

import mujoco
import numpy as np
from amsrr_det.linear_program import linprog


def box_face_mask(points,size,axis,normals=None,contact_distance=None):
    """Identify a box face by contact location, not a tilted pad's normal.

    Points and normals are in object coordinates. At an edge, two faces
    share the point: the contact normal identifies the supporting face.
    A 1 micrometre tie tolerance handles collision-coordinate roundoff.
    Compressed mesh contacts approximate the surface witnesses near edges.
    When distance is supplied, use a component-wise penetration envelope
    to collect nearby faces, then assign the dominant normal component.
    This is a tested edge-attribution convention, not an exact error bound
    for arbitrary collision geometry. Face-interior contacts must retain
    their identity under changes of depth (see the independent mesh tests).
    Without normals, ambiguous edges/corners are excluded.
    """
    magnitude=np.abs(np.asarray(points));half=np.asarray(size)/2
    distances=np.abs(magnitude-half)
    if normals is None:
        on_face=np.all(distances[...,axis,None]+1e-9<np.delete(distances,axis,axis=-1),axis=-1)
        tolerance=1e-9
    else:
        tolerance=1e-6
        if contact_distance is None:
            nearest=distances<=np.min(distances,axis=-1,keepdims=True)+tolerance
        else:
            tolerance=tolerance+np.maximum(-np.asarray(contact_distance),0.)[...,None]*np.abs(np.asarray(normals))
            nearest=distances<=tolerance
        scores=np.where(nearest,np.abs(np.asarray(normals)),-1.)
        on_face=(np.argmax(scores,axis=-1)==axis)&np.any(nearest,axis=-1)
    finite_tolerance=np.delete(tolerance,axis,axis=-1) if np.ndim(tolerance)>0 else tolerance
    finite=np.all(np.delete(magnitude-half,axis,axis=-1)<=finite_tolerance,axis=-1)
    return on_face & finite


@lru_cache(maxsize=1)
def library():
    root = Path(__file__).resolve().parents[1]
    source = root/'native/task_geometry.cpp'
    package = Path(mujoco.__file__).parent
    version = mujoco.mj_versionString()
    digest = hashlib.sha256(source.read_bytes()+version.encode()).hexdigest()[:16]
    binary = root/'build'/f'task_geometry_{digest}.so'
    if not binary.exists():
        fd, name = tempfile.mkstemp(prefix='task_geometry_', suffix='.so', dir=binary.parent)
        os.close(fd)
        try:
            subprocess.run(['c++', '-O3', '-std=c++17', '-shared', '-fPIC',
                            '-I'+str(package/'include'), str(source),
                            str(package/f'libmujoco.so.{version}'),
                            '-Wl,-rpath,'+str(package), '-o', name], check=True)
            os.replace(name, binary)
        finally:
            Path(name).unlink(missing_ok=True)
    lib = ctypes.CDLL(str(binary))
    pointer, integer = ctypes.c_void_p, ctypes.c_int
    lib.transport_frame.argtypes = [integer]+[pointer]*7
    lib.transport_frame.restype = None
    lib.opening_ik.argtypes = [pointer, pointer, integer, pointer, pointer,
                              pointer, pointer, integer, pointer, pointer,
                              integer, pointer, pointer, pointer]
    lib.opening_ik.restype = None
    return lib


class GraspFrame:
    """Follow an axisymmetric shape's axis without chasing its axial spin.

    Only the robot's geometric grasp reference uses this frame. The actual
    object rotation, velocity and full orientation task remain unchanged.
    """
    def __init__(self, shape, rotation):
        self.axis, self.spherical = None, False
        self.frame = rotation.copy()
        self.next = np.empty((3, 3))
        self.velocity, self.acceleration = np.empty(3), np.empty(3)
        if shape is not None:
            if shape.family == 'cylinder':
                self.axis = 2
            elif shape.family == 'ellipsoid':
                size = np.array(shape.size)
                equal = np.abs(size[:, None]-size[None, :]) < 1e-12*max(size)
                if equal.all():
                    self.spherical = True
                else:
                    for axis in range(3):
                        transverse = [j for j in range(3) if j != axis]
                        if equal[transverse[0], transverse[1]]:
                            self.axis = axis
        # Compile/load before the timed control loop, including worker setup.
        self.transport = library().transport_frame if self.active else None

    @property
    def active(self):
        return self.axis is not None or self.spherical

    def update(self, rotation, omega, alpha):
        if self.spherical:
            self.velocity[:] = 0.
            self.acceleration[:] = 0.
        else:
            self.transport(self.axis, self.frame.ctypes.data, rotation.ctypes.data,
                           omega.ctypes.data, alpha.ctypes.data, self.next.ctypes.data,
                           self.velocity.ctypes.data, self.acceleration.ctypes.data)
            self.frame[:] = self.next
        return self.frame, self.velocity, self.acceleration


def normal_opening(controller, q, v, distance=.04):
    """Four damped geometry steps opening the existing contact pads outward."""
    drive, load = controller.drive, controller.load
    model = drive.m
    drive.observe(q, v)
    contact = load.prepare(drive, np.zeros(6), squeeze=0., active_only=True)
    bodies = np.array([e[1] for e in load.elements], np.int32)
    local = np.array([e[3] for e in load.elements])
    start = drive.d.xpos[bodies]+np.einsum(
        'nij,nj->ni', drive.d.xmat[bodies].reshape(-1, 3, 3), local)
    normal = -contact['G'][:3, ::3].T
    target = np.ascontiguousarray(start+distance*normal)
    weights = np.ascontiguousarray(
        .25*np.eye(3)[None, :, :]+.75*np.einsum('ni,nj->nij', normal, normal))
    columns = np.r_[np.arange(6), drive.vi[drive.dock]].astype(np.int32)
    scales = np.r_[np.ones(3), np.full(3, 4.), np.full(np.count_nonzero(drive.dock), 5.)]
    qi = np.array(drive.qi, np.int32)
    ranges = np.ascontiguousarray(model.jnt_range[
        [model.joint(a['name']).id for a in drive.actuators]])
    delta = np.zeros(model.nv)
    library().opening_ik(
        model._address, drive.d._address, len(bodies), bodies.ctypes.data,
        local.ctypes.data, target.ctypes.data, weights.ctypes.data,
        len(columns), columns.ctypes.data, scales.ctypes.data, drive.nj,
        qi.ctypes.data, ranges.ctypes.data, delta.ctypes.data)
    return delta


def prepare_opening(controller, closed, original):
    """Prefer a much shorter opening with sampled clearance and bounded hover.

    The comparison is a quintic path's time scale at fixed root/joint velocities,
    not its actual dynamic execution time. A twofold reduction is an engineering
    threshold. Rejected proposals retain the generated original opening.
    """
    start = time.perf_counter()
    drive = controller.drive
    model = drive.m
    old = np.zeros(model.nv)
    mujoco.mj_differentiatePos(model, old, 1., closed, original)
    speeds = np.array([a['speed'] for a in drive.actuators])

    def travel(delta):
        return 1.875*max(np.linalg.norm(delta[:3])/.25,
                        np.linalg.norm(delta[3:6]),
                        np.max(abs(delta[drive.vi])/speeds))

    delta = normal_opening(controller, closed, np.zeros(model.nv))
    info = dict(original_travel_s=float(travel(old)),
                candidate_travel_s=float(travel(delta)), selected=False)

    def reject(reason):
        info.update(reason=reason, preparation_s=time.perf_counter()-start)
        return original.copy(), None, info

    if 2*travel(delta) >= travel(old):
        return reject('insufficient_travel_reduction')
    candidate = closed.copy()
    mujoco.mj_integratePos(model, candidate, delta, 1.)
    drive.observe(candidate, np.zeros(model.nv))
    contact = controller.load.prepare(drive, np.zeros(6), squeeze=0., active_only=False)
    info['minimum_sampled_pad_gap_m'] = float(np.min(contact['gaps']))
    if np.min(contact['gaps']) < .025:
        return reject('insufficient_open_gap')
    data = mujoco.MjData(model)
    links = controller.load.links
    names = [model.geom(i).name or '' for i in range(model.ngeom)]
    for fraction in np.linspace(0, 1, 11):
        data.qpos[:] = closed
        mujoco.mj_integratePos(model, data.qpos, delta, float(fraction))
        mujoco.mj_fwdPosition(model, data)
        for point in data.contact:
            if point.dist >= -1e-5:
                continue
            pair = [names[int(i)] for i in point.geom]
            if 'object' in pair and any(
                name in ('support', 'source_support', 'destination_support', 'ground', 'floor')
                or any(name.startswith(link+'/') for link in links) for name in pair
            ):
                continue
            info['collision_pair'] = pair
            return reject('sampled_path_collision')
    drive.observe(candidate, np.zeros(model.nv))
    rows = drive.robot
    equality = np.c_[drive.B[rows]*drive.limit, np.zeros(len(rows))]
    inequality = np.vstack([np.c_[np.eye(model.nu), -np.ones(model.nu)],
                            np.c_[-np.eye(model.nu), -np.ones(model.nu)]])
    lower = np.r_[-np.ones(drive.nj), np.zeros(model.nu-drive.nj), 0.]
    upper = np.ones(model.nu+1)
    trim = linprog(np.r_[np.zeros(model.nu), 1.], A_ub=inequality,
                   b_ub=np.zeros(2*model.nu), A_eq=equality,
                   b_eq=drive.d.qfrc_bias[rows], bounds=list(zip(lower, upper)))
    if not trim.success:
        return reject('no_bounded_hover')
    command = trim.x[:model.nu]*drive.limit
    command[:drive.nj] = (command[:drive.nj]+drive.kp*candidate[drive.qi])/drive.gain[:drive.nj]
    gi = np.flatnonzero(~drive.dock)
    ranges = model.jnt_range[[model.joint(drive.actuators[i]['name']).id for i in gi]]
    if np.any(command[gi] < ranges[:, 0]) or np.any(command[gi] > ranges[:, 1]):
        return reject('gimbal_command_range')
    info.update(selected=True, reason='shorter_geometry_path',
                preparation_s=time.perf_counter()-start,
                hover_residual=float(max(abs(equality@trim.x-drive.d.qfrc_bias[rows]))),
                hover_fraction=float(trim.x[-1]))
    return candidate, command, info


def transport_height(shape):
    """Lowest-point clearance with design allowances, not a tracking guarantee."""
    radius = (float(np.max(np.linalg.norm(shape.vertices, axis=1)))
              if shape.vertices is not None else float(np.linalg.norm(shape.size)/2))
    return max(.18, .10+.03+2*radius*np.sin(.2/2))
