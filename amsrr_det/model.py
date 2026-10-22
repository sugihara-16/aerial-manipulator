"""Build articulated forests from the copied URDF, including reversed dock joints.

All movable dock joints remain movable after assembly. Reversing a URDF edge
uses exp(-S*q) A^-1, not a fixed transform that silently locks the incoming arm.
Rotor spin is collapsed at zero; mass/inertia remain. Aerodynamic reaction torque
is zero in the source configuration. Gyroscopic rotor spin is not modeled.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import hashlib
import json
import xml.etree.ElementTree as ET

import numpy as np
import pinocchio as pin
import yaml

ROOT = Path(__file__).resolve().parents[1]
PORTS = {
    "pitch1": "pitch_connect_dummy_1", "pitch2": "pitch_connect_dummy_2",
    "yaw1": "yaw_connect_dummy_1", "yaw2": "yaw_connect_dummy_2",
}


@lru_cache(maxsize=4)
def _yaml_document(raw):
    """Read-only parsed configuration, invalidated by its exact source bytes."""
    return yaml.safe_load(raw)


def transform(origin):
    if origin is None:
        return pin.SE3.Identity()
    xyz = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
    rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
    return pin.SE3(pin.rpy.rpyToMatrix(rpy), xyz)


@dataclass(frozen=True)
class DockEdge:
    a: int
    port_a: str
    b: int
    port_b: str


def validate_graph(n: int, edges: list[DockEdge]):
    if not 1 <= n <= 8:
        raise ValueError("module count must be 1..8")
    parent = list(range(n))
    occupied = set()
    def root(i):
        while parent[i] != i:
            i = parent[i]
        return i
    for e in edges:
        if e.a not in range(n) or e.b not in range(n):
            raise ValueError("unknown module")
        for m, p in ((e.a, e.port_a), (e.b, e.port_b)):
            if p not in PORTS or (m, p) in occupied:
                raise ValueError("unknown or occupied port")
            occupied.add((m, p))
        if e.port_a[:3] == e.port_b[:3]:
            raise ValueError("only pitch-to-yaw connections are physical")
        if root(e.a) == root(e.b):
            raise ValueError("connections must form a forest")
        parent[root(e.b)] = root(e.a)


@lru_cache(maxsize=4)
def _urdf_parts(raw):
    """Parse immutable robot properties once per exact authored input."""
    tree = ET.fromstring(raw)
    links = {}
    for link in tree.findall("link"):
        inertial = link.find("inertial")
        inertia = pin.Inertia.Zero()
        if inertial is not None:
            mass = float(inertial.find("mass").get("value"))
            it = inertial.find("inertia")
            I = np.array([
                [float(it.get("ixx")), float(it.get("ixy")), float(it.get("ixz"))],
                [float(it.get("ixy")), float(it.get("iyy")), float(it.get("iyz"))],
                [float(it.get("ixz")), float(it.get("iyz")), float(it.get("izz"))]])
            T = transform(inertial.find("origin"))
            inertia = pin.Inertia(mass, T.translation, T.rotation @ I @ T.rotation.T)
        links[link.get("name")] = inertia
    joints = []
    for j in tree.findall("joint"):
        parent, child = [j.find(k).get("link") for k in ("parent", "child")]
        kind = j.get("type")
        if kind not in ['fixed','revolute','continuous'] or (kind=='continuous' and j.get('name') not in ['rotor1','rotor2','rotor3','rotor4']):
            raise ValueError('unsupported joint type; cannot silently lock a movable joint')
        moving = kind == "revolute"
        axis = np.fromstring(j.find("axis").get("xyz"), sep=" ") if moving else np.zeros(3)
        lim = j.find("limit")
        lo, hi = (float(lim.get("lower")), float(lim.get("upper"))) if moving else (0, 0)
        joints.append((parent, child, j.get("name"), transform(j.find("origin")), axis, moving, lo, hi))
    return links, joints


def build_model(n=1, edges=(), payload=None, urdf=None, gravity=(0.,0.,-9.80665)):
    edges = [e if isinstance(e, DockEdge) else DockEdge(**e) for e in edges]
    validate_graph(n, edges)
    source = Path(urdf or ROOT / "assets/robots/holon/holon.urdf").resolve()
    if not source.is_relative_to(ROOT):raise ValueError('URDF must be copied into this independent repository')
    inputs = [source, ROOT/"configs/robot/joint_actuators.yaml", ROOT/"configs/robot/thrust_model.yaml"]
    input_bytes = {path:path.read_bytes() for path in inputs}
    links, authored_joints = _urdf_parts(input_bytes[source])
    graph = {f"m{i}/{name}": [] for i in range(n) for name in links}
    for i in range(n):
        for parent, child, name, placement, axis, moving, lo, hi in authored_joints:
            a, b = f"m{i}/{parent}", f"m{i}/{child}"
            record = (f"m{i}/{name}", placement, axis, moving, lo, hi)
            graph[a].append((b, True, record))
            graph[b].append((a, False, record))
    mate = pin.SE3(pin.rpy.rpyToMatrix(0., 0., np.pi), np.zeros(3))
    for idx, e in enumerate(edges):
        a, b = f"m{e.a}/{PORTS[e.port_a]}", f"m{e.b}/{PORTS[e.port_b]}"
        rec = (f"seam{idx}", mate, np.zeros(3), False, 0., 0.)
        graph[a].append((b, True, rec)); graph[b].append((a, False, rec))

    profile = _yaml_document(input_bytes[inputs[1]])["joint_actuator_model"]["actuator_roles"]
    model = pin.Model()
    model.name = "holon_det"
    gravity=np.asarray(gravity,dtype=float)
    if gravity.shape!=(3,) or not np.all(np.isfinite(gravity)):
        raise ValueError('invalid gravity vector')
    model.gravity.linear=gravity
    visited, frames, actuators, bases = set(), {}, [], []
    def visit(name, parent_joint, placement):
        visited.add(name)
        localname = name.split("/", 1)[1]
        model.appendBodyToJoint(parent_joint, links[localname], placement)
        frames[name] = model.addFrame(pin.Frame(name, parent_joint, placement, pin.FrameType.BODY))
        for child, forward, (jname, A, axis, moving, lo, hi) in graph[name]:
            if child in visited:
                continue
            if not moving:
                visit(child, parent_joint, placement * (A if forward else A.inverse()))
                continue
            B = pin.SE3.Identity() if forward else A.inverse()
            origin = placement * A if forward else placement
            jid = model.addJoint(parent_joint, pin.JointModelRevoluteUnaligned(axis if forward else -axis), origin, jname)
            role = "vectoring" if "gimbal" in jname else "dock"
            prof = profile[role]; drive = prof["simulation_drive"]
            j = model.joints[jid]
            model.lowerPositionLimit[j.idx_q] = lo
            model.upperPositionLimit[j.idx_q] = hi
            model.velocityLimit[j.idx_v] = drive["safe_velocity_limit_rad_s"]
            model.effortLimit[j.idx_v] = prof["continuous_torque_limit_nm"]
            model.armature[j.idx_v] = drive.get("armature_kg_m2", 0.)
            actuators.append(dict(name=jname, joint=jid, q=j.idx_q, v=j.idx_v,
                                  role=role, lower=lo, upper=hi, kp=drive["stiffness"], kd=drive["damping"],
                                  effort=prof["continuous_torque_limit_nm"], peak=prof["peak_torque_nm"],
                                  speed=drive["safe_velocity_limit_rad_s"]))
            visit(child, jid, B)
    for i in range(n):
        name = f"m{i}/root"
        if name in visited:
            continue
        jid = model.addJoint(0, pin.JointModelFreeFlyer(), pin.SE3.Identity(), f"base{i}")
        bases.append(dict(joint=jid, q=model.joints[jid].idx_q, v=model.joints[jid].idx_v, module=i))
        visit(name, jid, pin.SE3.Identity())
    object_info = None
    if payload is not None:
        size = np.asarray(payload["size"], dtype=float); mass = float(payload["mass"])
        if not np.isfinite(mass) or mass <= 0 or size.shape != (3,) or not np.all(np.isfinite(size)) or np.any(size <= 0):
            raise ValueError("invalid payload")
        jid = model.addJoint(0, pin.JointModelFreeFlyer(), pin.SE3.Identity(), "object")
        I = mass / 12 * np.diag([size[1]**2+size[2]**2, size[0]**2+size[2]**2, size[0]**2+size[1]**2])
        I=np.asarray(payload.get('inertia',I),dtype=float)
        com=np.asarray(payload.get('com',[0.,0.,0.]),dtype=float)
        if I.shape!=(3,3) or com.shape!=(3,) or not np.all(np.isfinite(I)) or not np.all(np.isfinite(com)) or not np.allclose(I,I.T,atol=1e-13):
            raise ValueError('invalid payload inertia or COM')
        eig=np.linalg.eigvalsh(I)
        if eig[0]<=0 or eig[-1]>eig[0]+eig[1]+1e-12:
            raise ValueError('nonphysical payload inertia')
        model.appendBodyToJoint(jid, pin.Inertia(mass, com, I), pin.SE3.Identity())
        fid = model.addFrame(pin.Frame("object", jid, pin.SE3.Identity(), pin.FrameType.BODY))
        frames["object"] = fid
        object_info = dict(joint=jid, frame=fid, q=model.joints[jid].idx_q, v=model.joints[jid].idx_v, mass=mass, size=size.tolist(),com=com.tolist(),inertia=I.tolist())
    rotors = []
    rotor_profile={r['rotor_id']:r for r in _yaml_document(input_bytes[inputs[2]])['rotors']}
    for i in range(n):
        for k in range(1, 5):
            actuator = next(a for a in actuators if a["name"] == f"m{i}/gimbal{k}")
            spec=rotor_profile[f'thrust_{k}']
            if spec['reaction_torque_coeff_nm_per_n']!=0:
                raise ValueError('nonzero rotor reaction torque is not implemented')
            rotors.append(dict(frame=frames[f"m{i}/thrust_{k}"], gimbal=actuators.index(actuator),
                               minimum=spec['thrust_min_n'],maximum=spec['thrust_max_n'],module=i))
    metadata = dict(schema="amsrr_det.model.v1", modules=n, edges=[e.__dict__ for e in edges],
                    nq=model.nq, nv=model.nv, frames=frames, actuators=actuators, bases=bases, rotors=rotors,
                    object=object_info, mass=float(sum(i.mass for i in model.inertias)),gravity=gravity.tolist(),
                    source_hashes={str(p.relative_to(ROOT)): hashlib.sha256(raw).hexdigest() for p,raw in input_bytes.items()},
                    assumptions=["zero rotor spin and reaction torque", "rigid dock seam; latch transients not modeled",
                                 "source simulation PD/armature; not identified hardware response",
                                 "backlash, backdrive friction and electrical/thermal drive limits not modeled"])
    return model, metadata


def export_model(directory, n=1, edges=(), payload=None, gravity=(0.,0.,-9.80665)):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    model, metadata = build_model(n, edges, payload, gravity=gravity)
    model.saveToBinary(str(directory / "model.bin"))
    (directory / "model.json").write_text(json.dumps(metadata, indent=2)+"\n")
    return model, metadata
