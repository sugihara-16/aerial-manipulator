"""Identify runtime, physical inputs and numerical dependencies."""
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import subprocess
from .model import ROOT


def source_hashes():
    paths=set()
    for pattern in ['amsrr_det/*.py','amsrr_det/*.cc','build/*.py','native/*',
                    'build/native_bvls/*.py','build/native_bvls/*.cpp',
                    'configs/robot/*','configs/evaluation/*','third_party/mujoco/*','assets/robots/**/*','assets/contact_meshes/**/*',
                    'build/geometry_kernel/*','build/*.so','build/*/*.so',
                    'build/collision_physx_docks/**/*','third_party/quadprog/*',
                    'build/prediction_native_block_source/*']:
        paths.update(p for p in ROOT.glob(pattern) if p.is_file())
    for name in ['CMakeLists.txt','requirements.lock.txt','requirements.txt',
                 'assets/generated/authored_part_bounds.json','assets/generated/material_surfaces.npz',
                 'scripts/run_valve_task.py','scripts/run_valve_panel.py','scripts/report_valve_panel.py',
                 'scripts/run_walking_task.py','scripts/run_walking_panel.py','scripts/report_walking_panel.py']:
        if (ROOT/name).is_file():paths.add(ROOT/name)
    paths.update(p for p in (ROOT/'scripts').glob('*assembly*.py') if p.is_file())
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def dependency_versions():
    return {name:importlib.metadata.version(name) for name in ['numpy','scipy','mujoco','pin','clarabel','highspy','coal','trimesh']}


def physical_library():
    # /proc records the actual dynamic library loaded in this process.
    import mujoco
    libraries={Path(line.split()[-1]) for line in Path('/proc/self/maps').read_text().splitlines()
               if '/libmujoco.so' in line}
    if len(libraries)!=1:raise RuntimeError('cannot identify one loaded MuJoCo library')
    path=libraries.pop()
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),version=mujoco.__version__)


def environment():
    return dict(platform=platform.platform(),processor=platform.processor(),python=platform.python_version(),
        libraries=dependency_versions(),physical_library=physical_library(),cpu=subprocess.check_output(['lscpu','-J'],text=True),
        scope='PC simulation with measured control delay; no embedded-board or hard real-time guarantee')


def protocol_digest(specification):
    payload=dict(task=specification,files=source_hashes(),libraries=dependency_versions(),physical_library=physical_library())
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
