"""Save the exact local source bytes used when starting an experiment."""
import hashlib
import json
from pathlib import Path
from .model import ROOT


_paths=[*ROOT.glob('amsrr_det/*.py'),*ROOT.glob('amsrr_det/*.cc'),*ROOT.glob('scripts/*.py'),*ROOT.glob('configs/evaluation/*.yaml'),
        *ROOT.glob('configs/robot/*.yaml'),*ROOT.glob('build/physx_*.py'),
        ROOT/'build/load_saved_plan.py',ROOT/'build/physx_minimal.kit',ROOT/'build/physics_diagnostics.py',
        ROOT/'requirements.lock.txt',ROOT/'requirements.txt',ROOT/'requirements-physx-local.txt']
_paths.append(ROOT/'build/run_mujoco_task.py')
_paths.extend(ROOT/p for p in ['CMakeLists.txt','native/geometry_kernel.cpp',
    'build/geometry_kernel/libgeometry_kernel.so','build/geometry_kernel/identity.json',
    'assets/generated/authored_part_bounds.json','assets/generated/material_surfaces.npz'])
# Capture once per process before a long batch, not again from changing files
# when a later case starts in a worker that still holds the earlier imports.
_source_bytes={str(p.relative_to(ROOT)):p.read_bytes() for p in _paths if p.exists()}


def save_sources(output):
    destination=Path(output)/'source_snapshot'
    if destination.exists():
        raise ValueError('experiment directory already contains a source snapshot; choose a new output')
    hashes={}
    for relative,raw in _source_bytes.items():
        target=destination/relative
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
        hashes[relative]=hashlib.sha256(raw).hexdigest()
    (destination/'manifest.json').write_text(json.dumps(hashes,indent=2))
    return hashes


def save_plan(plan,output,scope):
    """Persist this newly generated plan for diagnosis, never for selection."""
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    (output/'scene.xml').write_text(plan.plant.xml)
    result=dict(scope=scope,initial_q=plan.initial_q.tolist(),open_q=plan.open_q.tolist(),
        closed_q=plan.closed_q.tolist(),surfaces=plan.surfaces,points=plan.points.tolist(),
        normals=plan.normals.tolist(),lift_height=plan.lift_height,approach_height=plan.approach_height,
        trims=plan.trims,fit=plan.fit,planning_time_s=plan.wall_s,model=plan.plant.meta,operating_point=getattr(plan,'operating_point',None))
    (output/'plan.json').write_text(json.dumps(result,indent=2))
