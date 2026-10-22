#!/usr/bin/env python3
"""Compare adaptive calculations against an already converged fixed-input reference."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pinocchio as pin

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from amsrr_det.benchmark import run_case,sha
from amsrr_det.numerics import diagnostic_tolerances

p=argparse.ArgumentParser()
p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
reference=json.loads((args.reference/'comparison.json').read_text())
nominal=json.loads((args.reference/'drive_implicit_euler_0.002000.json').read_text())
if sha(args.reference/'model.bin')!=nominal['model_binary_sha256']:raise RuntimeError('reference model changed')
model=pin.Model();model.loadFromBinary(str(args.reference/'model.bin'))
ref=reference['references'][-1];scenario=nominal['scenario']
scenario.update(integrator='adaptive_implicit_euler')
report=dict(reference_sha256=sha(args.reference/'comparison.json'),model_sha256=sha(args.reference/'model.bin'),
    note='Local error control only. Global peak accuracy is measured here, not guaranteed by local tolerances.',
    task_accepted=False,runs=[])
for scale in [10.,1.,.1]:
    scenario['step_tolerances']=diagnostic_tolerances(scale)
    r=run_case(args.reference,scenario,.002,args.output/f'scale_{scale}.json',closed=False)
    report['runs'].append(dict(tolerance_scale=scale,counts=r['counts'],wall_s=r['process_wall_s'],full_horizon=r['full_horizon'],stop_reason=r['stop_reason'],
        configuration_max_difference=float(np.max(abs(pin.difference(model,np.array(ref['final']['q0']),np.array(r['final']['q']))))),
        velocity_max_difference=float(np.max(abs(np.array(ref['final']['v0'])-r['final']['v']))),
        peak_joint_speed_rad_s=r['metrics']['max_joint_speed_rad_s'],
        peak_joint_speed_error_rad_s=abs(r['metrics']['max_joint_speed_rad_s']-ref['peak_joint_speed_rad_s'])))
    (args.output/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['runs'][-1]),flush=True)
