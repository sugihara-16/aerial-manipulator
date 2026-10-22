#!/usr/bin/env python3
"""Independent, offline numerical evidence with explicit model identity."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import argparse
import numpy as np
import pinocchio as pin

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from amsrr_det.diagnostics import fixture
from amsrr_det.reference import replay_reference
from amsrr_det.benchmark import run_case,sha

parser=argparse.ArgumentParser()
parser.add_argument('--output',type=Path,default=ROOT/'artifacts/validation_20261018/replay_converged')
parser.add_argument('--integrators',nargs='+',default=['drive_implicit_euler','drive_sdirk2'])
args=parser.parse_args()
out=args.output
out.mkdir(parents=True,exist_ok=True)
model,meta,scenario=fixture(out,1)
scenario.update(duration_s=.3,move_duration_s=.3,translation=[.002,0.,.001],integrator='drive_implicit_euler',
                thrust_lag_s=0.,gimbal_torque_lag_s=0.,dock_torque_lag_s=0.,profile='source_ideal_pd',
                assumptions=['Source ideal PD and armature are simulation settings, not measured hardware response'])
source=out/'closed_input.json';source.write_text(json.dumps(scenario,indent=2)+'\n')
subprocess.run([str(ROOT/'build/det_rollout'),str(out),str(source),'.002',str(out/'closed.json'),'closed',str(out/'commands.jsonl')],check=True,capture_output=True)
rows=[json.loads(line) for line in (out/'commands.jsonl').read_text().splitlines()]
references=[]
for rtol,atol,max_step,output_step in [(1e-9,1e-11,.001,.00001),(1e-11,1e-13,.00025,.0000025)]:
    settings=dict(rtol=rtol,atol=atol,max_step=max_step,output_step=output_step)
    start=time.perf_counter()
    value=replay_reference(model,meta,scenario,rows,scenario['duration_s'],**settings)
    references.append((value,dict(settings=settings,wall_s=time.perf_counter()-start,rhs_calls=value['nfev'],
        peak_joint_speed_rad_s=value['peak_joint_speed'],peak_joint_speeds_rad_s=value['peak_joint_speeds'].tolist(),
        final=value['final'])))
reference=references[-1][0]
ref_difference=dict(configuration_max=float(np.max(abs(pin.difference(model,np.array(references[0][0]['final']['q0']),np.array(reference['final']['q0']))))),
                    velocity_max=float(np.max(abs(np.array(references[0][0]['final']['v0'])-reference['final']['v0']))),
                    peak_joint_speed_max_rad_s=float(np.max(abs(references[0][0]['peak_joint_speeds']-reference['peak_joint_speeds']))))
scenario['command_schedule']=[dict(time=r['time'],thrust=r['thrust_command'],position=r['position_command'],velocity=r['velocity_command']) for r in rows]
report=dict(schema='amsrr_det.independent_numerics.v2',reference='SciPy Radau, two tolerance/time-step settings; restart at every command change',
            purpose='Numerical comparison of fixed recorded inputs; not independent physical validation or full closed-loop reference',
            references=[r[1] for r in references],reference_difference=ref_difference,
            reference_peak_joint_speed_rad_s=reference['peak_joint_speed'],
            source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'amsrr_det/reference.py',Path(__file__)]},runs=[],task_accepted=False)
for method in args.integrators:
    scenario['integrator']=method
    for h in [.002,.001,.0005]:
        r=run_case(out,scenario,h,out/f'{method}_{h:.6f}.json',closed=False)
        report['runs'].append(dict(integrator=method,step_s=h,stop_reason=r['stop_reason'],full_horizon=r['full_horizon'],wall_s=r['process_wall_s'],
            configuration_max_difference=float(np.max(abs(pin.difference(model,np.array(reference['final']['q0']),np.array(r['final']['q']))))),
            velocity_max_difference=float(np.max(abs(np.array(reference['final']['v0'])-r['final']['v']))),
            peak_joint_speed_rad_s=r['metrics']['max_joint_speed_rad_s'],
            peak_joint_speed_error_rad_s=abs(r['metrics']['max_joint_speed_rad_s']-reference['peak_joint_speed'])))
(out/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
