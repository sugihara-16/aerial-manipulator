"""Measured workloads, explicitly separate from task completion/acceptance."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import numpy as np
import pinocchio as pin
from .model import ROOT
from .diagnostics import fixture
from .numerics import diagnostic_tolerances


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_case(modeldir,scenario,h,output,closed=True,timeout=60.):
    path=output.with_suffix('.input.json');path.write_text(json.dumps(scenario,indent=2)+'\n')
    args=[str(ROOT/'build/det_rollout'),str(modeldir),str(path),str(h),str(output)]
    if closed:args.append('closed')
    start=time.perf_counter();p=subprocess.run(args,capture_output=True,text=True,timeout=timeout)
    if p.returncode not in [0,3]:raise RuntimeError(p.stderr)
    result=json.loads(output.read_text());result['process_wall_s']=time.perf_counter()-start
    result['binary_sha256']=sha(ROOT/'build/det_rollout');result['scenario_sha256']=sha(path)
    result['model_binary_sha256']=sha(modeldir/'model.bin')
    output.write_text(json.dumps(result,indent=2)+'\n');return result


def compare(model,a,b):
    numerical_keys={'integrator','step_tolerances','minimum_step_s'}
    physical=lambda r:{k:v for k,v in r['scenario'].items() if k not in numerical_keys}
    if physical(a)!=physical(b) or a['model']!=b['model'] or a['control_period_s']!=b['control_period_s']:
        raise ValueError('different physical inputs or control clocks')
    result=dict(both_full_horizon=a['full_horizon'] and b['full_horizon'],task_accepted=False,
                coarse_stop=a['stop_reason'],fine_stop=b['stop_reason'],total_process_s=a['process_wall_s']+b['process_wall_s'])
    if result['both_full_horizon']:
        result.update(final_configuration_max_difference=float(np.max(abs(pin.difference(model,np.array(a['final']['q']),np.array(b['final']['q']))))),
                      final_velocity_max_difference=float(np.max(abs(np.array(a['final']['v'])-b['final']['v']))),
                      peak_metric_differences={k:abs(a['metrics'][k]-b['metrics'][k]) for k in a['metrics']})
    # No numerical acceptance threshold is invented here; record differences.
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--repeats',type=int,default=1)
    p.add_argument('--duration',type=float,default=100.);p.add_argument('--modules',type=int,nargs='+',default=[1,3,8])
    p.add_argument('--integrator',choices=['drive_implicit_euler','adaptive_implicit_euler','drive_sdirk2'],default='drive_implicit_euler')
    p.add_argument('--tolerance-scale',type=float,default=1.)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    report=dict(schema='amsrr_det.benchmark.v1',started_at=datetime.now(timezone.utc).isoformat(),host=platform.node(),architecture=platform.machine(),
                processor=platform.processor(),python=platform.python_version(),pinocchio=pin.__version__,cases=[],
                claim='Prepared-structure, contact-free numerical diagnostic only; not online planning, task success, or real I/O timing',
                hardware_validated=False,source_sha256={str(f.relative_to(ROOT)):sha(f) for f in sorted((ROOT/'native').glob('*'))},
                python_sha256={str(f.relative_to(ROOT)):sha(f) for f in sorted((ROOT/'amsrr_det').glob('*.py'))})
    for n in args.modules:
        for repeat in range(args.repeats):
            name=f'n{n}_r{repeat}';directory=args.output/name;start=time.perf_counter()
            model,meta,scenario=fixture(directory,n)
            prep=time.perf_counter()-start
            scenario.update(duration_s=args.duration,translation=[.2,0.,.1],move_duration_s=8.,integrator=args.integrator,
                            thrust_lag_s=0.,gimbal_torque_lag_s=0.,dock_torque_lag_s=0.,profile='source_ideal_pd',
                            assumptions=['source ideal PD settings; actuator dynamics not identified','prepared structure; assembly excluded','no contact/collision acceptance'])
            if args.integrator=='adaptive_implicit_euler':scenario['step_tolerances']=diagnostic_tolerances(args.tolerance_scale)
            (directory/'scenario.json').write_text(json.dumps(scenario,indent=2)+'\n')
            a=run_case(directory,scenario,.002,directory/'h2.json',timeout=120.)
            if args.integrator=='adaptive_implicit_euler':scenario['step_tolerances']=diagnostic_tolerances(.25*args.tolerance_scale)
            b=run_case(directory,scenario,.001,directory/'h1.json',timeout=120.)
            entry=dict(name=name,modules=n,repeat=repeat,model_preparation_s=prep,coarse_wall_s=a['process_wall_s'],fine_wall_s=b['process_wall_s'],comparison=compare(model,a,b),
                       total_s=prep+a['process_wall_s']+b['process_wall_s'])
            report['cases'].append(entry)
            (args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps(entry),flush=True)
    report['finished_at']=datetime.now(timezone.utc).isoformat()
    (args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
