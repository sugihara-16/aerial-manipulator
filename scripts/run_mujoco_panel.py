"""Execute every frozen task through fresh generation and a completed-robot spawn."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def run_case(case, root, options=()):
    out=root/case['id'];out.mkdir()
    request=out/'input.json';request.write_text(json.dumps(case,indent=2))
    env=os.environ.copy();env.pop('PYTHONPATH',None)
    env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    started=time.perf_counter()
    with (out/'execution.log').open('w') as log:
        process=subprocess.run([sys.executable,'-m','build.run_mujoco_task',
                                '--request',str(request),'--output',str(out),*options],
                               stdout=log,stderr=subprocess.STDOUT,env=env)
    if process.returncode or not (out/'result.json').exists():
        # Keep failed processes in the denominator, including generation crashes.
        result=dict(task_success=False,reason='execution_process_failed',
                    execution_entrypoint='fresh_task_input',scope='fresh_task_generation',
                    request=case,returncode=process.returncode)
        (out/'result.json').write_text(json.dumps(result,indent=2))
    else:result=json.loads((out/'result.json').read_text())
    row=dict(case=case['id'],success=result['task_success'],reason=result['reason'],
             wall_s=time.perf_counter()-started,planning_time_s=result.get('planning_time_s'))
    print(json.dumps(row),flush=True)
    return row


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--contact-counts',nargs='+',type=int,choices=[2,3,4],default=[2,3,4])
    args=parser.parse_args()
    panel=json.loads(args.input.read_text())
    if args.workers<1:parser.error('workers must be positive')
    if len({c['id'] for c in panel['cases']})!=len(panel['cases']):raise ValueError('duplicate input IDs')
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'panel.json').write_text(json.dumps(panel,indent=2))
    options=['--contact-counts',*map(str,args.contact_counts)]
    (args.output/'runner_options.json').write_text(json.dumps(options))
    started=time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows=list(pool.map(lambda case:run_case(case,args.output,options),panel['cases']))
    (args.output/'batch.json').write_text(json.dumps(dict(workers=args.workers,
        wall_s=time.perf_counter()-started,rows=rows),indent=2))


if __name__=='__main__':main()
