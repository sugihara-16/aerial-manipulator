"""Freeze a controller version and replay a specified task panel without retries."""
import argparse,os,sys,json,time,hashlib,shutil,subprocess,platform,signal
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
parser=argparse.ArgumentParser();parser.add_argument('output',type=Path);parser.add_argument('--ids',type=Path);parser.add_argument('--jobs',type=int,default=4);parser.add_argument('--cores',default='0,4,8,12');parser.add_argument('--worker-cores',default='2,6,10,14');args=parser.parse_args()
cores=list(map(int,args.cores.split(',')));worker_cores=list(map(int,args.worker_cores.split(',')))
if len(cores)<args.jobs or len(worker_cores)<args.jobs or len(set(cores[:args.jobs]+worker_cores[:args.jobs]))<2*args.jobs:parser.error('each publisher and worker needs a distinct CPU')
if not 1 <= args.jobs <= 8: parser.error('jobs must be 1..8')
root=Path(__file__).resolve().parents[1];source=root/'artifacts/runs/control_tail150_v8_20261027'
runner=root/'build/check_relative_task.py'
panel=json.loads((root/'artifacts/control_tail_20261027/panel150_heavy_first.json').read_text())
if args.ids:
 ids=json.loads(args.ids.read_text());byid={c['id']:c for c in panel['cases']};panel['cases']=[byid[n] for n in ids]
out=args.output.resolve();out.mkdir(exist_ok=False)
files=[p for folder in ['amsrr_det','native','configs','third_party/quadprog','assets'] for p in (root/folder).rglob('*') if p.is_file() and p.suffix in ['.py','.cc','.cpp','.c','.yaml','.json','.obj','.stl','.h','.urdf'] and '__pycache__' not in str(p)]
files += [root/'scripts/report_relative_panel.py']
files += list((root/'build').glob('*.py')) + [runner,Path(__file__).resolve()]
hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()for p in files}
for rel in hashes:
 target=out/'source_snapshot'/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/rel,target)
(out/'source_manifest.json').write_text(json.dumps(hashes,indent=2));(out/'panel.json').write_text(json.dumps(panel,indent=2))
(out/'evaluation_scope.json').write_text(json.dumps(dict(source=str(source),runner=str(runner.relative_to(root)),timing='measured',all_robot_collisions=True,auxiliary_stops=False,scope='Controller replay of task-generated structures; not fresh planning or a structure catalogue',hostname=platform.node(),jobs=args.jobs,cores=cores[:args.jobs],worker_cores=worker_cores[:args.jobs]),indent=2))
import numpy,mujoco,scipy
(out/'environment.json').write_text(json.dumps(dict(python=sys.version,numpy=numpy.__version__,scipy=scipy.__version__,mujoco=mujoco.__version__,mujoco_path=mujoco.__file__),indent=2))
started=time.perf_counter()
def slot(i):
 rows=[]
 for case in panel['cases'][i::args.jobs]:
  name=case['id'];target=out/name;before=time.perf_counter();inputs={};missing=[]
  for rel in ['generated_plan/plan.json','generated_plan/scene.xml','generated_plan/request.json','endpoint.npz']:
   p=source/name/rel
   if p.exists():inputs[rel]=hashlib.sha256(p.read_bytes()).hexdigest()
   else:missing.append(rel)
  if missing:
   target.mkdir();result=dict(task_success=False,executed=False,reason='source_plan_unavailable',missing=missing)
   (target/'result.json').write_text(json.dumps(result,indent=2));returncode=None
  else:
   try:
    with (out/(name+'.log')).open('w') as log:
     proc=subprocess.Popen(['taskset','-c',str(cores[i]),sys.executable,str(runner),str(source/name),str(target),'--collision','shape-pairs','--worker-cpu',str(worker_cores[i])],cwd=root,env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONPATH=str(root)),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
     try:proc.wait(timeout=620)
     except subprocess.TimeoutExpired:
      os.killpg(proc.pid,signal.SIGTERM);proc.wait();raise
    returncode=proc.returncode
   except subprocess.TimeoutExpired:returncode=-9
   if (target/'result.json').exists():result=json.loads((target/'result.json').read_text())
   else:
    target.mkdir(exist_ok=True);result=dict(task_success=False,executed=True,reason='wall_time_limit'if returncode==-9 else'process_error',returncode=returncode)
    (target/'result.json').write_text(json.dumps(result,indent=2))
  (target/'input_hashes.json').write_text(json.dumps(inputs,indent=2))
  row=dict(case=name,exit=returncode,wall_s=time.perf_counter()-before,success=result['task_success'],reason=result['reason'],phase=result.get('phase'),modules=result.get('modules'),control=result.get('control'))
  rows.append(row);print(json.dumps(row),flush=True)
 return rows
with ThreadPoolExecutor(max_workers=args.jobs)as pool:rows=[r for batch in pool.map(slot,range(args.jobs))for r in batch]
changes=[k for k,v in hashes.items()if not(root/k).exists()or hashlib.sha256((root/k).read_bytes()).hexdigest()!=v]
summary=dict(completed=len(rows),successes=sum(r['success']for r in rows),wall_s=time.perf_counter()-started,source_changes=changes,rows=rows)
(out/'run_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps({k:v for k,v in summary.items()if k!='rows'}),flush=True)
