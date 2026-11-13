"""Frozen integrated walking panel: same tasks/seeds, no outcome-based retries."""
import json,os,signal,subprocess,time,shutil,hashlib
from pathlib import Path
from amsrr_det.runtime_manifest import source_hashes,environment
from amsrr_det.task_spec import defaults,make_spec
from amsrr_det.task_execution import write

root=Path('artifacts/walking_integrated_20261112/full36')
root.mkdir(exist_ok=False)
panel=json.loads(Path('artifacts/walking_face_20261109/panel599.json').read_text())
panel['scope']='Fresh generation, individual takeoff and assembly, simultaneous flight/morph, descent, ground load transfer, 5 m walking; same inputs and seeds as panel599'
panel['execution_environment']=dict(host='local',concurrency=1,planning='all allowed cores',physics_cpu=8,control_worker_cpu=2,nice=-15,rendering=False,live_publication=True)
write(root/'panel.json',panel)
def hashes():
 h=source_hashes()
 for name in ['scripts/run_task.py','artifacts/walking_integrated_20261112/run_panel.py']:
  h[name]=hashlib.sha256(Path(name).read_bytes()).hexdigest()
 return h
source=hashes()
for name in source:
 p=root/'source_snapshot'/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(name,p)
write(root/'sources.json',source);write(root/'environment.json',environment())
for case in panel['cases']:
 t=case['task'];v=defaults('walking');v.update(modules=t['modules'],speed=t['speed'],distance=t['distance'],curvature=t['curvature'],seed=case['seed'])
 folder=root/case['id'];folder.mkdir();write(folder/'input.json',make_spec('walking',v,'integrated'))
subprocess.run(['sudo','-n','renice','-n','-15','-p',str(os.getpid())],check=True,stdout=subprocess.DEVNULL)
env=dict(os.environ,PYTHONPATH='.',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1');rows=[];started=time.perf_counter()
def stop(proc):
 try:os.killpg(proc.pid,signal.SIGTERM)
 except ProcessLookupError:return
 try:proc.wait(timeout=3)
 except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
for case in panel['cases']:
 if hashes()!=source:raise RuntimeError('Source changed before case '+case['id'])
 folder=root/case['id'];began=time.perf_counter();reason=None;ready=None;affinity_set=False
 print('START',case['id'],flush=True)
 with (folder/'execution.log').open('w') as log:
  proc=subprocess.Popen(['.venv/bin/python','-m','scripts.run_task',str(folder/'input.json'),str(folder)],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  write(root/'active.json',dict(case=case['id'],pid=proc.pid,started=time.time()))
  while proc.poll() is None:
   if (folder/'ready.json').exists():
    ready=json.loads((folder/'ready.json').read_text())['preparation_s']
    if not affinity_set:
     try:os.sched_setaffinity(proc.pid,{8});affinity_set=True
     except ProcessLookupError:pass
    if ready>15:reason='initial_planning_deadline';break
   elapsed=time.perf_counter()-began
   if elapsed>600:reason='execution_wall_timeout';break
   if ready is None and elapsed>45:reason='planning_watchdog_timeout';break
   time.sleep(.15)
  if reason:stop(proc)
  code=proc.wait()
 if reason and not (folder/'result.json').exists():
  write(folder/'result.json',dict(success=False,reason=reason,preparation_s=ready,modules=case['task']['modules'],execution_started=ready is not None))
 if not (folder/'result.json').exists():write(folder/'result.json',dict(success=False,reason='runner_error',exit_code=code))
 result=json.loads((folder/'result.json').read_text())
 row=dict(id=case['id'],task=case['task'],seed=case['seed'],success=result['success'],reason=result.get('reason'),preparation_s=result.get('preparation_s'),wall_s=time.perf_counter()-began,result=str(folder/'result.json'))
 rows.append(row);changed=hashes()!=source
 write(root/'summary.json',dict(completed=len(rows),total=len(panel['cases']),successes=sum(r['success'] for r in rows),wall_s=time.perf_counter()-started,source_changes=changed,results=rows))
 print('END',case['id'],row['success'],row['reason'],'successes',sum(r['success'] for r in rows),'/',len(rows),flush=True)
 if changed:raise RuntimeError('Source changed during case '+case['id'])
write(root/'active.json',dict(finished=True));print('COMPLETE',flush=True)
