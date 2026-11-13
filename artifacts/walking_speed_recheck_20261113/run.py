"""Fixed preparation-only paired recheck; no task success claims."""
import hashlib,json,os,signal,subprocess,time
from pathlib import Path
from amsrr_det.runtime_manifest import source_hashes,environment
root=Path('artifacts/walking_speed_recheck_20261113')
previous=Path('artifacts/walking_integrated20_20261112/full36')
cases=['n6_v22_right','n6_v30_right','n7_v22_straight','n7_v30_left','n8_v22_straight','n8_v26_left']
def write(p,x):
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2));tmp.replace(p)
def stop(p):
 try:os.killpg(p.pid,signal.SIGTERM)
 except ProcessLookupError:return
 try:p.wait(timeout=3)
 except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
source=source_hashes();old=json.loads((previous/'sources.json').read_text())
assert all(old.get(k)==v for k,v in source.items()),'Runtime differs from the previous 20 second test'
write(root/'sources.json',source);write(root/'environment.json',environment())
write(root/'protocol.json',dict(cases=cases,repetitions=2,scope='Initial preparation only; intentionally stop on ready; same inputs and runtime as full36 with 20 second budget',previous=str(previous),concurrency=1,nice=-15))
subprocess.run(['sudo','-n','renice','-n','-15','-p',str(os.getpid())],check=True,stdout=subprocess.DEVNULL)
env=dict(os.environ,PYTHONPATH='.',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1');rows=[]
for rep in range(1,3):
 for case in cases:
  assert source_hashes()==source
  folder=root/f'repeat{rep}'/case;folder.mkdir(parents=True,exist_ok=False)
  (folder/'input.json').write_bytes((previous/case/'input.json').read_bytes())
  print('START',rep,case,flush=True);began=time.perf_counter();ready=None;timed_out=False
  with (folder/'execution.log').open('w') as log:
   p=subprocess.Popen(['.venv/bin/python','-m','scripts.run_task',str(folder/'input.json'),str(folder)],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   write(root/'active.json',dict(repetition=rep,case=case,pid=p.pid,started=time.time()))
   while p.poll() is None:
    if (folder/'ready.json').exists():ready=json.loads((folder/'ready.json').read_text())['preparation_s'];stop(p);break
    if time.perf_counter()-began>45:timed_out=True;stop(p);break
    time.sleep(.05)
   p.wait()
  selection=json.loads((folder/'execution/selection.json').read_text()) if (folder/'execution/selection.json').exists() else {}
  result=json.loads((folder/'result.json').read_text()) if (folder/'result.json').exists() else {}
  old_result=json.loads((previous/case/'result.json').read_text());old_selection=json.loads((previous/case/'execution/selection.json').read_text())
  row=dict(repetition=rep,case=case,ready=ready is not None,preparation_s=ready,within20=ready is not None and ready<=20.,generation_s=selection.get('planning_s'),completed_trials=(selection.get('evidence')or{}).get('completed_trials'),selection_status=selection.get('status'),outcome='intentionally_stopped_after_preparation' if ready is not None else 'watchdog_timeout' if timed_out else result.get('reason'),previous_preparation_s=old_result.get('preparation_s'),previous_reason=old_result.get('reason'),previous_generation_s=old_selection.get('planning_s'),wall_s=time.perf_counter()-began)
  write(folder/'measurement.json',row);rows.append(row);changed=source_hashes()!=source
  write(root/'summary.json',dict(completed=len(rows),total=12,source_changes=changed,rows=rows));print('END',rep,case,row['preparation_s'],row['selection_status'],flush=True)
  assert not changed
write(root/'active.json',dict(finished=True));print('COMPLETE',flush=True)
