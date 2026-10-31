"""Frozen-source valve panel runner; physics never waits for controller work."""
import argparse,hashlib,json,os,subprocess,time,shutil
from pathlib import Path

from amsrr_det.runtime_manifest import source_hashes, environment

def sources():
 return source_hashes()

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('output',type=Path);ap.add_argument('--ids',type=str);ap.add_argument('--stop-timeouts',type=int,default=2);ap.add_argument('--panel',type=Path,default=Path('configs/evaluation/valve_panel36.json'))
 a=ap.parse_args();out=a.output;out.mkdir(parents=True,exist_ok=False)
 panel=json.loads(a.panel.read_text());cases=panel['cases']
 if a.ids:cases=[cases[i] for i in map(int,a.ids.split(','))]
 (out/'panel.json').write_text(json.dumps(dict(panel,cases=cases),indent=2))
 (out/'environment.json').write_text(json.dumps(environment(),indent=2))
 source=sources();(out/'sources.json').write_text(json.dumps(source,indent=2));(out/'cases.json').write_text(json.dumps(cases,indent=2))
 for name in source:
  saved=out/'source_snapshot'/name;saved.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(name,saved)
 env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONPATH='.')
 results=[];timeouts=0;begin=time.perf_counter()
 for case in cases:
  if sources()!=source:raise RuntimeError('source changed during panel')
  request=out/(case['id']+'_request.json');request.write_text(json.dumps(case['request']))
  command=['taskset','-c','0','.venv/bin/python','-m','scripts.run_valve_task',str(out/case['id']),
    '--request',str(request),'--seed',str(case['seed']),'--worker-cpu','2']
  with (out/(case['id']+'.log')).open('w') as log:
   proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
   try:code=proc.wait(timeout=600)
   except subprocess.TimeoutExpired:
    import signal
    os.killpg(proc.pid,signal.SIGTERM);proc.wait();code=-15
  path=out/case['id']/'result.json'
  r=json.loads(path.read_text()) if path.exists() else dict(task_success=False,reason='runner_error',exit_code=code)
  r.update(id=case['id'],axis_group=case['axis_group'],torque=case['torque'],initial_angle_group=case['initial_angle_group'])
  results.append(r);timeouts+=r['reason']=='planning_timeout'
  print(case['id'],r['task_success'],r['reason'],round(r.get('planning_s',0),3),flush=True)
  summary=dict(completed=len(results),total=len(cases),successes=sum(r['task_success'] for r in results),results=results,wall_s=time.perf_counter()-begin,source_changes=sources()!=source)
  (out/'summary.json').write_text(json.dumps(summary,indent=2))
  if a.stop_timeouts and timeouts>=a.stop_timeouts:break
