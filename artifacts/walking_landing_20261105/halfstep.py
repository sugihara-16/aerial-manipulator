"""Predeclared six saved-plan sensitivity checks, not fresh planning results."""
import concurrent.futures,json,os,subprocess,sys
from pathlib import Path
from amsrr_det.runtime_manifest import source_hashes
root=Path(sys.argv[1]);out=Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=False)
panel=json.loads((root/'panel.json').read_text());ids=[2,6,15,20,24,35];cases=[panel['cases'][i] for i in ids]
source=source_hashes();(out/'sources.json').write_text(json.dumps(source,indent=2));(out/'panel.json').write_text(json.dumps(dict(panel,cases=cases),indent=2))
env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONPATH='.')
def slot_run(slot):
 rows=[]
 for c in cases[slot::2]:
  assert source_hashes()==source
  dest=out/c['id'];cmd=['.venv/bin/python','-m','scripts.run_walking_task','--plan',str(root/c['id']/'selection.json'),'--out',str(dest),'--timestep','.00025','--control-cpu',str(4*slot),'--worker-cpu',str(4*slot+2)]
  with (out/(c['id']+'.log')).open('w') as log:code=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,timeout=600).returncode
  if code:raise RuntimeError((c['id'],code))
  r=json.loads((dest/'result.json').read_text());old=json.loads((root/c['id']/'result.json').read_text())
  row=dict(id=c['id'],success=r['task_success'],original_success=old['task_success'],same_outcome=r['task_success']==old['task_success'],walking_ratio=r.get('rotor_burden_hover_ratio'),original_ratio=old.get('rotor_burden_hover_ratio'),failed_checks=[k for k,v in r.get('checks',{}).items() if not v],control=r.get('control'),command=cmd)
  rows.append(row);print(json.dumps(row),flush=True)
 return rows
with concurrent.futures.ThreadPoolExecutor(2) as pool:rows=sum(list(pool.map(slot_run,[0,1])),[])
report=dict(rows=rows,source_changes=source_hashes()!=source,all_outcomes_match=all(r['same_outcome'] for r in rows),scope='Predeclared saved-plan sensitivity at 0.25 ms vs 0.5 ms with newly measured control delays; excluded from fresh-generation success score.')
(out/'step_comparison.json').write_text(json.dumps(report,indent=2))
