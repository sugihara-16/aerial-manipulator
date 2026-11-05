"""Complete full saved-plan step sensitivity without repeating the first six."""
import concurrent.futures,json,os,shutil,subprocess,sys
from pathlib import Path
from amsrr_det.runtime_manifest import source_hashes
root=Path(sys.argv[1]);first=Path(sys.argv[2]);out=Path(sys.argv[3]);out.mkdir(parents=True,exist_ok=False)
panel=json.loads((root/'panel.json').read_text());source=source_hashes();old=json.loads((first/'step_comparison.json').read_text());assert json.loads((first/'sources.json').read_text())==source and not old['source_changes']
rows=old['rows'];done={r['id'] for r in rows}
for r in rows:shutil.copytree(first/r['id'],out/r['id'])
(out/'sources.json').write_text(json.dumps(source,indent=2));(out/'panel.json').write_text(json.dumps(panel,indent=2))
cases=[c for c in panel['cases'] if c['id'] not in done];env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONPATH='.')
def run_slot(slot):
 result=[]
 for c in cases[slot::4]:
  assert source_hashes()==source
  original=json.loads((root/c['id']/'result.json').read_text());selection=json.loads((root/c['id']/'selection.json').read_text());dest=out/c['id']
  if selection.get('plan') is None:
   dest.mkdir();shutil.copyfile(root/c['id']/'selection.json',dest/'selection.json');shutil.copyfile(root/c['id']/'result.json',dest/'result.json')
   row=dict(id=c['id'],success=False,original_success=False,same_outcome=True,physical_replay=False,reason='No original plan; retain original generation failure, not a repeated generation test.')
  else:
   cmd=['.venv/bin/python','-m','scripts.run_walking_task','--plan',str(root/c['id']/'selection.json'),'--out',str(dest),'--timestep','.00025','--control-cpu',str(4*slot),'--worker-cpu',str(4*slot+2)]
   with (out/(c['id']+'.log')).open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,timeout=600,check=True)
   r=json.loads((dest/'result.json').read_text());row=dict(id=c['id'],physical_replay=True,success=r['task_success'],original_success=original['task_success'],same_outcome=r['task_success']==original['task_success'],walking_ratio=r.get('rotor_burden_hover_ratio'),original_ratio=original.get('rotor_burden_hover_ratio'),failed_checks=[k for k,v in r.get('checks',{}).items() if not v],control=r.get('control'),command=cmd)
  result.append(row);print(json.dumps(row),flush=True)
 return result
with concurrent.futures.ThreadPoolExecutor(4) as pool:rows+=sum(list(pool.map(run_slot,range(4))),[])
yes=[r for r in rows if r['success']];report=dict(rows=rows,source_changes=source_hashes()!=source,all_outcomes_match=all(r['same_outcome'] for r in rows),successes=len(yes),total=len(rows),mean_walking_ratio=sum(r['walking_ratio'] for r in yes)/len(yes),scope='All 35 saved executable plans replayed at 0.25 ms; original single no-plan failure retained. First six predeclared runs reused once, other 29 executed once. Newly measured delay, same source; excluded from fresh-generation score.')
(out/'step_comparison.json').write_text(json.dumps(report,indent=2))
