"""Compare the fixed plans at two physical steps, with fresh measured delays."""
import json,sys,hashlib
from pathlib import Path
import numpy as np
base=Path(sys.argv[1]);target=Path(sys.argv[2]);panel=json.loads((base/'panel.json').read_text());rows=[];issues=[]
manifest=json.loads((target/'sources.json').read_text())
original=json.loads((base/'sources.json').read_text())
if manifest['runtime']!=original:issues.append('changed_runtime_source')
if hashlib.sha256(Path(manifest['private_runner']).read_bytes()).hexdigest()!=manifest['private_sha256']:issues.append('changed_halfstep_runner')
for c in panel['cases']:
 name=c['id'];old=json.loads((base/name/'result.json').read_text());r=json.loads((target/name/'result.json').read_text())
 if json.loads((base/name/'selection.json').read_text())!=json.loads((target/name/'selection.json').read_text()):issues.append([name,'changed_plan'])
 if r['timestep']!=.00025 or r['fresh_structure_generation']:issues.append([name,'invalid_replay_mode'])
 calls=json.loads((target/name/'calls.json').read_text());alloc=json.loads((target/name/'allocations.json').read_text())
 for a in alloc:
  if abs(a['available_s']-a['observation_s']-a['calculation_s'])>1e-8:issues.append([name,'invalid_delay'])
  if a.get('application_s') is not None and a['application_s']+1e-8<a['available_s']:issues.append([name,'early_application'])
 if any(a.get('application_s') is not None and a['application_s']+1e-8<a['observation_s']+a['calculation_s'] for a in calls):issues.append([name,'early_publication'])
 if r['task_success']!=all(r['checks'].values()) or r['peak_actuator_ratio']>1.00001:issues.append([name,'invalid_success_or_actuation'])
 rows.append(dict(id=name,normal_success=old['task_success'],half_success=r['task_success'],normal_ratio=old['rotor_burden_hover_ratio'],half_ratio=r['rotor_burden_hover_ratio'],relative_ratio_change=r['rotor_burden_hover_ratio']/old['rotor_burden_hover_ratio']-1,normal_joint_work_j=old['positive_joint_work_j'],half_joint_work_j=r['positive_joint_work_j'],checks=r['checks'],publication_max_ms=r['control']['max_ms'],publication_over10=r['control']['over10']))
passed=[r for r in rows if r['half_success']];common=[r for r in passed if r['normal_success']]
out=dict(cases=len(rows),successes=len(passed),mean_success_ratio=float(np.mean([r['half_ratio']for r in passed])),changed_outcomes=[r['id']for r in rows if r['normal_success']!=r['half_success']],common_success_max_relative_ratio_change=max(abs(r['relative_ratio_change'])for r in common),common_success_mean_ratio_change=float(np.mean([r['half_ratio']-r['normal_ratio']for r in common])),publication_max_ms=max(r['publication_max_ms']for r in rows),publication_over10=sum(r['publication_over10']for r in rows),issues=issues,rows=rows,meaning='Saved-plan physical-step sensitivity with unchanged control and newly measured delays; not new structure generation or planning-time evidence.')
(target/'step_comparison.json').write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items()if k!='rows'},indent=2))
