import os,json,subprocess
from pathlib import Path
root=Path('artifacts/walking_face_20261109/full36_599');out=Path('artifacts/walking_face_20261109/half599');out.mkdir(exist_ok=True)
from amsrr_det.runtime_manifest import source_hashes
expected=json.loads((root/'sources.json').read_text());assert source_hashes()==expected
(out/'sources.json').write_text(json.dumps(expected,indent=2))
panel=json.loads((root/'panel.json').read_text());protocol=json.loads((root/'sensitivity_protocol.json').read_text());rows=[]
E=dict(os.environ,PYTHONPATH='.',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
indices=list(protocol['indices'])
for j,c in enumerate(panel['cases']):
 p=root/c['id']/'result.json'
 if p.exists() and json.loads(p.read_text()).get('face_alignment_feet'):
  if j not in indices:indices.append(j)
  break
for j,c in enumerate(panel['cases']):
 p=root/c['id']/'result.json'
 if c['task']['modules']==8 and p.exists() and json.loads(p.read_text()).get('face_alignment_feet'):
  if j not in indices:indices.append(j)
  break
(out/'panel.json').write_text(json.dumps(dict(panel,cases=[panel['cases'][i] for i in indices]),indent=2))
for i in indices:
 assert source_hashes()==expected
 c=panel['cases'][i];name=c['id'];selection=root/name/'selection.json';main=json.loads((root/name/'result.json').read_text());s=json.loads(selection.read_text())
 if s['status']!='accepted' or not s.get('plan'):
  rows.append(dict(id=name,main_success=main['task_success'],skipped='no accepted saved plan'));continue
 p=out/name
 with (out/(name+'.log')).open('w')as f:
  cmd=['taskset','-c','0,2','.venv/bin/python','-m','scripts.run_walking_task','--plan',str(selection),'--timestep','.00025','--out',str(p)]
  r=subprocess.run(cmd,env=E,stdout=f,stderr=subprocess.STDOUT,timeout=600)
 if not (p/'result.json').exists():raise RuntimeError((name,r.returncode))
 value=json.loads((p/'result.json').read_text());rows.append(dict(id=name,predefined=i in protocol['indices'],main_success=main['task_success'],half_success=value['task_success'],same=main['task_success']==value['task_success'],result=value));(out/'comparison.json').write_text(json.dumps(rows,indent=2));print(name,main['task_success'],value['task_success'],value.get('rotor_burden_hover_ratio'),flush=True)

assert source_hashes()==expected

requested=json.loads((out/"panel.json").read_text())
(out/"requested_panel.json").write_text(json.dumps(requested,indent=2))
(out/"panel.json").write_text(json.dumps(dict(requested,cases=[c for c in requested["cases"] if (out/c["id"]/"result.json").exists()]),indent=2))
