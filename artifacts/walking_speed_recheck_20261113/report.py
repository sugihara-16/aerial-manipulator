import json,hashlib
from pathlib import Path
root=Path('artifacts/walking_speed_recheck_20261113');summary=json.loads((root/'summary.json').read_text());issues=[];rows=summary['rows']
source=json.loads((root/'sources.json').read_text())
for p,h in source.items():
 if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h:issues.append(['source_changed',p])
for row in rows:
 p=root/f"repeat{row['repetition']}"/row['case'];old=Path('artifacts/walking_integrated20_20261112/full36')/row['case']
 if (p/'input.json').read_bytes()!=(old/'input.json').read_bytes():issues.append(['input_changed',str(p)])
 s=json.loads((p/'execution/selection.json').read_text());old_s=json.loads((old/'execution/selection.json').read_text())
 if s['request']!=old_s['request'] or s['seed']!=old_s['seed']:issues.append(['generation_input',str(p)])
 row['comparison_complete']=(s.get('evidence')or{}).get('complete',False)
 row['refinement_s']=(s.get('refinement')or{}).get('seconds')
 q=p/'sequence_preparation.json';row['assembly_preparation_s']=json.loads(q.read_text())['seconds'] if q.exists() else None
by_case=[]
for name in json.loads((root/'protocol.json').read_text())['cases']:
 rs=[r for r in rows if r['case']==name]
 if not rs:continue
 by_case.append(dict(case=name,old_generation_s=rs[0]['previous_generation_s'],old_reason=rs[0]['previous_reason'],old_preparation_s=rs[0]['previous_preparation_s'],new_generation_s=[r['generation_s'] for r in rs],new_preparation_s=[r['preparation_s'] for r in rs],new_refinement_s=[r['refinement_s'] for r in rs],new_assembly_preparation_s=[r['assembly_preparation_s'] for r in rs]))
a=dict(complete=len(rows)==12,ready=sum(x['ready'] for x in rows),within20=sum(x['within20'] for x in rows),completed_comparisons=sum(x['comparison_complete'] for x in rows),issues=issues,by_case=by_case,scope='Initial preparation only; deliberately stopped at ready; no task execution or task success rate measured')
(root/'audit.json').write_text(json.dumps(a,indent=2));print(json.dumps(a,indent=2))
