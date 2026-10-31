"""Summarize observed failure order; do not infer a physical cause."""
import json,sys,collections
from pathlib import Path
import numpy as np
root=Path(sys.argv[1]);qualification={} if not (root/'audit_summary.json').exists() else {r['id']:r for r in json.loads((root/'audit_summary.json').read_text())['rows']};rows=[];contacts=collections.defaultdict(lambda:[0,0])
for case in json.loads((root/'cases.json').read_text()):
 f=root/case['id'];r=json.loads((f/'result.json').read_text()) if (f/'result.json').exists() else {'reason':'unexecuted','task_success':False}
 evaluation=qualification.get(case['id'],{'success':r['task_success'],'reason':r['reason']})
 if 'contacts' in r:contacts[str(r['contacts'])][0]+=int(evaluation['success']);contacts[str(r['contacts'])][1]+=1
 if evaluation['success']:continue
 row={'id':case['id'],'reason':evaluation['reason'],'modules':r.get('modules'),'contacts':r.get('contacts'),'planning_s':r.get('planning_s'),'failed_checks':[k for k,v in r.get('checks',{}).items() if not v]}
 if (f/'trace.json').exists():
  trace=json.loads((f/'trace.json').read_text());work=[x for x in trace if x['phase'] in ['LOAD','TURN','SETTLE']]
  row['final_angle_deg']=float(np.rad2deg(r['angle_rad']))
  for name,predicate in [('large_pose_error',lambda x:x['relative_position_error_m']>.1 or x['relative_angle_error_rad']>.3),('contact_gap_over_1cm',lambda x:any(y>.01 for y in x['pad_gaps']))]:
   found=next((x for x in work if predicate(x)),None)
   row['first_'+name]=None if found is None else {k:found[k] for k in ['time_s','phase','angle','relative_position_error_m','relative_angle_error_rad','pad_gaps']}
  row['numerical_fallback_samples']=sum(x['numerical_fallback'] for x in work);row['work_samples']=len(work)
  row['publication']=r['control'];row['allocation']=r['allocation']
 rows.append(row)
result=dict(failures=rows,by_contacts={k:dict(successes=v[0],executed=v[1])for k,v in contacts.items()},scope='Recorded observations and event order, not an identified common physical cause')
(root/'failure_observations.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
