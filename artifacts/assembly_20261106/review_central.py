"""Independent engineering summary of saved assembly runs, without rerunning physics."""
import json,sys
from pathlib import Path
import numpy as np,mujoco
root=Path(sys.argv[1]);rows=json.load(open(root/'summary.json'))['results'];timing=[];handoffs=[];issues=[];nearest=[];observed=[]
for r in rows:
 p=root/r['case']
 if not (p/'physical.xml').exists():continue
 m=mujoco.MjModel.from_xml_path(str(p/'physical.xml'));qi=np.array([int(m.jnt_qposadr[j])for j in range(m.njnt)if m.jnt_type[j]==0]);z=np.load(p/'planned.npz');xy=z['initial'][qi[:,None]+np.arange(2)]
 if len(xy)>1:
  distance=np.linalg.norm(xy[:,None]-xy[None,:],axis=2);np.fill_diagonal(distance,np.inf);values=distance.min(axis=1);nearest.extend(values.tolist())
  if not np.all((values>=1.-1e-10)&(values<=2.+1e-10)):issues.append(r['case']+' initial spacing')
 times=json.load(open(p/'timing.json'));timing.extend(times['allocation']);observed.extend(x['publication_s']for x in times['calls']);events=json.load(open(p/'handoffs.json'));handoffs.extend(events)
 if r['peak_joint_fraction']>1.+1e-8:issues.append(r['case']+' actuator limit')
 for e in events:
  if e.get('first_output_difference',0.)>1e-10:issues.append(r['case']+' discontinuous first output')
def stats(x):
 return None if not x else dict(count=len(x),minimum=min(x),median=float(np.median(x)),p95=float(np.percentile(x,95)),p99=float(np.percentile(x,99)),maximum=max(x))
report=dict(total=len(rows),publication_s=stats(observed),worker_s=stats([x['worker_calculation_s']for x in timing]),
 observation_to_action_s=stats([x['application_s']-x['observation_s']for x in timing if 'application_s'in x]),
 dispatch_s=stats([x['dispatch_delay_s']for x in timing]),queue_s=stats([x['queue_delay_s']for x in timing]),
 transport_s=stats([x['response_transport_s']for x in timing]),initial_nearest_distance_m=stats(nearest),
 handoff_events=len(handoffs),handoff_first_difference_max=max((e.get('first_output_difference',0.)for e in handoffs),default=0.),
 successful_planning_s=stats([r['planning_s']for r in rows if r['success']]),successful_assembly_s=stats([r['assembly_s']for r in rows if r['success']]),
 recovery_cases=[dict(case=r['case'],retries=r.get('retries'),success=r['success'],reason=r['reason'])for r in rows if r.get('retries',0)],
 peak_air_loss=max((r.get('air_max_loss',0.)for r in rows),default=0.),issues=issues)
(root/'engineering_review.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
