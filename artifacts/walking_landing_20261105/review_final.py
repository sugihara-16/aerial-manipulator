"""Summarize executed phases, actual steps, workload and remaining failures."""
import json,sys
from pathlib import Path
import numpy as np
root=Path(sys.argv[1]);panel=json.loads((root/'panel.json').read_text());rows=[]
for c in panel['cases']:
 p=root/c['id'];r=json.loads((p/'result.json').read_text());s=json.loads((p/'selection.json').read_text());plan=s.get('plan');total=r.get('whole_sequence')
 row=dict(id=c['id'],modules=c['task']['modules'],speed=c['task']['speed'],curvature=c['task']['curvature'],success=r['task_success'],reason=r.get('reason'),checks=r.get('checks'),planning_s=r.get('planning_s'),trials=len(s.get('trials',[])),evidence_complete=(s.get('evidence') or {}).get('complete'),refinement=s.get('refinement'))
 if total:
  row.update(feet=r['feet'],walking_proxy_ratio=r.get('rotor_burden_hover_ratio'),whole_proxy_ratio=total['rotor_burden_hover_ratio'],walking_joint_mean=r.get('joint_burden_mean'),walking_positive_work=r.get('positive_joint_work_j'),whole_joint_proxy=total['joint_proxy'],whole_positive_work=total['positive_joint_work_j'],landing_support_s=total['landing_support_s'],prefix_duration_s=total['walk_start_s'],peak_actuator_ratio=total['peak_actuator_ratio'],phases=total['phases'])
  valid=[x for x in r.get('steps',[]) if x['valid']]
  row['valid_step_stride_range_m']=[min(x['stride_m'] for x in valid),max(x['stride_m'] for x in valid)] if valid else None
  row['valid_steps_per_foot']=r.get('valid_steps_per_foot');row['actual_speed']=r.get('mean_speed_m_s')
  row['ground_load_fraction']=r.get('ground_load_fraction');row['other_load_fraction']=r.get('other_load_fraction');row['posture_error_p95_rad']=r.get('posture_error_p95_rad')
 rows.append(row)
by={}
for n in range(3,9):
 rs=[r for r in rows if r['modules']==n];yes=[r for r in rs if r['success']]
 by[n]=dict(successes=len(yes),total=len(rs),mean_walking_ratio=np.mean([r['walking_proxy_ratio'] for r in yes]),mean_whole_ratio=np.mean([r['whole_proxy_ratio'] for r in yes]),mean_joint_burden=np.mean([r['walking_joint_mean'] for r in yes]),mean_positive_joint_work_j=np.mean([r['walking_positive_work'] for r in yes]))
report=dict(rows=rows,by_modules=by,failures=[r for r in rows if not r['success']],scope='Fixed development panel, independently rerun from fresh generation; not unseen-task performance or calibrated electric power.')
(root/'engineering_review.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ['rows','failures']},indent=2))
