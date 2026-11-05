"""Read-only accounting of generation coverage and physical walking outcomes."""
import json,sys,collections
from pathlib import Path
import numpy as np

root=Path(sys.argv[1]);panel=json.loads((root/'panel.json').read_text())
proposals=collections.Counter();accepted=collections.Counter();chosen=collections.Counter()
rows=[];issues=[];all_steps=[];by_modules=collections.defaultdict(list)
for case in panel['cases']:
 folder=root/case['id'];r=json.loads((folder/'result.json').read_text())
 s=json.loads((folder/'selection.json').read_text())
 proposals.update(s.get('fitted_proposal_foot_counts',{}).values())
 accepted.update(t['legs'] for t in s['trials'] if t['status']=='accepted')
 indices=[t['index'] for t in s['trials']]
 if len(indices)!=len(set(indices)):issues.append([case['id'],'duplicate_trial'])
 if r['task_success'] and sorted(indices)!=list(range(59)):issues.append([case['id'],'incomplete_trials'])
 if s.get('plan'):
  chosen[len(s['plan']['surfaces'])]+=1
  costs=[t['cost'] for t in s['trials'] if t['status']=='accepted']
  if s['plan']['cost']>min(costs)+1e-9:issues.append([case['id'],'not_minimum_accepted_cost'])
 row=dict(id=case['id'],modules=case['task']['modules'],success=r['task_success'],
     failure=r.get('reason'),failed_checks=[k for k,v in r.get('checks',{}).items() if not v],
     completed_trials=len(indices),planning_s=r['planning_s'])
 if 'checks' in r:
  row.update(feet=r['feet'],ratio=r['rotor_burden_hover_ratio'],mean_speed=r['mean_speed_m_s'],
      joint_burden=r['joint_burden_mean'],joint_work_j=r['positive_joint_work_j'],
      steps_per_foot=r['valid_steps_per_foot'],ground_load=r['ground_load_fraction'],
      actuator_peak_ratio=r['peak_actuator_ratio'],motion=r['actuator_motion'],
      foot_dominant_fraction=r['foot_dominant_fraction'],posture=r['posture_error_p95_rad'])
  if r['task_success']:
   all_steps.extend(t for t in r['steps'] if t['valid'])
 rows.append(row);by_modules[row['modules']].append(row)
def summary(rs):
 success=[r for r in rs if r['success']]
 return dict(total=len(rs),success=len(success),
     mean_ratio=float(np.mean([r['ratio'] for r in success])) if success else None,
     mean_joint_burden=float(np.mean([r['joint_burden'] for r in success])) if success else None,
     mean_positive_joint_work_j=float(np.mean([r['joint_work_j'] for r in success])) if success else None)
report=dict(overall=summary(rows),by_modules={k:summary(v) for k,v in by_modules.items()},
     proposal_feet=dict(proposals),accepted_feet=dict(accepted),selected_feet=dict(chosen),
     issues=issues,rows=rows,meaning='Actual selected feet and generated proposals are separate; motor effort and joint-stop reaction are separate; energy is an uncalibrated rotor proxy.',
     successful_step_stride_range_m=[min(t['stride_m'] for t in all_steps),max(t['stride_m'] for t in all_steps)],
     successful_clearance_range_m=[min(t['clearance_m'] for t in all_steps),max(t['clearance_m'] for t in all_steps)])
(root/'engineering_review.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
