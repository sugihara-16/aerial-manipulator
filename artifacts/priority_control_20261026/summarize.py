"""Audit saved completion conditions and compare saved examples, no simulation."""
import json,hashlib,math,sys
from pathlib import Path
root=Path(sys.argv[1]);old=Path('artifacts/runs/span_load30_v10_20261026')
pairs={'box040':'span_load_box_040','cylinder040':'span_load_cylinder_040','prism045':'span_load_triangular_prism_045'}
rows=[];issues=[];hashes=[];configs=[]
for name,source in pairs.items():
 p=root/name/'result.json'
 if not p.exists():raise ValueError('missing '+name)
 new=json.loads(p.read_text());before=json.loads((old/source/'result.json').read_text());audit=new['audit'];bad=[]
 if not new['task_success']:bad.append(new['reason'])
 for key,limit in {'peak_joint_speed_rad_s':3.00001,'joint_limit_violation_rad':1e-5,'peak_effort_ratio':1.000001}.items():
  if not math.isfinite(audit[key]) or audit[key]>limit:bad.append(key)
 if audit['unexpected_contacts']:bad.append('unexpected_contacts')
 if audit['minimum_part_gap_capped_m'] < -1e-5 or audit['minimum_environment_gap_capped_m'] < -1e-5:bad.append('geometric_collision')
 if audit['minimum_transport_clearance_m'] is None or audit['minimum_transport_clearance_m']<.1:bad.append('transport_clearance')
 if audit['time_since_gripper_contact_s'] is None or audit['time_since_gripper_contact_s']<2.:bad.append('release_dwell')
 for key,limit in [('object_position_error_m',.03),('object_orientation_error_rad',.2)]:
  if not math.isfinite(new[key]) or new[key]>=limit:bad.append(key)
 for values,limit in [(new['object_velocity'][:3],.02),(new['object_velocity'][3:],.1)]:
  speed=math.sqrt(sum(x*x for x in values))
  if not math.isfinite(speed) or speed>=limit:bad.append('final_speed')
 if new['phase']!='RETREAT':bad.append('completion_phase')
 scene=(root/name/'generated_plan/scene.xml').read_bytes();old_scene=(old/source/'generated_plan/scene.xml').read_bytes()
 if scene!=old_scene:bad.append('scene_changed')
 if json.loads((root/name/'request.json').read_text())!=json.loads((old/source/'request.json').read_text()):bad.append('task_changed')
 for phase in ['CARRY','UNLOAD']:
  new['phase_control_time_s'][phase]['old_mean_s']=before['phase_control_time_s'][phase]['mean']
 rows.append(dict(case=name,source=str(old/source),modules=new['modules'],success=new['task_success'],
   old_wall_s=before['wall_s'],new_wall_s=new['wall_s'],wall_speedup=before['wall_s']/new['wall_s'],simulation_s=new['simulation_s'],
   control=new['phase_control_time_s'],peak_effort_ratio=audit['peak_effort_ratio'],peak_joint_speed_rad_s=audit['peak_joint_speed_rad_s'],
   peak_force_n=audit['peak_contact_force_n'],peak_moment_nm=audit['peak_contact_moment_nm'],
   secondary_violations=new['secondary_safety_violations'],position_error_m=new['object_position_error_m'],orientation_error_rad=new['object_orientation_error_rad'],
   identical_saved_scene=scene==old_scene,scene_sha256=hashlib.sha256(scene).hexdigest(),issues=bad))
 if bad:issues.append(dict(case=name,issues=bad))
 hashes.append(json.loads((root/name/'source_snapshot/manifest.json').read_text()));configs.append(new['execution_configuration'])
summary=dict(scope='Three previously successful saved structures; changed safety priority, no new structural search or unseen-case success rate',
 successes=sum(r['success'] for r in rows),total=len(rows),issues=issues,
 same_sources=all(x==hashes[0] for x in hashes),same_configuration=all(x==configs[0] for x in configs),
 timing_caveat='Historical v10 used 6 concurrent tasks plus development jobs; current run uses 3. Wall ratios are historical comparisons, not isolated implementation speedups. Simulation waits for each control computation.',
 physics_timestep_s=.000125,control_period_s=.01,rows=rows)
(root/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps({k:v for k,v in summary.items() if k!='rows'}));
for row in rows:print(row['case'],row['success'],round(row['new_wall_s'],3),row['issues'])
