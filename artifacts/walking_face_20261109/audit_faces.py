"""Cross-check Dock roles and measured face directions in saved walking states.

Full-rate contact/angle evidence is recorded by the independent physics audit.
This reader separately recomputes orientations at the saved 100 ms states.
"""
import json,sys
from pathlib import Path
import numpy as np,mujoco
root=Path(sys.argv[1]);rows=[];issues=[]
for case in json.loads((root/'panel.json').read_text())['cases']:
 p=root/case['id'];rfile=p/'result.json'
 if not rfile.exists():issues.append([case['id'],'missing_result']);continue
 result=json.loads(rfile.read_text());sel=json.loads((p/'selection.json').read_text());plan=sel.get('plan')
 if plan is None or not (p/'physical.xml').exists():continue
 names=[f'm{n}/{port[:-1]}_dock_mech{port[-1]}' for n,port in plan['surfaces']]
 required=[name for name in names if '/pitch_dock_' in name]
 if result['face_alignment_feet']!=required:issues.append([case['id'],'wrong_role_mask'])
 m=mujoco.MjModel.from_xml_path(str(p/'physical.xml'));d=mujoco.MjData(m);weight=m.body_mass.sum()*abs(m.opt.gravity[2])
 ids=[m.body(name).id for name in names];aligned=np.array([name in required for name in names]);max_angle=0.;checked=0
 samples=json.loads((p/'audit.json').read_text());bytime={round(x['time_s'],6):x for x in samples};start=result['whole_sequence']['walk_start_s']
 for s in json.loads((p/'states.json').read_text()):
  if s['phase']!='WALK':continue
  sample=bytime.get(round(s['time_s']-start,6))
  if sample is None:continue
  loaded=(np.array(sample['foot_load_n'])>.02*weight)&aligned
  if not loaded.any():continue
  d.qpos[:]=s['q'];mujoco.mj_kinematics(m,d)
  angles=np.degrees(np.arccos(np.clip(-d.xmat[np.array(ids)[loaded],6],-1.,1.)))
  max_angle=max(max_angle,float(angles.max()));checked+=int(loaded.sum())
 if result['task_success'] and (max_angle>5. or not result['checks']['face_alignment'] or not result['checks']['landing_face_alignment']):issues.append([case['id'],'successful_case_face_violation'])
 rows.append(dict(id=case['id'],success=result['task_success'],female_feet=len(required),male_feet=len(names)-len(required),
  independent_loaded_states=checked,independent_max_angle_deg=max_angle,full_rate_max_angle_deg=result['maximum_loaded_face_angle_deg'],
  off_face_load_fraction=result['off_face_load_fraction'],landing=result['landing_face_alignment']))
report=dict(issues=issues,rows=rows,scope='Independent saved-pose orientation cross-check; does not replace the every-physics-step contact audit')
(root/'face_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(dict(issues=issues,executed=len(rows),female_cases=sum(x['female_feet']>0 for x in rows),successful_female_cases=sum(x['female_feet']>0 and x['success'] for x in rows)),indent=2))
