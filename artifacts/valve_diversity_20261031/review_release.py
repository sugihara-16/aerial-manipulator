"""Sampled release response from actual physical states, for engineering review."""
import json,sys
from pathlib import Path
import numpy as np,mujoco
from amsrr_det.collision_policy import configure
from amsrr_det.valve_search import links_for
from build.contact_step_control import pad_gap_jacobians
root=Path(sys.argv[1]);rows=[]
folders=sorted(p.parent for p in root.glob('*/result.json')) if (root/'cases.json').exists() else [root]
for f in folders:
 if not (f/'physical_states.npz').exists():continue
 r=json.loads((f/'result.json').read_text());s=json.loads((f/'selection.json').read_text());states=np.load(f/'physical_states.npz');trace=json.loads((f/'physical_trace.json').read_text());control=json.loads((f/'trace.json').read_text())
 m=mujoco.MjModel.from_xml_path(str(f/'physical.xml'));configure(m);m.opt.impratio=10000.;m.opt.ccd_tolerance=1e-12;m.opt.ccd_iterations=1000;d=mujoco.MjData(m);links=links_for(s['plan']['surfaces']);samples=[]
 for i,t in enumerate(trace):
  if t['phase'] not in ['OPEN','RETREAT']:continue
  d.qpos[:]=states['q'][i];mujoco.mj_fwdPosition(m,d);gaps,_=pad_gap_jacobians(m,d,links)
  samples.append(dict(time_s=t['time_s'],minimum_gap_m=float(min(gaps)),speed_m_s=float(np.linalg.norm(states['v'][i,:3])),angular_speed_rad_s=float(np.linalg.norm(states['v'][i,3:6]))))
 separated=next((i for i,x in enumerate(samples) if x['minimum_gap_m']>.002),None);after=[] if separated is None else samples[separated:]
 integrals=[x['flight_error_integral_world'] for x in control if 'flight_error_integral_world' in x and x['phase'] in ['OPEN','RETREAT']]
 row=dict(id=f.name,success=r['task_success'],first_separated_sample_s=None if separated is None else samples[separated]['time_s'],recontact_samples=sum(x['minimum_gap_m']<0 for x in after),maximum_speed_after_separation_m_s=max((x['speed_m_s']for x in after),default=None),maximum_angular_speed_after_separation_rad_s=max((x['angular_speed_rad_s']for x in after),default=None),maximum_absolute_integral_world=None if not integrals else np.max(abs(np.array(integrals)),axis=0).tolist(),final_checks=r['checks']);rows.append(row);print(json.dumps(row),flush=True)
(root/'release_response_review.json').write_text(json.dumps(dict(scope='Physical states sampled every 0.1 s; separation threshold 2 mm is a diagnostic event, not the unchanged 2 cm success criterion',rows=rows),indent=2))
