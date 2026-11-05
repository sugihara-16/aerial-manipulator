"""Independent interval arithmetic, airborne geometry and continuous clock audit."""
import json,sys
from pathlib import Path
import numpy as np,mujoco
from amsrr_det.walking_plan import plant_for
from amsrr_det.grasp_geometry import robot_bottom
root=Path(sys.argv[1]);panel=json.loads((root/'panel.json').read_text());issues=[];rows=[];pending=[]
for case in panel['cases']:
    p=root/case['id']
    if not (p/'result.json').exists():pending.append(case['id']);continue
    result=json.loads((p/'result.json').read_text());total=result.get('whole_sequence')
    if total is None:
        rows.append(dict(id=case['id'],success=False,reason=result.get('reason')));continue
    phases=total['phases'];summed={k:sum(x[k] for x in phases.values()) for k in ['seconds','rotor_proxy','joint_proxy','positive_joint_work_j']}
    for key,target in [('seconds','simulation_s'),('rotor_proxy','rotor_proxy'),('joint_proxy','joint_proxy'),('positive_joint_work_j','positive_joint_work_j')]:
        if abs(summed[key]-total[target])>1e-6:issues.append([case['id'],'interval_sum',key,summed[key],total[target]])
    walk=phases['WALK']
    if 'simulation_s' in result:
        if abs(walk['seconds']-result['simulation_s'])>1e-8:issues.append([case['id'],'walking_interval'])
        if abs(walk['rotor_proxy']/walk['seconds']-result['rotor_burden_mean'])>1e-8:issues.append([case['id'],'walking_energy'])
    selection=json.loads((p/'selection.json').read_text());plan=selection['plan'];initial=np.load(p/'initial.npz')
    plant=plant_for(plan);d=mujoco.MjData(plant.mj);d.qpos[:]=initial['q'];d.qvel[:]=initial['v'];d.ctrl[:]=initial['ctrl'];mujoco.mj_forward(plant.mj,d)
    qp,_=plant.observation(d);bottom=robot_bottom(plant.pin,plant.meta,qp)
    if abs(bottom-.6)>1e-7 or np.linalg.norm(initial['v'])>1e-12:issues.append([case['id'],'airborne_initial_state',bottom])
    if np.linalg.norm(d.qacc)>1e-3:issues.append([case['id'],'initial_command_unbalanced',float(np.linalg.norm(d.qacc))])
    calls=json.loads((p/'calls.json').read_text());times=np.array([x['observation_s'] for x in calls]);indices=[i for i,x in enumerate(calls) if x['phase']=='WALK']
    if len(times)>1 and min(np.diff(times))<=0:issues.append([case['id'],'physical_clock_reset'])
    if indices and abs(times[indices[0]]-total['walk_start_s'])>1e-7:issues.append([case['id'],'walk_start_time'])
    if indices:
        samples=json.loads((p/'audit.json').read_text())
        if not samples or abs(samples[0]['time_s'])>1e-7:issues.append([case['id'],'walk_audit_origin'])
    hover=(plant.meta['mass']*9.80665)**1.5/np.sqrt(len(plant.meta['rotors']))
    if abs(total['rotor_burden_hover_ratio']-total['rotor_proxy']/total['simulation_s']/hover)>1e-8:issues.append([case['id'],'whole_energy_basis'])
    rows.append(dict(id=case['id'],modules=case['task']['modules'],success=result['task_success'],prefix_s=total['walk_start_s'],
        whole_hover_ratio=total['rotor_burden_hover_ratio'],walking_hover_ratio=result.get('rotor_burden_hover_ratio'),
        whole_rotor_proxy=total['rotor_proxy'],walking_rotor_proxy=walk['rotor_proxy'],whole_joint_proxy=total['joint_proxy'],walking_joint_proxy=walk['joint_proxy'],
        whole_positive_work_j=total['positive_joint_work_j'],walking_positive_work_j=walk['positive_joint_work_j'],
        prefix_rotor_proxy=total['rotor_proxy']-walk['rotor_proxy'],whole_to_walking_integral=total['rotor_proxy']/max(walk['rotor_proxy'],1e-9),
        whole_hover_at_walking_duration_ratio=total['rotor_proxy']/max(walk['seconds']*hover,1e-9),
        initial_bottom_m=bottom,initial_acceleration_norm=float(np.linalg.norm(d.qacc)),landing_support_s=total['landing_support_s']))
successful=[r for r in rows if r['success']]
report=dict(issues=issues,pending=pending,complete=not pending,rows=rows,successful_cases=len(successful),
    mean_whole_hover_ratio=float(np.mean([r['whole_hover_ratio'] for r in successful])) if successful else None,
    mean_walk_hover_ratio=float(np.mean([r['walking_hover_ratio'] for r in successful])) if successful else None,
    aggregate_whole_to_walking_integral=sum(r['whole_rotor_proxy'] for r in successful)/max(sum(r['walking_rotor_proxy'] for r in successful),1e-9),
    mean_whole_hover_at_walking_duration_ratio=float(np.mean([r['whole_hover_at_walking_duration_ratio'] for r in successful])) if successful else None,
    energy_basis='uncalibrated rotor thrust**1.5 integral, normalized to ideal equal-thrust hover over the same respective time; arithmetic mean over successful tasks')
(root/'sequence_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
