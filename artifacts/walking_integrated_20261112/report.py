"""Audit integrated outcomes, task motion, latency records and frozen inputs."""
import collections,hashlib,json
from pathlib import Path
import numpy as np
import mujoco
from amsrr_det.walking import WalkSpec
from amsrr_det.task_spec import make_spec,defaults
from amsrr_det.runtime_manifest import source_hashes
from scripts.report_grasp_panel import failure_concentration
from scripts.report_walking_panel import energy_summary
root=Path('artifacts/walking_integrated_20261112/full36')
panel=json.loads((root/'panel.json').read_text());summary=json.loads((root/'summary.json').read_text())
issues=[];rows=[];groups=collections.defaultdict(lambda:collections.defaultdict(list));publications=[];ages=[];stage_max=collections.defaultdict(list)
source=json.loads((root/'sources.json').read_text())
for name,digest in source.items():
 if hashlib.sha256((root/'source_snapshot'/name).read_bytes()).hexdigest()!=digest:issues.append(['snapshot',name])
 if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:issues.append(['source_changed',name])
for case in panel['cases']:
 folder=root/case['id'];f=folder/'result.json'
 if not f.exists():continue
 r=json.loads(f.read_text());task=case['task'];raw=r.get('engine_result',{})
 inp=json.loads((folder/'input.json').read_text());v=defaults('walking');v.update(modules=task['modules'],speed=task['speed'],distance=task['distance'],curvature=task['curvature'],seed=case['seed'])
 if inp!=make_spec('walking',v,'integrated'):issues.append(['input',case['id']])
 if (folder/'task.json').exists() and json.loads((folder/'task.json').read_text())!=inp:issues.append(['executed_input',case['id']])
 row=dict(id=case['id'],success=bool(r['success']),reason=r.get('reason'),modules=task['modules'],task=task,preparation_s=r.get('preparation_s'),failed_checks=[k for k,val in raw.get('checks',{}).items() if not val],control=r.get('control'),feet=raw.get('feet'))
 sel=folder/'execution/selection.json'
 if sel.exists():
  s=json.loads(sel.read_text())
  if s['request']!=task or s['seed']!=case['seed']:issues.append(['generation_input',case['id']])
  row['generation_complete']=(s.get('evidence') or {}).get('complete');row['generation_status']=s['status']
  if row['success'] and not row['generation_complete']:issues.append(['incomplete_selection',case['id']])
 assembly=r.get('assembly')
 if assembly:
  row.update(assembly_success=assembly['success'],assembly_s=assembly.get('assembly_s'))
  if row['success'] and (not assembly['success'] or assembly.get('assembly_s',1e9)>30 or assembly['latched']!=assembly['required']):issues.append(['assembly_success',case['id']])
 for stage,entry in r.get('control',{}).get('stages',{}).items():stage_max[stage].append(entry['max_ms'])
 if raw.get('checks') and (folder/'execution/calls.json').exists():
  if not raw['fresh_structure_generation']:issues.append(['saved_plan',case['id']])
  if row['success'] and (not all(raw['checks'].values()) or r['preparation_s']>15 or r['control']['over10']):issues.append(['invalid_success',case['id']])
  calls=json.loads((folder/'execution/calls.json').read_text());alloc=json.loads((folder/'execution/allocations.json').read_text())
  cpu=np.array([c['publication_cpu_s'] for c in calls]);publications.extend(cpu)
  if abs(cpu.max()*1000-raw['control']['max_ms'])>1e-8:issues.append(['publication_max',case['id']])
  for c in calls:
   if c['application_s'] is not None and c['application_s']+1e-8<c['observation_s']+c['calculation_s']:issues.append(['early_output',case['id']]);break
  for c in alloc:
   if abs(c['available_s']-c['observation_s']-c['calculation_s'])>1e-8:issues.append(['delay_mapping',case['id']]);break
   if c.get('application_s') is not None:
    if c['application_s']+1e-8<c['available_s']:issues.append(['early_allocation',case['id']]);break
    ages.append(c['application_s']-c['observation_s'])
  samples=json.loads((folder/'execution/audit.json').read_text());spec=WalkSpec(**task)
  if samples:
   times=np.array([a['time_s'] for a in samples]);xy=np.array([a['com'][:2] for a in samples]);origin=np.array(s['plan']['gait']['origin']);k=spec.curvature;relative=xy-origin[:2]
   progress=relative[:,0] if abs(k)<1e-10 else np.unwrap(np.arctan2(k*relative[:,0],1-k*relative[:,1]))/k
   distance=progress[-1]-progress[0];mean=distance/(times[-1]-times[0]);error=np.linalg.norm(xy-origin[:2]-spec.path(times)[0],axis=1)
   boundaries=np.r_[np.arange(0,times[-1],5.),times[-1]];windows=np.diff(np.interp(boundaries,times,progress))/np.diff(boundaries)
   independent=dict(distance=bool(distance>=spec.distance),minimum_speed=bool(distance/raw['simulation_s']>=.2),mean_speed=bool(abs(mean/spec.speed-1)<=.1),window_speed=bool(max(abs(windows/spec.speed-1))<=.2),path=bool(error.max()<=.1 and error[-1]<=.05),control_deadline=bool(cpu.max()<.01 and max(np.diff([c['observation_s'] for c in calls]))<=.01+1e-8),planning_deadline=bool(raw['planning_s']<=15.))
   if any(raw['checks'][key]!=value for key,value in independent.items()):issues.append(['task_check_mismatch',case['id'],independent])
   m=mujoco.MjModel.from_xml_path(str(folder/'execution/physical.xml'));d=mujoco.MjData(m);final=np.load(folder/'execution/final.npz');d.qpos[:]=final['q'];d.qvel[:]=final['v'];mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
   if np.linalg.norm(d.subtree_com[0]-np.array(samples[-1]['com']))>1e-8:issues.append(['final_com',case['id']])
   hover=(m.body_mass.sum()*abs(m.opt.gravity[2]))**1.5/np.sqrt(4*task['modules']);ratio=raw['rotor_burden_mean']/hover
   if abs(ratio-raw['rotor_burden_hover_ratio'])>1e-8:issues.append(['energy',case['id']])
   row.update(distance_m=float(distance),mean_speed_m_s=float(mean),maximum_path_error_m=float(error.max()),endpoint_path_error_m=float(error[-1]),rotor_burden_hover_ratio=float(ratio),landing_and_walk_hover_ratio=raw.get('whole_sequence',{}).get('rotor_burden_hover_ratio'),ground_load_fraction=raw['ground_load_fraction'],face_alignment=raw.get('face_alignment'),landing_face_alignment=raw.get('landing_face_alignment'))
 rows.append(row)
 strata=dict(modules=task['modules'],speed=task['speed'],path='straight' if task['curvature']==0 else 'left' if task['curvature']>0 else 'right',module_speed=str((task['modules'],task['speed'])))
 for axis,key in strata.items():groups[axis][str(key)].append(row)
def score(rs):return dict(successes=sum(r['success'] for r in rs),total=len(rs),fraction=sum(r['success'] for r in rs)/len(rs) if rs else 0.)
overall=score(rows);strata={axis:{key:score(rs) for key,rs in gr.items()} for axis,gr in groups.items()};bias=failure_concentration(strata,overall)
energy=energy_summary(rows);energy['by_modules']={key:energy_summary(rs) for key,rs in groups['modules'].items()}
def timing(xs):
 a=np.array(xs)*1000
 return dict(count=len(xs),max_ms=float(a.max()),p99_ms=float(np.quantile(a,.99)),over10=int(sum(a>=10))) if len(xs) else None
initial=[r['preparation_s'] for r in rows if r['preparation_s'] is not None]
complete=len(rows)==len(panel['cases']);target=overall['fraction']>=.8 and not bias['clear_concentration']
report=dict(complete=complete,overall=overall,strata=strata,rows=rows,issues=issues,failure_concentration=bias,energy=energy,initial=dict(max_s=max(initial,default=0),over15=sum(t>15 for t in initial)),task_publication=timing(publications),new_observation_application=timing(ages),maximum_ms_by_stage={k:max(v) for k,v in stage_max.items()},success_and_statistical_bias_pass=complete and not issues and target,energy_pass=energy['valid'] and energy['mean']<.4,mechanical_failure_review_required=True,source_changes=summary['source_changes'])
(root/'audit_summary.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ['rows','failure_concentration']},indent=2))
