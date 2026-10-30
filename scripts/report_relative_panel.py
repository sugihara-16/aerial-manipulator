"""Independent endpoint, timing, input and subgroup audit of the fixed panel."""
import argparse,collections,hashlib,json,math
from pathlib import Path
import numpy as np
import mujoco
from amsrr_det.collision_policy import configure as configure_collision
from scripts.report_grasp_panel import failure_concentration
from build.load_saved_plan import load_saved_plan
from build.mit_drive import dock_mit_model
from build.contact_step_control import pad_gap_jacobians

parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);args=parser.parse_args()
root=args.root;assignment=json.loads((root/'assignment.json').read_text()) if (root/'assignment.json').exists() else {};host_times=collections.defaultdict(list);host_allocation=collections.defaultdict(list);panel=json.loads((root/'panel.json').read_text());issues=[];rows=[];groups=collections.defaultdict(lambda:collections.defaultdict(list));allocation_times=[];allocation_ages=[];held_ticks=0;times=[];phases=collections.defaultdict(list);misses=0;skipped=0
manifest=json.loads((root/'source_manifest.json').read_text())
for name,digest in manifest.items():
 p=Path(name)
 if not p.exists()or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:issues.append(['source_changed',name])
for case in panel['cases']:
 name=case['id'];folder=root/name;p=folder/'result.json'
 if not p.exists():issues.append(['missing_result',name]);r=dict(task_success=False,reason='missing_result')
 else:r=json.loads(p.read_text())
 passed=r['task_success'];row=dict(case=name,success=passed,reason=r['reason'],modules=r.get('modules'),failed_checks=[k for k,v in r.get('task_checks',{}).items()if not v]);rows.append(row)
 strata=dict(case['strata'],host=assignment.get(name,'local'),cohort='large30' if name.startswith('span_load_')else'regular120',modules=str(r.get('modules','unplanned')))
 for axis,value in strata.items():groups[axis][str(value)].append(row)
 if 'task_checks' not in r:continue
 request=json.loads((folder/'request.json').read_text())
 for key in ['source','destination','support_top','payload']:
  if request[key]!=case[key]:issues.append(['input_mismatch',name,key])
 config=json.loads((folder/'config.json').read_text());source=Path(config['source']); original_source=source
 if not source.exists():source=Path('artifacts/runs/control_tail150_v8_20261027')/name
 for rel,digest in json.loads((folder/'input_hashes.json').read_text()).items():
  data=(source/rel).read_bytes()
  if source != original_source and rel.endswith('scene.xml'):
   data=data.replace(b'/home/leus/amsrr_det/',b'/home/leus/amsrr_det_large30/')
  if hashlib.sha256(data).hexdigest()!=digest:issues.append(['saved_input_changed',name,rel])
 if config['timing']!='measured' or config['period_s']!=.01 or not config['all_robot_collisions']or config['auxiliary_stops'] or config['collision']!='shape-pairs':issues.append(['configuration',name])
 plan,_=load_saved_plan(source/'generated_plan');plant=dock_mit_model(plan.plant,kp=200.);m=plant.mj
 configure_collision(m)
 initial=np.load(folder/'initial.npz');final=np.load(folder/'final.npz')
 expected_open=plant.configuration(plan.open_q)
 if (folder/'opening_preparation.json').exists():
  from build.relative_task_control import ObjectRelativeControl
  from amsrr_det.task_geometry import prepare_opening as prepare
  m.opt.timestep=config['timestep_s'];m.opt.impratio=10000.;m.opt.ccd_tolerance=1e-12;m.opt.ccd_iterations=1000
  closed=np.load(source/'endpoint.npz')['q'];opening_links=[f'm{n}/'+('yaw_dock_mech'if a.startswith('yaw')else'pitch_dock_mech')+a[-1]for n,a in plan.surfaces]
  ctl=ObjectRelativeControl(plant,closed,opening_links,object_shape=plan.shape)
  expected_open,_,opening_info=prepare(ctl,closed,expected_open)
  recorded_opening=json.loads((folder/'opening_preparation.json').read_text())
  if opening_info['selected']!=recorded_opening['selected'] or opening_info['reason']!=recorded_opening['reason']:issues.append(['opening_selection_mismatch',name])
 if not np.allclose(initial['q'],expected_open,rtol=0.,atol=1e-10)or np.any(initial['v']):issues.append(['spawn_mismatch',name])
 d=mujoco.MjData(m);d.qpos[:]=final['q'];d.qvel[:]=final['v'];mujoco.mj_fwdPosition(m,d)
 oi=int(m.jnt_qposadr[m.joint('object').id]);ov=int(m.jnt_dofadr[m.joint('object').id])
 position=float(np.linalg.norm(d.qpos[oi:oi+3]-plan.destination[:3]));angle=2*math.acos(min(1.,abs(float(d.qpos[oi+3:oi+7]@plan.destination[[6,3,4,5]]))))
 links=[f'm{n}/'+('yaw_dock_mech'if a.startswith('yaw')else'pitch_dock_mech')+a[-1]for n,a in plan.surfaces]
 gaps,_=pad_gap_jacobians(m,d,links);a=r['audit']
 expected=dict(position=position<.03,orientation=angle<.2,linear_velocity=float(np.linalg.norm(d.qvel[ov:ov+3]))<.02,angular_velocity=float(np.linalg.norm(d.qvel[ov+3:ov+6]))<.1,released=bool(len(gaps)and min(gaps)>.02),release_settled=(a['time_since_gripper_contact_s']is not None and a['time_since_gripper_contact_s']>=2.),lifted=a['maximum_lift_clearance_m']>=.1,transport_clearance=(a['minimum_transport_clearance_m']is not None and a['minimum_transport_clearance_m']>=.1))
 for k,v in expected.items():
  if v!=r['task_checks'][k]:issues.append(['endpoint_check_mismatch',name,k])
 if passed and (not all(expected.values())or a['peak_effort_ratio']>1.000001 or a['peak_penetration_m']>.05):issues.append(['invalid_success',name])
 row.update(position_error_m=position,orientation_error_rad=angle,peak_effort_ratio=a['peak_effort_ratio'],peak_joint_speed=a['peak_joint_speed_rad_s'])
 calls=json.loads((folder/'control_calls.json').read_text());values=np.array([c.get('publication_cpu_s',c['calculation_s'])for c in calls]);times.extend(values);host_times[assignment.get(name,'local')].extend(values);misses+=int(np.count_nonzero(values>=.01));skipped+=sum(c['skipped_ticks']for c in calls)
 if len(calls)!=r['control']['calls']or abs(max(values)*1000-r['control']['maximum_ms'])>1e-9:issues.append(['timing_summary_mismatch',name])
 for c in calls:
  if c['application_s']is not None and c['application_s']+1e-8<c['observation_s']+c['calculation_s']:issues.append(['delay_not_applied',name]);break
  phases[c['phase']].append(c.get('publication_cpu_s',c['calculation_s']))
 row['maximum_control_ms']=float(max(values)*1000)
 if config.get('asynchronous'):
  acalls=json.loads((folder/'allocation_calls.json').read_text())
  events=json.loads((folder/'publication_events.json').read_text())if(folder/'publication_events.json').exists()else[]
  for event in events:
   if event['application_s']+1e-8 < event['observation_s']+event['calculation_s']:issues.append(['publication_delay_not_applied',name])
  event_by_source={round(e['source_observation_s'],7):e for e in events}
  held_ticks+=r['allocation']['held_ticks']
  tick_by_time={round(c['observation_s'],7):c for c in calls}
  for a in acalls:
   if abs(a['calculation_s']-sum(a[k] for k in ['dispatch_delay_s','queue_delay_s','worker_calculation_s','response_transport_s']))>1e-8:issues.append(['worker_latency_partition',name])
   allocation_times.append(a['calculation_s']);host_allocation[assignment.get(name,'local')].append(a['calculation_s'])
   if abs(a['available_s']-a['observation_s']-a['calculation_s'])>1e-8:issues.append(['worker_delay_mapping',name])
   if 'application_s' in a:
    t=a['application_s']
    if t+1e-8<a['available_s']:issues.append(['worker_applied_early',name])
    event=event_by_source.get(round(a['observation_s'],7))
    tick=tick_by_time.get(round(a['application_tick_s'],7))
    if event is not None:
     if abs(t-event['application_s'])>1e-8:issues.append(['publication_event_mismatch',name])
    elif tick is None or tick['application_s'] is None or abs(t-tick['application_s'])>1e-8:issues.append(['missing_publication_event',name])
    allocation_ages.append(t-a['observation_s'])
   elif 'application_tick_s' in a:
    t=a['application_tick_s'];tick=tick_by_time.get(round(t,7))
    if t+1e-8<a['available_s']:issues.append(['worker_applied_early',name])
    if tick is None or tick['application_s'] is None:issues.append(['missing_publisher_tick',name])
    else:allocation_ages.append(tick['application_s']-a['observation_s'])
   if a['error']:issues.append(['worker_error',name,a['error']])

def score(rr):return dict(successes=sum(r['success']for r in rr),total=len(rr),fraction=sum(r['success']for r in rr)/len(rr))
def timing(v):
 x=np.array(v)*1000
 return dict(calls=len(x),mean_ms=float(x.mean()),p99_ms=float(np.quantile(x,.99)),max_ms=float(x.max()),over_10ms=int(np.count_nonzero(x>=10)))
strata={k:{g:score(v)for g,v in vv.items()}for k,vv in groups.items()};overall=score(rows)
result=dict(overall=overall,scope='Same task-generated structures replayed with new measured-delay controller; not a fresh 150-task planning benchmark',issues=issues,control=timing(times),skipped_ticks=skipped,phases={k:timing(v)for k,v in phases.items()},strata=strata,failure_concentration=failure_concentration(strata,overall),rows=rows)
if allocation_times:
 result['background_allocation']=timing(allocation_times)
 result['observation_to_actuation']=timing(allocation_ages)
 result['held_ticks']=held_ticks
result['hosts']={h:dict(publisher=timing(v),allocation=timing(host_allocation[h]) if host_allocation[h] else None) for h,v in host_times.items()}
(root/'audit_summary.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:result[k]for k in ['overall','issues','control','skipped_ticks']},indent=2));print('concentration',result['failure_concentration']['clear_concentration'])
