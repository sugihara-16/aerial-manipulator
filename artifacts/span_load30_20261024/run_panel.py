"""Run one frozen input panel with one common local-mechanics configuration."""
import argparse,json,subprocess,sys,time,threading,os,signal
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from scripts.report_grasp_panel import report

p=argparse.ArgumentParser();p.add_argument('--panel',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=3)
p.add_argument('--controlled-transitions',action='store_true');p.add_argument('--interval-control',action='store_true');p.add_argument('--generated-feedback',action='store_true')
p.add_argument('--momentum-feedback',action='store_true');p.add_argument('--finite-slip-preload',action='store_true');p.add_argument('--background-withdrawal',action='store_true')
p.add_argument('--task-priority-control',action='store_true')
p.add_argument('--bounded-allocation',action='store_true');p.add_argument('--hold-withdrawal-pose',action='store_true')
p.add_argument('--minimize-input',action='store_true');p.add_argument('--global-structure-search',action='store_true')
p.add_argument('--structure-workers',type=int,default=8)
p.add_argument('--structure-remote',type=Path)
p.add_argument('--defer-execution',action='store_true',help='finish all plans before sharing CPUs among independent simulations')
p.add_argument('--keep-going',action='store_true',help='finish the entire panel even when repeated planning deadlines occur')
p.add_argument('--execution-timeout',type=float,default=None,help='wall seconds per execution, excluding planning and queue wait')
p.add_argument('--planning-watchdog',type=float,default=None,help='hard process wall limit before a plan is ready; the inner acceptance limit remains 15 s')
p.add_argument('--structure-generator',choices=['legacy','task','mixed'],default='task')
p.add_argument('--steps',type=int,default=40,help='physical substeps per 10 ms; control periods are unchanged')
p.add_argument('--trial-seed',type=int,help='fix the independent structure-search seed for every panel case')
p.add_argument('--governed-motion',action='store_true')
p.add_argument('--serialized-planning',action='store_true',help='benchmark scheduling: overlap execution but start only one planner at a time')
p.add_argument('--supported-load-transfer',action='store_true');p.add_argument('--preload-floor',action='store_true')
p.add_argument('--planned-preload-floor',action='store_true')
p.add_argument('--observed-handoff',action='store_true')
p.add_argument('--inertial-input-updates',action='store_true');p.add_argument('--nearby-handoff',action='store_true');p.add_argument('--contact-coupled',action='store_true')
p.add_argument('--finite-acquisition-support',action='store_true');p.add_argument('--process-withdrawal',action='store_true');p.add_argument('--normal-tracking',action='store_true');p.add_argument('--internal-tracking',action='store_true');a=p.parse_args()
if (a.momentum_feedback or a.finite_slip_preload or a.background_withdrawal)and not a.generated_feedback:p.error('additional feedback options require generated feedback')
panel=json.loads(a.panel.read_text());a.output.mkdir(exist_ok=False);(a.output/'requests').mkdir()
if a.trial_seed is not None:
 trial_seeds={case['id']:a.trial_seed+i for i,case in enumerate(panel['cases'])}
 (a.output/'trial_seeds.json').write_text(json.dumps(trial_seeds,indent=2))
else:trial_seeds={}
(a.output/'panel.json').write_text(json.dumps(panel,indent=2));(a.output/'runner.py').write_bytes(Path(__file__).read_bytes())
(a.output/'run_limits.json').write_text(json.dumps(dict(keep_going=a.keep_going,execution_timeout_s=a.execution_timeout,planning_acceptance_s=15.,planning_process_watchdog_s=a.planning_watchdog),indent=2))
planning_lock=threading.Lock()
prepared_processes={}

def time_out(process,out,reason,elapsed):
 # Only this condition's process group is stopped; the panel continues.
 try:os.killpg(process.pid,signal.SIGTERM)
 except ProcessLookupError:pass
 try:process.wait(timeout=3.)
 except subprocess.TimeoutExpired:
  try:os.killpg(process.pid,signal.SIGKILL)
  except ProcessLookupError:pass
  process.wait()
 path=out/'result.json'
 if path.exists():return
 info=json.loads((out/'experiment.json').read_text())if(out/'experiment.json').exists()else{}
 ready=json.loads((out/'execution_ready.json').read_text())if(out/'execution_ready.json').exists()else{}
 result=dict(**info,task_success=False,reason=reason,wall_s=elapsed,initial_planning_s=ready.get('initial_planning_s'))
 plan=out/'generated_plan/plan.json'
 if plan.exists():result['modules']=json.loads(plan.read_text())['model']['modules']
 for line in (a.output/(out.name+'.log')).read_text(errors='replace').splitlines():
  try:item=json.loads(line)
  except json.JSONDecodeError:continue
  if isinstance(item,dict):
   if 'time_s'in item:result['simulation_s']=item['time_s']
   if 'phase'in item:result['phase']=item['phase']
 out.mkdir(exist_ok=True);path.write_text(json.dumps(result,indent=2))

def launch(case):
 name=case['id'];request=a.output/'requests'/(name+'.json');request.write_text(json.dumps(case,indent=2));out=a.output/name;began=time.perf_counter()
 command=[sys.executable,'-m','build.check_local_task','--request',str(request),'--output',str(out),'--predictive-motion','--distance-closing','--condensed-solve','--steps',str(a.steps)]
 if name in trial_seeds:command+=['--structure-seed',str(trial_seeds[name])]
 if a.structure_remote:command+=['--structure-remote',str(a.structure_remote)]
 if a.defer_execution:command+=['--execution-gate',str(a.output/(name+'.start'))]
 if a.generated_feedback:
  command+=['--generated-feedback','--staged-approach','--hybrid-withdrawal']
  if a.momentum_feedback:command+=['--momentum-feedback']
  if a.finite_slip_preload:command+=['--finite-slip-preload']
  if a.minimize_input:command+=['--minimize-input']
  if a.global_structure_search:command+=['--global-structure-search','--structure-workers',str(a.structure_workers),'--structure-generator',a.structure_generator]
  if a.governed_motion:command+=['--governed-motion']
  if a.background_withdrawal:command+=['--background-withdrawal']
  if a.bounded_allocation:command+=['--bounded-allocation']
  if a.hold_withdrawal_pose:command+=['--hold-withdrawal-pose']
  if a.supported_load_transfer:command+=['--supported-load-transfer']
  if a.preload_floor:command+=['--preload-floor']
  if a.observed_handoff:command+=['--observed-handoff']
  if a.inertial_input_updates:command+=['--inertial-input-updates']
  if a.nearby_handoff:command+=['--nearby-handoff']
  if a.contact_coupled:command+=['--contact-coupled']
  if a.task_priority_control:command+=['--task-priority-control']
  if a.planned_preload_floor:command+=['--planned-preload-floor']
  if a.finite_acquisition_support:command+=['--finite-acquisition-support']
  if a.process_withdrawal:command+=['--process-withdrawal']
  if a.normal_tracking:command+=['--normal-tracking']
  if a.internal_tracking:command+=['--internal-tracking']
 elif a.interval_control:
  command[2]='build.run_active_task';command+=['--interval-motion','--interval-acquisition','--staged-approach','--hybrid-withdrawal']
 elif a.controlled_transitions:
  command[2]='build.run_active_task';command+=['--constrained-load','--constrained-motion','--staged-approach','--planned-withdrawal']
 log=(a.output/(name+'.log')).open('w')
 process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 marker=out/('execution_ready.json' if a.defer_execution else 'endpoint.npz')
 if a.serialized_planning or a.defer_execution:
  while process.poll() is None and not marker.exists():
   if a.planning_watchdog is not None and time.perf_counter()-began>a.planning_watchdog:
    time_out(process,out,'planning_process_time_limit',time.perf_counter()-began);break
   time.sleep(.05)
 return process,log,began,time.perf_counter()


def run(case):
 name=case['id'];out=a.output/name;request=a.output/'requests'/(name+'.json')
 queued=time.perf_counter()
 if a.defer_execution:
  process,log,began,ready_time=prepared_processes[name]
  (a.output/(name+'.start')).touch()
  queue_wait=time.perf_counter()-ready_time
 else:
  with planning_lock if a.serialized_planning else nullcontext():
   process,log,began,_=launch(case)
  queue_wait=began-queued
 execution_started=time.perf_counter()
 try:code=process.wait(timeout=a.execution_timeout)
 except subprocess.TimeoutExpired:
  time_out(process,out,'execution_wall_time_limit',time.perf_counter()-execution_started);code=process.returncode
 log.close()
 result_path=out/'result.json'
 if not result_path.exists():
  info=json.loads((out/'experiment.json').read_text())if(out/'experiment.json').exists()else{}
  result=dict(**info,task_success=False,reason='process_error',returncode=code)
  out.mkdir(exist_ok=True);result_path.write_text(json.dumps(result,indent=2));(out/'request.json').write_text(json.dumps(case,indent=2))
 result=json.loads(result_path.read_text())
 row=dict(case=name,returncode=code,benchmark_queue_wait_s=queue_wait,process_wall_s=time.perf_counter()-began-queue_wait,**{key:result.get(key)for key in ['task_success','reason','contacts','modules','motion_methods','initial_planning_s']})
 print(json.dumps(row),flush=True);return row

started=time.perf_counter()
if a.defer_execution:
 planning_timeouts=0;consecutive_timeouts=0
 for case in panel['cases']:
  prepared_processes[case['id']]=launch(case)
  result_path=a.output/case['id']/'result.json'
  ready=a.output/case['id']/'execution_ready.json'
  if ready.exists():
   consecutive_timeouts=0
   print(json.dumps(dict(case=case['id'],prepared=json.loads(ready.read_text()))),flush=True)
  elif result_path.exists():
   result=json.loads(result_path.read_text())
   print(json.dumps(dict(case=case['id'],planning_failure=result.get('reason'),seconds=result.get('initial_planning_s'))),flush=True)
   if result.get('reason')=='initial_planning_deadline':
    planning_timeouts+=1;consecutive_timeouts+=1
   else:consecutive_timeouts=0
   # An isolated timeout remains a failed case. Repeated timeouts stop
   # validation before spending hours on dynamics. Neither changes 15 s.
   if not a.keep_going and (planning_timeouts>=3 or consecutive_timeouts>=2):
    import os,signal
    for process,log,_,_ in prepared_processes.values():
     if process.poll() is None:os.killpg(process.pid,signal.SIGTERM)
     process.wait();log.close()
    (a.output/'interrupted.json').write_text(json.dumps(dict(reason='repeated planning deadlines; fix before extending panel', planning_timeouts=planning_timeouts, consecutive_timeouts=consecutive_timeouts)))
    raise SystemExit(2)
with ThreadPoolExecutor(max_workers=a.workers)as pool:rows=list(pool.map(run,panel['cases']))
(a.output/'timing.json').write_text(json.dumps(dict(wall_s=time.perf_counter()-started,workers=a.workers,serialized_planning=a.serialized_planning,rows=rows),indent=2))
subprocess.run([sys.executable,'artifacts/span_load30_20261024/analyze_panel.py',str(a.output)],check=True)
