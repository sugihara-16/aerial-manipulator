"""Run one frozen input panel with one common local-mechanics configuration."""
import argparse,json,subprocess,sys,time,threading
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from scripts.report_grasp_panel import report

p=argparse.ArgumentParser();p.add_argument('--panel',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=3)
p.add_argument('--controlled-transitions',action='store_true');p.add_argument('--interval-control',action='store_true');p.add_argument('--generated-feedback',action='store_true')
p.add_argument('--momentum-feedback',action='store_true');p.add_argument('--finite-slip-preload',action='store_true');p.add_argument('--background-withdrawal',action='store_true')
p.add_argument('--bounded-allocation',action='store_true');p.add_argument('--hold-withdrawal-pose',action='store_true')
p.add_argument('--minimize-input',action='store_true');p.add_argument('--global-structure-search',action='store_true')
p.add_argument('--structure-workers',type=int,default=8)
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
 import numpy as np
 trial_seeds={case['id']:int(seed.generate_state(1)[0]) for case,seed in zip(panel['cases'],np.random.SeedSequence(a.trial_seed).spawn(len(panel['cases'])))}
 (a.output/'trial_seeds.json').write_text(json.dumps(trial_seeds,indent=2))
else:trial_seeds={}
(a.output/'panel.json').write_text(json.dumps(panel,indent=2));(a.output/'runner.py').write_bytes(Path(__file__).read_bytes())
planning_lock=threading.Lock()

def run(case):
 name=case['id'];request=a.output/'requests'/(name+'.json');request.write_text(json.dumps(case,indent=2));out=a.output/name;began=time.perf_counter()
 command=[sys.executable,'-m','build.check_local_task','--request',str(request),'--output',str(out),'--predictive-motion','--distance-closing','--condensed-solve','--steps',str(a.steps)]
 if name in trial_seeds:command+=['--structure-seed',str(trial_seeds[name])]
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
  if a.planned_preload_floor:command+=['--planned-preload-floor']
  if a.finite_acquisition_support:command+=['--finite-acquisition-support']
  if a.process_withdrawal:command+=['--process-withdrawal']
  if a.normal_tracking:command+=['--normal-tracking']
  if a.internal_tracking:command+=['--internal-tracking']
 elif a.interval_control:
  command[2]='build.run_active_task';command+=['--interval-motion','--interval-acquisition','--staged-approach','--hybrid-withdrawal']
 elif a.controlled_transitions:
  command[2]='build.run_active_task';command+=['--constrained-load','--constrained-motion','--staged-approach','--planned-withdrawal']
 with (a.output/(name+'.log')).open('w')as log:
  # This is test-bench queuing before process launch, not an unreported part
  # of online planning. The task runner measures its own complete planning.
  with planning_lock if a.serialized_planning else nullcontext():
   queue_wait=time.perf_counter()-began
   process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
   if a.serialized_planning:
    while process.poll()is None and not(out/'endpoint.npz').exists():time.sleep(.05)
  code=process.wait()
 result_path=out/'result.json'
 if not result_path.exists():
  info=json.loads((out/'experiment.json').read_text())if(out/'experiment.json').exists()else{}
  result=dict(**info,task_success=False,reason='process_error',returncode=code)
  out.mkdir(exist_ok=True);result_path.write_text(json.dumps(result,indent=2));(out/'request.json').write_text(json.dumps(case,indent=2))
 result=json.loads(result_path.read_text())
 row=dict(case=name,returncode=code,benchmark_queue_wait_s=queue_wait,process_wall_s=time.perf_counter()-began-queue_wait,**{key:result.get(key)for key in ['task_success','reason','contacts','modules','motion_methods','initial_planning_s']})
 print(json.dumps(row),flush=True);return row

started=time.perf_counter()
with ThreadPoolExecutor(max_workers=a.workers)as pool:rows=list(pool.map(run,panel['cases']))
(a.output/'timing.json').write_text(json.dumps(dict(wall_s=time.perf_counter()-started,workers=a.workers,serialized_planning=a.serialized_planning,rows=rows),indent=2))
result=report(panel,[a.output],panel['configuration']);(a.output/'report.json').write_text(json.dumps(result,indent=2))
print(json.dumps({key:result[key]for key in ['overall','missing','all_results_same_runtime_sources','all_results_same_execution_settings','reaches_all_rate_thresholds']}),flush=True)
