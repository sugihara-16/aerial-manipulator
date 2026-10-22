"""Current common controller in MuJoCo; saved plans are development only."""
from pathlib import Path
import argparse
import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor


def main():
    parser=argparse.ArgumentParser()
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--plan')
    source.add_argument('--request')
    parser.add_argument('--output',required=True)
    parser.add_argument('--stop-at',type=float)
    parser.add_argument('--timestep',type=float,default=.00025)
    parser.add_argument('--preserve-preload',action='store_true')
    parser.add_argument('--posture-tracking',action='store_true')
    parser.add_argument('--trajectory-feedforward',action='store_true')
    parser.add_argument('--normal-moment-refinement',action='store_true')
    parser.add_argument('--candidate-limit',type=int,default=None)
    parser.add_argument('--search-seconds',type=float,default=30.)
    parser.add_argument('--rest-to-rest-retreat',action='store_true')
    parser.add_argument('--preserve-joint-action',action='store_true')
    parser.add_argument('--coupled-free-flight',action='store_true')
    parser.add_argument('--relative-acquisition',action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--reduced-physics',action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--contact-counts',nargs='+',type=int,choices=[2,3,4],default=[2,3,4])
    parser.add_argument('--preferred-mode',choices=['yaw','pitch'],default='yaw')
    parser.add_argument('--contact-tilts-deg',nargs='+',type=float,default=None)
    parser.add_argument('--screen-prefix',action='store_true')
    parser.add_argument('--physics-workers',type=int,default=0)
    args=parser.parse_args()
    from build.load_saved_plan import load_saved_plan
    from scripts.run_grasp_task import execute
    from amsrr_det.release import warm_planner
    from amsrr_det.evidence import save_sources
    from amsrr_det.structure_search import plan_grasp
    from amsrr_det.objects import ObjectShape
    import numpy as np
    import xml.etree.ElementTree as ET
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    save_sources(out)
    (out/'run_mujoco_task.py').write_bytes(Path(__file__).read_bytes())
    started=time.perf_counter()
    if args.request:
        request=json.loads(Path(args.request).read_text())
        (out/'request.json').write_text(json.dumps(request,indent=2))
        payload=request['payload'];shape=ObjectShape(payload['shape'],payload['size'],payload['mass'])
        try:
            plan=plan_grasp(shape,np.array(request['source']),np.array(request['destination']),request['support_top'],preferred_mode=args.preferred_mode,contact_counts=args.contact_counts,normal_moment_refinement=args.normal_moment_refinement,candidate_limit=args.candidate_limit,search_seconds=args.search_seconds,contact_tilts=None if args.contact_tilts_deg is None else np.deg2rad(args.contact_tilts_deg))
        except ValueError as error:
            result=dict(task_success=False,reason='planning_failed:'+str(error),scope='fresh_task_generation',
                        execution_entrypoint='fresh_task_input',request=request,wall_s=time.perf_counter()-started,
                        evidence=getattr(error,'evidence',None))
            (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True);return
    else:
        plan,request=load_saved_plan(args.plan)
    plan.plant.mj.opt.timestep=args.timestep
    doc=ET.fromstring(plan.plant.xml)
    doc.find('option').set('timestep',str(args.timestep))
    plan.plant.xml=ET.tostring(doc,encoding='unicode')
    (out/'request.json').write_text(json.dumps(request,indent=2))
    with ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context('spawn')) as executor:
        executor.submit(warm_planner).result()
        result=execute(plan,out,request,stop_at=args.stop_at,reduced_physics=args.reduced_physics,screen_prefix=args.screen_prefix,physics_workers=args.physics_workers,task_options=dict(
            guard_servo=True,acquisition_gap_m=.0002,release_method='supported',release_executor=executor,
            preserve_preload=args.preserve_preload,posture_tracking=args.posture_tracking,
            trajectory_feedforward=args.trajectory_feedforward,preserve_joint_action=args.preserve_joint_action,rest_to_rest_retreat=args.rest_to_rest_retreat,coupled_free_flight=args.coupled_free_flight,relative_acquisition=args.relative_acquisition))
    result.update(scope='fresh_generation_spawn_task' if args.request else 'prepared_structure_task_execution_only',
                  execution_entrypoint='fresh_task_input' if args.request else 'saved_development_candidate',
                  assembly_executed=False,engine='MuJoCo',dt=args.timestep,release_method='supported',
                  async_release=True,guard_servo=True,preserve_preload=args.preserve_preload,posture_tracking=args.posture_tracking,
                  trajectory_feedforward=args.trajectory_feedforward,
                  preserve_joint_action=args.preserve_joint_action,normal_moment_refinement=args.normal_moment_refinement,rest_to_rest_retreat=args.rest_to_rest_retreat,
                  coupled_free_flight=args.coupled_free_flight,relative_acquisition=args.relative_acquisition,
                  candidate_limit=args.candidate_limit,
                  search_seconds=args.search_seconds,
                  acquisition_gap_m=.0002,minimum_preload_m=.00015,
                  support_representation='native_unsplit',integrator=int(plan.plant.mj.opt.integrator),
                  solver=int(plan.plant.mj.opt.solver),solver_iterations=int(plan.plant.mj.opt.iterations),
                  solver_tolerance=float(plan.plant.mj.opt.tolerance),total_wall_s=time.perf_counter()-started)
    (out/'result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:result[k] for k in ['task_success','reason','phase','simulation_s','wall_s','control_time_s']}),flush=True)


if __name__=='__main__':
    main()
