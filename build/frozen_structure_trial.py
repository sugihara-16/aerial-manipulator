"""One fixed-work, independent count-specific generation/improvement trial.

No incumbent, previous trial output or task simulation enters this procedure.
The generator includes all legal trees, ports and nonempty free-pad subsets.
Half the starts emphasize two opposed contact targets; the remaining half
retain the explicit all-count distribution. Those weights are algorithmic
initialization choices, not probabilities that structures are good.
"""
import time
from types import SimpleNamespace
import numpy as np
import mujoco
import build.check_precise_contacts_100
from amsrr_det.objects import ObjectShape
from amsrr_det.structure_groups import skeleton_groups
from amsrr_det.structure_proposal import proposal
from amsrr_det.grasp_task import support_obstacles,refine_material_posture,finish_plan,material_force_contacts,prepare_static_trims
from amsrr_det.model import build_model
from amsrr_det.equilibrium import contact_normal_obstruction
from amsrr_det.mujoco_model import make_plant,CONTACT_CACHE,contact_dynamics_model
from amsrr_det.structure_bounds import hover_input_burden
from build.mit_drive import dock_mit_model
from build.preload_endpoint import construct
from build.finite_slip_preload import construct_finite_slip
from build.staged_approach import StagedApproach


def prepare(plan,*,task_wrench=False):
    """Common acceptance and cost of one constructed posture, without rollout."""
    started=time.perf_counter()
    full=dock_mit_model(plan.plant,kp=200.)
    links=[f'm{a}/'+('yaw_dock_mech' if b.startswith('yaw') else 'pitch_dock_mech')+b[-1]
           for a,b in plan.surfaces]
    plant=contact_dynamics_model(full,links);m=plant.mj
    m.opt.impratio=10000.;m.opt.timestep=.00025
    m.opt.ccd_tolerance=1e-12;m.opt.ccd_iterations=1000
    q=plant.configuration(plan.closed_q)
    heights=m.jnt_qposadr[np.flatnonzero(m.jnt_type==mujoco.mjtJoint.mjJNT_FREE)]+2
    q[heights]+=.15
    endpoint,report=construct(plant,full.mj,q,links)
    if endpoint is None:raise ValueError('finite reference: '+report['reason'])
    (qair,u,velocity),finite=construct_finite_slip(plant,full.mj,endpoint[0],links,23.,minimize_input=True)
    wrench_info=None
    if task_wrench:
        from build.task_wrench_margin import refine_task_reference
        (qair,u,velocity),finite,wrench_info=refine_task_reference(plant,full.mj,qair,u,velocity,links,
            plan.destination[:3]-plan.source[:3],finite)
    goal=qair.copy();goal[heights]-=.15
    approach=StagedApproach(plan,plant,goal,links)
    final=finite['iterations'][-1]['recheck'] if finite['iterations'] else finite['initial_fixed']
    return dict(burden=hover_input_burden(u[-len(plant.meta['rotors']):]),reserve=final['reserve'],
                qair=qair,goal=goal,input=u,velocity=velocity,finite=finite,task_wrench=wrench_info,
                approach=approach.info,approach_path=approach.prepared(),wall_s=time.perf_counter()-started)


def trial(request,modules,trial_index,seed, *, starts=4,task_wrench=False):
    started=time.perf_counter()
    rng=np.random.default_rng(np.random.SeedSequence([seed,modules,trial_index]))
    payload=request['payload'];shape=ObjectShape(payload['shape'],payload['size'],payload['mass'])
    source=np.array(request['source']);destination=np.array(request['destination'])
    obstacles=support_obstacles(shape,source,destination,request['support_top'])
    groups=[g for g in skeleton_groups() if g['modules']==modules]
    rows=[];best=None;best_prepared=None
    for start in range(starts):
        group=groups[int(rng.integers(len(groups)))];paired=rng.random()<.5
        candidate=proposal(group,shape,source,obstacles,rng,
                           contact_count=2 if paired else None,guided=True,fixed_targets=paired)
        row=dict(start=start,group=group['id'],edges=candidate['edges'],surfaces=candidate['surfaces'],
                 geometry=candidate['fit'],geometry_s=candidate['wall_s'],accepted=False)
        rows.append(row);fit=candidate['fit']
        if fit['position_error_m']>=.015 or fit['normal_error']>=.15:
            row['reason']='geometric fit unfinished';continue
        surfaces=list(map(tuple,candidate['surfaces']))
        model,meta=build_model(modules,candidate['edges'],shape.payload())
        meta['material_mesh_cache']=str(CONTACT_CACHE.resolve())
        geometry=SimpleNamespace(pin=model,meta=meta)
        points=np.array(fit['object_points']);normals=np.array(fit['inward_normals'])
        stage='material';stage_started=time.perf_counter();row['stage_wall_s']={}
        try:
            opened,closed,fit=refine_material_posture(geometry,shape,source,np.array(candidate['q']),
                                                      surfaces,points,normals,obstacles=obstacles)
            contacts=material_force_contacts(geometry,shape,closed,surfaces)
            if contact_normal_obstruction(contacts,shape.mass*model.gravity.linear,.5) is not None:
                raise ValueError('material contact force directions cannot support this load')
            row['stage_wall_s'][stage]=time.perf_counter()-stage_started
            stage='static';stage_started=time.perf_counter()
            statics=prepare_static_trims(geometry,shape,opened,closed,surfaces,points,normals,contacts=contacts)
            row['stage_wall_s'][stage]=time.perf_counter()-stage_started
            stage='model_and_clearance';stage_started=time.perf_counter()
            p=make_plant(modules,candidate['edges'],shape.payload(),mesh_cache=CONTACT_CACHE,floor=True)
            plan=finish_plan(p,shape,source,destination,request['support_top'],
                             opened,closed,surfaces,points,normals,fit,started,static_preparation=statics)
            row['stage_wall_s'][stage]=time.perf_counter()-stage_started
            stage='finite_and_approach';stage_started=time.perf_counter()
            prepared=prepare(plan,task_wrench=task_wrench)
        except (ValueError,mujoco.FatalError) as error:
            row['stage_wall_s'][stage]=time.perf_counter()-stage_started
            row['reason']=str(error)
            if isinstance(error,mujoco.FatalError):row['numerical_model_unresolved']=True
            continue
        row['stage_wall_s'][stage]=time.perf_counter()-stage_started
        row.update(accepted=True,burden=prepared['burden'],reserve=prepared['reserve'],
                   physical_preparation_s=prepared['wall_s'])
        if best_prepared is None or prepared['burden']<best_prepared['burden']:
            best,best_prepared=plan,prepared
    return dict(modules=modules,trial=trial_index,seed=seed,rows=rows,
                burden=best_prepared['burden'] if best_prepared is not None else float('inf'),
                plan=best,prepared=best_prepared,wall_s=time.perf_counter()-started)
