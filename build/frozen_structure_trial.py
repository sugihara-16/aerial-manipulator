"""One fixed-work, independent count-specific generation/improvement trial.

No incumbent, earlier proposal or task simulation guides this procedure.
Exact deterministic calculations can be memoized without changing the draws.
The task generator constructs port-labelled trees from contact requirements.
Legacy and mixed generators remain explicit development comparisons. Two task
contacts can move on the surface while retaining compressive opposition.
"""
import time
from types import SimpleNamespace
import numpy as np
import mujoco
import build.check_precise_contacts_100
from amsrr_det.objects import ObjectShape
from amsrr_det.structure_groups import skeleton_groups
from amsrr_det.structure_proposal import proposal
from amsrr_det.grasp_task import support_obstacles,refine_material_posture,open_material_posture,finish_plan,material_force_contacts,prepare_static_trims
from amsrr_det.model import build_model
from amsrr_det.equilibrium import contact_normal_obstruction
from amsrr_det.mujoco_model import make_plant,CONTACT_CACHE,contact_dynamics_model
from amsrr_det.structure_bounds import hover_input_burden
from build.mit_drive import dock_mit_model
from build.preload_endpoint import construct
from build.finite_slip_preload import construct_finite_slip
from build.staged_approach import StagedApproach
from amsrr_det.planning_memo import evaluate as memoized


_screen_memo = None


def initialize_screen_memo(directory):
    from amsrr_det import planning_memo
    global _screen_memo
    _screen_memo = planning_memo.active = planning_memo.PlanningMemo(directory)


def prepare(plan,*,task_wrench=False):
    """Common acceptance and cost of one constructed posture, without rollout."""
    started=time.perf_counter()
    links=[f'm{a}/'+('yaw_dock_mech' if b.startswith('yaw') else 'pitch_dock_mech')+b[-1]
           for a,b in plan.surfaces]
    def models():
        full=dock_mit_model(plan.plant,kp=200.)
        return full,contact_dynamics_model(full,links)
    full,plant=memoized(dict(operation='planning_drive_models',xml=plan.plant.xml,
                            timestep=plan.plant.mj.opt.timestep,links=links),models)
    m=plant.mj
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


def generate_candidate(request,modules,rng,*,generator='legacy',descriptor=None,common_initialization=False,retain_task_pose=False):
    """A task-conditioned generator is fixed before any independent trials."""
    payload=request['payload'];shape=ObjectShape(payload['shape'],payload['size'],payload['mass'])
    source=np.array(request['source']);destination=np.array(request['destination'])
    obstacles=support_obstacles(shape,source,destination,request['support_top'])
    if generator not in ('legacy','task','mixed'):
        raise ValueError('unknown structure generator')
    use_task=generator=='task' or (generator=='mixed' and rng.random()<.5)
    # Separate the common fitter's random stream from discrete generation.
    fitting_rng=np.random.default_rng(int(rng.integers(0,2**63))) if common_initialization else None
    if use_task:
        from amsrr_det.task_contacts import task_requirements
        from amsrr_det.task_structure import fit_generated
        from amsrr_det.structure_proposal import fit_assignment
        abstract=descriptor or request
        requirements=task_requirements(abstract)
        points,normals,roles=requirements.choose(modules,rng)
        abstract_obstacles=support_obstacles(requirements.shape,np.array(abstract['source']),
                                            np.array(abstract['destination']),abstract['support_top'])
        discrete_only=(common_initialization and not retain_task_pose) or descriptor is not None
        candidate=fit_generated(modules,requirements.shape,np.array(abstract['source']),abstract_obstacles,points,normals,rng,
                                fit_pose=not discrete_only)
        candidate['generation']['requirements']=roles
        if discrete_only:
            generated=candidate['generation'];generation_s=candidate['wall_s']
            candidate=fit_assignment(modules,candidate['edges'],candidate['surfaces'],shape,source,obstacles,
                fitting_rng if fitting_rng is not None else rng,guided=True,opposed_pair=len(points)==2)
            candidate['generation']=generated
            candidate['wall_s']+=generation_s
        return candidate
    groups=[g for g in skeleton_groups() if g['modules']==modules]
    group=groups[int(rng.integers(len(groups)))];paired=rng.random()<.5
    if common_initialization:
        from amsrr_det.structure_groups import sample_assignment
        from amsrr_det.structure_proposal import fit_assignment
        edges,surfaces=sample_assignment(group,rng,contact_count=2 if paired else None)
        candidate=fit_assignment(modules,edges,surfaces,shape,source,obstacles,fitting_rng,
                                 guided=True,fixed_targets=len(surfaces)==2)
        candidate['group']=group['id']
    else:
        candidate=proposal(group,shape,source,obstacles,rng,
                           contact_count=2 if paired else None,guided=True,fixed_targets=paired)
    candidate['generation']=dict(method='legacy')
    return candidate


def screen_candidate(request,candidate,*,task_wrench=False):
    """Common physical acceptance for independent and auxiliary generation."""
    started=time.perf_counter()
    if _screen_memo is None:
        return _screen_candidate(request,candidate,task_wrench=task_wrench)
    physical={k:request[k] for k in ('payload','source','destination','support_top')}
    inputs=dict(request=physical,task_wrench=task_wrench,
                candidate={k:candidate[k] for k in ('modules','edges','surfaces','q','fit')})
    (row,plan,prepared),hit=_screen_memo.evaluate(inputs,
        lambda:_screen_candidate(request,candidate,task_wrench=task_wrench))
    # Preserve this draw's generation trace and actual elapsed time. Cached
    # stage timings describe the original computation, not work done again.
    row.update(group=candidate['group'],generation=candidate['generation'],
               geometry=candidate['fit'],geometry_s=candidate['wall_s'],
               screen_cache_hit=hit,screen_wall_s=time.perf_counter()-started)
    if hit:
        row['cached_stage_wall_s']=row.pop('stage_wall_s',{})
    return row,plan,prepared


def _screen_candidate(request,candidate,*,task_wrench=False):
    started=time.perf_counter();modules=candidate['modules'];payload=request['payload']
    shape=ObjectShape(payload['shape'],payload['size'],payload['mass'])
    source=np.array(request['source']);destination=np.array(request['destination'])
    obstacles=support_obstacles(shape,source,destination,request['support_top'])
    row=dict(group=candidate['group'],edges=candidate['edges'],surfaces=candidate['surfaces'],
             generation=candidate['generation'],
             geometry=candidate['fit'],geometry_s=candidate['wall_s'],accepted=False)
    fit=candidate['fit']
    if fit['position_error_m']>=.015 or fit['normal_error']>=.15 or fit.get('opposition_error',0.)>1e-4:
        row['reason']='geometric fit unfinished';return row,None,None
    surfaces=list(map(tuple,candidate['surfaces']))
    model,meta=build_model(modules,candidate['edges'],shape.payload())
    meta['material_mesh_cache']=str(CONTACT_CACHE.resolve())
    geometry=SimpleNamespace(pin=model,meta=meta)
    points=np.array(fit['object_points']);normals=np.array(fit['inward_normals'])
    stage='material';stage_started=time.perf_counter();row['stage_wall_s']={}
    try:
        opened,closed,fit=refine_material_posture(geometry,shape,source,np.array(candidate['q']),
                                                  surfaces,points,normals,obstacles=obstacles,open_posture=False)
        if len(surfaces)==2 and 'opposed_friction' in candidate['fit']:
            from amsrr_det.grasp_geometry import opposed_contact_terms
            error=float(opposed_contact_terms(points,normals,4.5).max())
            row['material_opposition_error']=error
            if error>1e-4:raise ValueError('material contact opposition lost')
        contacts=material_force_contacts(geometry,shape,closed,surfaces)
        if contact_normal_obstruction(contacts,shape.mass*model.gravity.linear,.5) is not None:
            raise ValueError('material contact force directions cannot support this load')
        opened=open_material_posture(geometry,shape,source,closed,surfaces,normals,fit,obstacles=obstacles)
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
        return row,None,None
    row['stage_wall_s'][stage]=time.perf_counter()-stage_started
    row.update(accepted=True,burden=prepared['burden'],reserve=prepared['reserve'],
               physical_preparation_s=prepared['wall_s'])
    return row,plan,prepared


def trial(request,modules,trial_index,seed, *, starts=4,task_wrench=False,generator='legacy',descriptor=None,common_initialization=False,retain_task_pose=False):
    started=time.perf_counter()
    rng=np.random.default_rng(np.random.SeedSequence([seed,modules,trial_index]))
    payload=request['payload'];shape=ObjectShape(payload['shape'],payload['size'],payload['mass'])
    source=np.array(request['source']);destination=np.array(request['destination'])
    obstacles=support_obstacles(shape,source,destination,request['support_top'])
    rows=[];best=None;best_prepared=None
    for start in range(starts):
        candidate=generate_candidate(request,modules,rng,generator=generator,descriptor=descriptor,
                                     common_initialization=common_initialization,retain_task_pose=retain_task_pose)
        row,plan,prepared=screen_candidate(request,candidate,task_wrench=task_wrench)
        row['start']=start;rows.append(row)
        if prepared is None:continue
        if best_prepared is None or prepared['burden']<best_prepared['burden']:
            best,best_prepared=plan,prepared
    return dict(modules=modules,trial=trial_index,seed=seed,rows=rows,
                burden=best_prepared['burden'] if best_prepared is not None else float('inf'),
                plan=best,prepared=best_prepared,wall_s=time.perf_counter()-started)
