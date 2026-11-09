"""Task-directed fixed-count tree, periodic motion and force synthesis.

Only a single periodic path is screened. Physical distance-based execution is
an independent evaluation, never a proposal acceptance test or stored library.
"""
import time,copy
import numpy as np
import pinocchio as pin
import mujoco
from .objects import ObjectShape
from .model import build_model
from .task_structure import grow_to_contacts,part_frames,port_axes,port_origins
from .grasp_geometry import fit_contacts,aim_rotors_up,grasp_frame
from .mujoco_model import make_plant,CONTACT_CACHE
from .walking import foot_link,foot_material,rotation_z,requires_face_alignment
from .walking_control import GroundControl
from .walking_posture import propose_support_pose,refine_support_pose
from .walking_gait import CycleKinematics,validate_cycle
from .walking_cycle import CycleReference,smooth_cycle
from .walking_load import initial_directions,support_switch_times,support_quadrature,motion_load,ground_contacts
from .equilibrium import static_balance,certify_load_changes
from .walking_vectoring import periodic_directions
from .walking_servo import walking_servo_model


from .walking_growth import generate


def plant_for(record,timestep=.0005):
    return walking_servo_model(make_plant(record['modules'],record['edges'],mesh_cache=CONTACT_CACHE,
                                   floor=True,timestep=timestep,compile_threads=False))


from .walking_screen import screen_motion


def prepare_candidate(spec,rng,record=None,*,support_weight=None):
    started=time.perf_counter()
    if record is None:
        record=generate(spec,rng);generation_s=time.perf_counter()-started;previous_s=0.
    else:
        generation_s=record.pop('_generation_s');previous_s=generation_s
    plant=plant_for(record);m=plant.mj;q=plant.configuration(record['q']);links=[foot_link(*s) for s in record['surfaces']]
    duty=record['generation'].get('duty',.62);phases=record['generation'].get('phases')
    d=mujoco.MjData(m);d.qpos[:]=q;mujoco.mj_kinematics(m,d)
    material=foot_material(plant,links)
    gap=min(np.min((vertices@d.xmat[bid].reshape(3,3).T+d.xpos[bid])[:,2]) for bid,vertices in material)
    q[2]-=gap;d.qpos[:]=q;mujoco.mj_kinematics(m,d);mujoco.mj_collision(m,d)
    # Limit the pose-repair effort for deeply self-intersecting proposals.
    # This is a generation heuristic, not proof that the graph cannot walk.
    owners=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,i).split('/')[0] for i in range(m.ngeom)]
    if any(c.dist<-.01 and owners[c.geom1]!='floor' and owners[c.geom2]!='floor'
           and owners[c.geom1]!=owners[c.geom2] for c in d.contact):
        raise ValueError('initial_self_overlap_proposal')
    controller=GroundControl(plant,q,links)
    if support_weight is None:
        motion=spec if any(owner==0 or requires_face_alignment(port) for owner,port in record["surfaces"]) else None
        q,posture,parameters=propose_support_pose(plant,controller,q,iterations=16+8*spec.modules,duty=duty,phases=phases,motion=motion)
    else:
        q,posture=refine_support_pose(plant,controller,q,iterations=16+8*spec.modules,support_weight=support_weight,duty=duty,phases=phases,motion=spec)
        parameters=dict(support_weight=support_weight,capacity_timing=support_weight==20.)
    d.qpos[:]=q;mujoco.mj_kinematics(m,d);mujoco.mj_collision(m,d)
    controller=GroundControl(plant,q,links);kin=CycleKinematics(plant,controller,q)
    if d.ncon and min(d.contact.dist)<-.001:
        floor=m.geom('floor').id;foot_bodies={bid for bid,_ in controller.material}
        penetrating=[c for c in d.contact if c.dist<-.001]
        foot_only=all(floor in c.geom and m.geom_bodyid[c.geom2 if c.geom1==floor else c.geom1] in foot_bodies for c in penetrating)
        if foot_only and min(c.dist for c in penetrating)>-.01:
            # This is a planning seed. A small common height correction keeps
            # a finite-precision sole fit from being rejected before motion
            # synthesis. Full periodic material heights are checked below.
            q[2]-=min(c.dist for c in penetrating)-1e-6
        else:
            feet=kin.points()[0];feet[:,2]=0.
            q,error=kin.fit(q,q[:7],feet,iterations=20)
        d.qpos[:]=q;mujoco.mj_kinematics(m,d);mujoco.mj_collision(m,d)
        if d.ncon and min(d.contact.dist)<-.001:raise ValueError('initial_cad_collision')
        controller=GroundControl(plant,q,links);kin=CycleKinematics(plant,controller,q)
    offsets_list=[phases]
    if len(links)==4 and len({owner for owner,_ in record['surfaces']})==4:
        from .walking_load import support_phases
        pair,_=support_phases(plant,controller,q);offsets_list=[pair,phases]
    proposals=[];reports=[];capacities=[]
    controller.drive.observe(q,np.zeros(plant.mj.nv));qp,_=plant.observation(controller.drive.d)
    weight=plant.meta['mass']*9.80665;cache={}
    for offsets in offsets_list:
        offsets=np.asarray(offsets);cuts=np.unique(np.r_[0.,1.,(-offsets)%1.,(duty-offsets)%1.]);widths=np.diff(cuts);values=[]
        for t in (cuts[:-1]+cuts[1:])/2:
            active=tuple((t+offsets)%1<duty)
            if active not in cache:
                force=static_balance(plant.pin,plant.meta,qp,ground_contacts(controller,active),fast_lp=True,maximize_contact_load=True,peak_effort_bound=.9)
                cache[active]=sum(c['normal_force']for c in force.get('contacts',[]))/weight
            values.append(cache[active])
        mean=float(widths@values);fraction=float(widths@(np.asarray(values)>=.55))
        capacities.append(dict(mean=mean,fraction=fraction,deficit=max(0.,.55-mean)+max(0.,.6-fraction)))
    # This frozen-pose screen limits effort spent on weak proposals. It is
    # not an impossibility bound on other postures of the same graph.
    if not any(c['mean']>=.5 and c['fraction']>=.4 for c in capacities):
        raise ValueError('nominal_support_proposal')
    for offsets in offsets_list:
        preview=kin.generate(spec,period=record['period'],samples=17,iterations=6,
                             fixed_duty=duty,fixed_phases=offsets)
        preview,sm=smooth_cycle(kin,spec,preview)
        report=validate_cycle(plant,preview,spec=spec)
        proposals.append(preview);reports.append(report)
    def violation(i):
        r=reports[i];return max(1.,r['position_error']/.005,-r['penetration']/.001,
            r['maximum_face_angle_rad']/np.deg2rad(1.),r['velocity_ratio'],
            1+r['angle_excess_rad']/.01)
    selected=min(range(len(proposals)),key=lambda i:(violation(i),capacities[i]['deficit'],i))
    gait=proposals[selected];record['generation']['phases']=gait['phases'].tolist()
    record['generation']['finite_order_proposal']=dict(reports=reports,capacities=capacities,selected=selected)
    ref=CycleReference(plant,spec,gait)
    gait['configurations']=np.array([ref.pose(t) for t in np.linspace(0,ref.period,49)])
    gait,smoothing=smooth_cycle(kin,spec,gait)
    if gait['closure_error']>=.002 or max(gait['errors'])>=.005:raise ValueError('periodic_geometry')
    geometry=validate_cycle(plant,gait,spec=spec)
    if not geometry['passed']:raise ValueError('periodic_geometry')
    record.update(_prepared_plant=plant,proposal_parameters=parameters,gait=gait,geometry=geometry,
                  smoothing=smoothing,posture=posture,generation_s=generation_s,controller_reference=q,support_pose=q.copy(),
                  preparation_s=previous_s+time.perf_counter()-started)
    return record

def finish_candidate(spec,record,*,phase_cpus=None):
    started=time.perf_counter();plant=record.pop('_prepared_plant',None)
    if plant is None:plant=plant_for(record)
    links=[foot_link(*s) for s in record['surfaces']]
    controller=GroundControl(plant,np.array(record.pop('controller_reference')),links,foot_points=record['gait']['foot_points'])
    def screen(*args):
        return screen_motion(*args,cpus=None if phase_cpus is None else phase_cpus())
    if phase_cpus is not None:screen.phase_cpus=phase_cpus
    gait=record['gait'];smoothing=record['smoothing'];posture=record['posture'];generation_s=record['generation_s']
    from .walking_vectoring import periodic_directions
    # One periodic direction synthesis serves feasibility and consumption.
    # Every independent trial uses the same bounded number of updates.
    gait,direction=periodic_directions(plant,controller,spec,gait,screen,iterations=6)
    geometry=direction.get('geometry') or validate_cycle(plant,gait,spec=spec)
    if not geometry['passed']:raise ValueError('directed_periodic_geometry')
    report=direction.get('accepted_report') or screen(plant,controller,spec,gait)
    record.update(gait=gait,report=report,geometry=geometry,smoothing=smoothing,cost=report['cost'],
                  posture=posture,direction=direction,_execution_plant=plant,generation_s=generation_s,wall_s=record['preparation_s']+time.perf_counter()-started)
    from .walking_landing import screen_landing
    record['landing']=screen_landing(plant,record,spec)
    record['wall_s']=record['preparation_s']+time.perf_counter()-started
    return record


def candidate(spec,rng,record=None):
    """One fixed trial, including any posture repair; never two samples.

    Small models can afford a second full posture attempt within the planning
    budget. This computational boundary is not a physical impossibility rule.
    Larger models retain the static-capacity proposal. Every returned plan
    passes the same complete motion screen, without an execution rollout.
    """
    started=time.perf_counter()
    if record is None:
        record=generate(spec,rng);record['_generation_s']=time.perf_counter()-started
    previous_s=record['_generation_s'];started=time.perf_counter();attempts=[]
    choices=(5.,20.) if spec.modules<=4 else (None,)
    for weight in choices:
        try:
            prepared=prepare_candidate(spec,None,copy.deepcopy(record),support_weight=weight)
            result=finish_candidate(spec,prepared)
            result.update(proposal_attempts=attempts,wall_s=previous_s+time.perf_counter()-started)
            return result
        except ValueError as error:
            attempts.append(dict(support_weight=weight,reason=str(error)))
    raise ValueError(attempts[-1]['reason'])
