"""Torque-directed contact roles and the shared connection/pose generator."""
import time
from types import SimpleNamespace
import numpy as np
import pinocchio as pin
from .valve import ValveSpec,ValvePlant
from .task_structure import fit_generated
from .grasp_task import refine_material_posture,material_force_contacts,open_material_posture
from .grasp_geometry import aim_rotors_up,grasp_frame,fit_contacts
from .valve_trim import static_balance
from .structure_bounds import hover_input_burden


def contact_roles(spec,count,rng):
    """Compressive bar-side force has signed leverage |x| about the axis.

    The torque demand sets force per role; leverage and spatial separation
    direct site selection before any assembled robot exists.
    """
    if not 1<=count<=4:raise ValueError('valve uses one to four contact pads')
    radii=spec.length*np.array([.24,.34,.44])
    sites=np.r_[-radii,radii]
    selected=[];trace=[]
    for _ in range(count):
        remaining=[i for i in range(len(sites)) if i not in selected]
        # Greater leverage reduces required force. Separation avoids asking
        # different modules to occupy the same patch; neither proves fit.
        costs=[]
        for i in remaining:
            force=spec.torque/(sum(abs(sites[j]) for j in selected)+abs(sites[i]))
            crowd=sum(np.exp(-((sites[i]-sites[j])/.2)**2) for j in selected)
            costs.append(force+5*crowd)
        w=np.exp(-(np.array(costs)-min(costs))/.25)
        chosen=remaining[int(rng.choice(len(remaining),p=w/w.sum()))]
        selected.append(chosen);trace.append(dict(x=float(sites[chosen]),cost=float(costs[remaining.index(chosen)])))
    x=sites[selected];normals=np.c_[np.zeros(count),np.sign(spec.target_angle)*np.sign(x),np.zeros(count)]
    points=np.c_[x,-normals[:,1]*spec.width/2,np.zeros(count)]
    return points,normals,dict(method='signed_torque_leverage',torque_Nm=spec.torque,
        nominal_normal_force_N=float(spec.torque/sum(abs(x))),trace=trace,
        scope='proposal; robot load and full rotation path require separate checks')


def bearing_support(spec):
    basis=np.zeros((6,5));basis[:3,:3]=np.eye(3);basis[3:,3:]=spec.rotation[:,:2]
    return basis


def gravity_axis_symmetry(p,spec):
    return np.linalg.norm(np.cross(p.pin.gravity.linear,np.asarray(spec.axis)))<1e-12


def rotate_assembly(p,spec,configuration,angle,origin):
    q=np.asarray(configuration).copy();turn=pin.exp3(np.asarray(spec.axis)*(angle-origin))
    center=np.asarray(spec.center)
    q[:3]=center+turn@(q[:3]-center);q[3:7]=pin.Quaternion(turn@pin.Quaternion(q[3:7]).matrix()).coeffs()
    pose=spec.pose(angle);oq=p.meta['object']['q'];q[oq:oq+7]=np.r_[pose[:3],pose[4:],pose[3]]
    return q


def axis_symmetric_environment(model,spec,configuration):
    """Inspect actual fixed collision geometry, not a task-name assumption.

    A plane normal to the axis and a coaxial cylinder are invariant under
    rotation about it. Every moving root is rotated by AxisRotationPath.
    """
    import mujoco
    axis=np.asarray(spec.axis);center=np.asarray(spec.center)
    data=mujoco.MjData(model);data.qpos[:]=configuration;mujoco.mj_kinematics(model,data)
    moving=np.zeros(model.nbody,bool)
    for body in range(1,model.nbody):
        moving[body]=moving[model.body_parentid[body]] or model.body_jntnum[body]>0
    for geom in range(model.ngeom):
        if moving[model.geom_bodyid[geom]]:continue
        kind=model.geom_type[geom];normal=data.geom_xmat[geom].reshape(3,3)[:,2]
        if np.linalg.norm(np.cross(normal,axis))>1e-12:return False
        if kind==mujoco.mjtGeom.mjGEOM_PLANE:continue
        if kind!=mujoco.mjtGeom.mjGEOM_CYLINDER:return False
        if np.linalg.norm(np.cross(data.geom_xpos[geom]-center,axis))>1e-12:return False
    return True


def mounting_obstacles(spec):
    """Conservative world boxes for the wheel and stationary mounting."""
    R=spec.rotation;center=np.asarray(spec.center)
    return [(center+R@np.array([0.,0.,z]),abs(R)@np.asarray(half))
            for z,half in [(-.055,(.16,.16,.009)),(-.24,(.065,.065,.17))]]


def compress_reference(plant,q,surfaces,shape,depth=.0002):
    """Small geometric pressure seed; it is not an executed grasp state."""
    surfaces=list(map(tuple,surfaces))
    d=plant.pin.createData();pin.framesForwardKinematics(plant.pin,d,q)
    frames=[d.oMf[plant.meta['frames'][grasp_frame(*s)]] for s in surfaces]
    points=np.array([f.translation for f in frames]);normals=np.array([f.rotation[:,0] for f in frames])
    result,fit=fit_contacts(plant.pin,plant.meta,q,surfaces,points+depth*normals,normals,
                            base_rotation=True,max_evaluations=30)
    if fit['position_error_m']>depth/3:raise ValueError('cannot form valve pressure reference')
    return result


def loaded_trim(p,spec,q,surfaces,previous=None):
    """Continue directions that support the task's whole torque interval."""
    from .planning_memo import check_cancelled
    check_cancelled()
    from .valve_trim import load_span_trim
    from .steerable_trim import steerable_trim
    actuators=[a for a in p.meta['actuators'] if a['role']=='vectoring']
    indices=np.array([a['q'] for a in actuators]);seed=q.copy()
    if previous is not None:
        seed[indices]=aim_rotors_up(p.pin,p.meta,q)[indices]+previous[indices]-aim_rotors_up(p.pin,p.meta,previous)[indices]
    contacts=material_force_contacts(p,spec.shape,q,surfaces,force_limit=None,material_tangents=True)
    settings=dict(payload_fraction=0.,object_support=bearing_support(spec),
        minimum_pad_normal_force_n=.1,minimize_burden=True,fast_lp=True,direct_burden=True,burden_power=2.)
    def checks(pose):
        trims=[];workspace={}
        for amount in [.9,1.1]:
            trim=static_balance(p.pin,p.meta,pose,contacts,
                object_wrench=np.r_[np.zeros(3),np.asarray(spec.axis)*np.sign(spec.target_angle)*spec.torque*amount],workspace=workspace,**settings)
            if trim['status']!='static_screen_pass':return None
            trims.append(trim)
        result=max(trims,key=lambda t:hover_input_burden(t['thrust']))
        result['torque_interval_factors']=[.9,1.1]
        return result
    trim=checks(seed)
    if trim is not None:return seed,trim
    proposed,upper=steerable_trim(p.pin,p.meta,seed,contacts,
        object_wrench=np.r_[np.zeros(3),np.asarray(spec.axis)*np.sign(spec.target_angle)*spec.torque*1.1],**settings)
    if upper['status']=='static_screen_pass':
        trim=checks(proposed)
        if trim is not None:return proposed,trim
    fixed,span=load_span_trim(p,spec,proposed if upper['status']=='static_screen_pass' else seed,surfaces)
    if span['free']==span['load']=='static_screen_pass':
        trim=checks(fixed)
        if trim is not None:return fixed,trim
    rng=np.random.default_rng(1049)
    for k in range(24):
        trial=seed.copy();trial[indices]+=rng.normal(0.,.06 if k<8 else .2 if k<16 else .5,len(indices))
        trial[indices]=np.clip(trial[indices],[a['lower']+.02 for a in actuators],[a['upper']-.02 for a in actuators])
        trim=checks(trial)
        if trim is not None:return trial,trim
    raise ValueError('rotation torque interval trim unavailable')


def candidate(spec,modules,contacts,rng,*,compile_model=True):
    start=time.perf_counter();points,normals,roles=contact_roles(spec,contacts,rng)
    pose=spec.pose(spec.target_angle/2);source=np.r_[pose[:3],pose[4:],pose[3]]
    obstacles=mounting_obstacles(spec)
    from .valve_root import task_root_score
    from .valve_growth import support_score
    generated=fit_generated(modules,spec.shape,source,obstacles,points,normals,rng,opposed=False,motion_axis=spec.axis,
        root_score=(lambda port,k,sign,R:task_root_score(spec,points[0],normals[0],port,k,sign,R)) if modules==1 else None,
        support_score=lambda edges,surfaces,root,values:support_score(spec,edges,surfaces,root,values))
    # Geometry fitting and the force screen only use rigid-body kinematics.
    # Compile the dynamic/contact model after these inexpensive filters pass.
    from .model import build_model
    from .mujoco_model import CONTACT_CACHE
    model,meta=build_model(modules,generated['edges'],spec.payload)
    meta['material_mesh_cache']=str(CONTACT_CACHE.resolve())
    plant=SimpleNamespace(pin=model,meta=meta)
    points=np.asarray(generated['generation']['target_points']);normals=np.asarray(generated['generation']['target_normals'])
    opened,closed,fit=refine_material_posture(plant,spec.shape,source,np.asarray(generated['q']),
        generated['surfaces'],points,normals,open_posture=False,obstacles=obstacles)
    closed=aim_rotors_up(plant.pin,plant.meta,closed)
    force_contacts=material_force_contacts(plant,spec.shape,closed,generated['surfaces'],force_limit=None,material_tangents=True)
    wanted=np.r_[np.zeros(3),np.asarray(spec.axis)*np.sign(spec.target_angle)*spec.torque*1.1]
    support=bearing_support(spec);gimbals=[a for a in plant.meta['actuators'] if a['role']=='vectoring']
    from .steerable_trim import steerable_trim
    proposed,trim=steerable_trim(plant.pin,plant.meta,closed,force_contacts,payload_fraction=0.,
        object_wrench=wanted,object_support=support,minimum_pad_normal_force_n=.1,minimize_burden=True,fast_lp=True,direct_burden=True,burden_power=2.)
    records=[trim];best=(proposed,trim) if trim['status']=='static_screen_pass' else None
    for k in range(12 if best is None else 0):
        q=closed.copy()
        if k:
            for a in gimbals:q[a['q']]=np.clip(q[a['q']]+rng.normal(0,.65),a['lower']+.02,a['upper']-.02)
        trim=static_balance(plant.pin,plant.meta,q,force_contacts,payload_fraction=0.,
            object_wrench=wanted,object_support=support,minimum_pad_normal_force_n=.1,minimize_burden=True,fast_lp=True,direct_burden=True,burden_power=2.)
        records.append(trim)
        if trim['status']=='static_screen_pass':best=(q,trim);break
    if best is None:
        e=ValueError('valve load trim not found');e.evidence=records;raise e
    closed,trim=best
    pressure_reference=False
    if gravity_axis_symmetry(plant,spec):
        # Test load transfer on the exact compressed reference before mesh
        # compilation. Symmetry carries this pose to both rotation ends.
        from .valve_trim import endpoint_load_envelope
        closed=compress_reference(plant,closed,generated['surfaces'],spec.shape)
        closed,envelope=endpoint_load_envelope(plant,spec,closed,generated['surfaces'])
        if envelope['free']!='static_screen_pass' or envelope['load']!='static_screen_pass':
            raise ValueError('release load envelope unresolved')
        pressure_reference=True
    opened=closed.copy();empty=None
    system=ValvePlant(modules,generated['edges'],spec) if compile_model else SimpleNamespace(control=plant,spec=spec)
    return SimpleNamespace(system=system,closed=closed,opened=opened,surfaces=generated['surfaces'],
        points=points,normals=normals,fit=fit,trim=trim,empty_trim=empty,
        generation=dict(**generated['generation'],requirements=roles),
        pressure_reference=pressure_reference,burden=hover_input_burden(trim['thrust']),planning_s=time.perf_counter()-start)


def posture_path(plan, *, steps=13,force_screen=True):
    """Continue contact kinematics along the requested angle, favouring up.

    Every point is generated from bar motion, not a saved robot catalogue.
    This is a sampled geometric path, not a dynamic execution certificate.
    """
    p=plan.system.control;spec=plan.system.spec;model=p.pin;meta=p.meta
    surfaces=list(map(tuple,plan.surfaces));q=np.asarray(plan.closed).copy()
    if gravity_axis_symmetry(p,spec):
        angles=np.linspace(0,spec.target_angle,steps);path=[];reports=[];previous=None
        for angle in angles:
            pose=rotate_assembly(p,spec,q,angle,spec.target_angle/2)
            info=dict(position_error_m=0.,normal_error=0.,tangent_error=0.,gravity_axis_symmetry=True)
            if force_screen:
                pose,info['trim']=loaded_trim(p,spec,pose,surfaces,previous);previous=pose
            path.append(pose);reports.append(info)
        return angles,np.asarray(path),reports
    data=model.createData();pin.framesForwardKinematics(model,data,q)
    frame=[data.oMf[meta['frames'][grasp_frame(*s)]].copy() for s in surfaces]
    points=np.array([f.translation for f in frame]);normals=np.array([f.rotation[:,0] for f in frame]);tangents=np.array([f.rotation[:,1] for f in frame])
    oq=meta['object']['q'];source=pin.Quaternion(q[oq+3:oq+7]).matrix();center=np.asarray(spec.center)
    angles=np.linspace(0,spec.target_angle,steps);path=[None]*steps;reports=[None]*steps
    middle=steps//2;origin=q.copy();previous=None
    order=list(range(middle,-1,-1))+list(range(middle+1,steps))
    for index in order:
        if index==middle+1:q=origin.copy();previous=origin.copy()
        angle=angles[index]
        pose=spec.pose(angle);R=pin.Quaternion(np.r_[pose[4:],pose[3]]).matrix();turn=R@source.T
        q[oq:oq+7]=np.r_[pose[:3],pose[4:],pose[3]]
        targets=(points-center)@turn.T+center;directions=normals@turn.T
        q,info=fit_contacts(model,meta,q,surfaces,targets,directions,base_rotation=True,
                            base_upright_weight=.005,max_evaluations=60,obstacles=mounting_obstacles(spec),frame_tangents=tangents@turn.T)
        if len(surfaces)==1:
            pin.framesForwardKinematics(model,data,q)
            desired=pin.SE3(turn@frame[0].rotation,targets[0])
            base=pin.SE3(pin.Quaternion(q[3:7]).matrix(),q[:3])
            base=desired*data.oMf[meta['frames'][grasp_frame(*surfaces[0])]].inverse()*base
            q[:3]=base.translation;q[3:7]=pin.Quaternion(base.rotation).coeffs()
            info.update(position_error_m=0.,normal_error=0.,tangent_error=0.,exact_single_frame=True)
        reports[index]=info
        if info['position_error_m']>.002 or info['normal_error']>.02 or info['tangent_error']>.02:
            error=ValueError('valve material-frame path unresolved');error.evidence=dict(angle=float(angle),fit=info);raise error
        if force_screen:
            q,trim=loaded_trim(p,spec,q,surfaces,previous)
            reports[index]['trim']=trim
        path[index]=q.copy()
        previous=q.copy()
        if index==middle:origin=q.copy()
    return angles,np.asarray(path),reports


def complete_path(plan):
    from .planning_memo import evaluate
    def calculate():
        try:_complete_path(plan)
        except ValueError as error:
            return dict(failure=str(error),evidence=getattr(error,'evidence',None))
        names=['closed','opened','empty_trim','final_open','final_trim','angles','pin_path',
               'path_report','burden','reference_path','geometry_angles','geometry_path','curve_record','direction_repairs','intermediate_loads','geometry_samples']
        return {name:getattr(plan,name) for name in names}
    result=evaluate(dict(operation='valve_contact_path',specification=plan.system.spec.__dict__,
        modules=plan.system.control.meta['modules'],edges=plan.system.control.meta['edges'],
        surfaces=plan.surfaces,closed=np.asarray(plan.closed).tolist(),
        points=np.asarray(plan.points).tolist(),normals=np.asarray(plan.normals).tolist()),calculate)
    if 'failure' in result:
        error=ValueError(result['failure']);error.evidence=result['evidence'];raise error
    plan.__dict__.update(result)
    return plan


def _complete_path(plan):
    """Add actual-angle load screens, free-flight ends and geometric checks."""
    from .free_flight_trim import free_flight_trim
    from .grasp_geometry import separating_translation
    import mujoco
    p=plan.system.control;spec=plan.system.spec
    steps=max(3,2*int(np.ceil(abs(spec.target_angle)/np.deg2rad(30)))+1)
    angles,path,reports=posture_path(plan,steps=steps,force_screen=False);ends=[]
    plan.angles=angles;plan.pin_path=path;plan.path_report=reports
    plan.burden=0.
    links=[f'm{n}/'+('yaw_dock_mech' if port.startswith('yaw') else 'pitch_dock_mech')+port[-1] for n,port in plan.surfaces]
    from .configuration_path import TabulatedContactPath,AxisRotationPath
    symmetry=gravity_axis_symmetry(p,spec)
    if symmetry:
        mid=len(angles)//2
        compressed=path[mid] if getattr(plan,'pressure_reference',False) else compress_reference(p,path[mid],plan.surfaces,spec.shape)
        reference=np.asarray([p.configuration(rotate_assembly(p,spec,compressed,a,angles[mid])) for a in angles])
    else:reference=np.asarray([p.configuration(compress_reference(p,q,plan.surfaces,spec.shape)) for q in path])
    plan.geometry_angles=angles.copy();plan.geometry_path=reference.copy()
    curve=(AxisRotationPath(p,angles,reference[len(angles)//2],spec.axis,spec.center,angles[len(angles)//2]) if symmetry else
        TabulatedContactPath(p,angles,reference,plan.surfaces,plan.normals,
        geometry_parameters=plan.geometry_angles,geometry_configurations=plan.geometry_path,object_shape=spec.shape))
    # Trace one supported branch backwards from an unloadable end. Each
    # sampled pose uses exactly the fixed geometric curve used at execution.
    from .valve_trim import endpoint_load_envelope
    pq,mq,*_=p.coordinate_maps
    d=mujoco.MjData(p.mj);checked=0;intermediate=[]
    pq,mq,*_=p.coordinate_maps
    moving_bodies={int(p.mj.jnt_bodyid[p.mj.joint(a['name']).id]) for a in p.meta['actuators'] if a['role']=='vectoring'}
    for body in range(1,p.mj.nbody):
        if int(p.mj.body_parentid[body]) in moving_bodies:moving_bodies.add(body)
    rotor_geoms={g for g in range(p.mj.ngeom) if int(p.mj.geom_bodyid[g]) in moving_bodies}
    def geometry(q,*,before_directions=False):
        nonlocal checked
        d.qpos[:]=q;mujoco.mj_fwdPosition(p.mj,d);checked+=1
        for contact in d.contact:
            if contact.dist>=-.001:continue
            if before_directions and any(int(g) in rotor_geoms for g in contact.geom):continue
            names=[p.mj.geom(int(g)).name for g in contact.geom]
            allowed='object' in names and any(any(name.startswith(link+'/') for link in links) for name in names)
            if not allowed:
                error=ValueError('rotation path intersects geometry')
                error.evidence=dict(pair=names,depth_m=float(-contact.dist));raise error
    dense=np.linspace(angles[0],angles[-1],max(3,int(np.ceil(abs(spec.target_angle)/np.deg2rad(1.)))+1))
    geometry_symmetry=symmetry and axis_symmetric_environment(p.mj,spec,curve.pose(dense[0]))
    # Body/Dock intersections cannot be repaired by changing rotor angles.
    # Screen the entire fixed geometry before expensive force continuation.
    frame_ids=[p.meta['frames'][grasp_frame(*s)] for s in plan.surfaces]
    pdata=p.pin.createData();material_reference=None;material_position_error=material_angle_error=0.
    for angle in ([dense[0]] if geometry_symmetry else dense):
        pose=curve.pose(angle);geometry(pose,before_directions=True)
        configuration=pin.neutral(p.pin);configuration[pq]=pose[mq]
        pin.framesForwardKinematics(p.pin,pdata,configuration)
        obj=pdata.oMf[p.meta['object']['frame']]
        material=[obj.inverse()*pdata.oMf[f] for f in frame_ids]
        if material_reference is None:material_reference=material
        for expected,actual in zip(material_reference,material):
            material_position_error=max(material_position_error,float(np.linalg.norm(expected.translation-actual.translation)))
            material_angle_error=max(material_angle_error,float(np.linalg.norm(pin.log3(expected.rotation.T@actual.rotation))))
    if material_position_error>.003 or material_angle_error>.03:
        raise ValueError('interpolated contact material frame is not preserved')
    reports.append(dict(scope='dense_material_frame_check',position_error_m=material_position_error,
                        rotation_error_rad=material_angle_error,samples=1 if geometry_symmetry else len(dense),
                        rigid_rotation_environment_symmetry=geometry_symmetry))
    final=pin.neutral(p.pin);final[pq]=curve.pose(angles[-1])[mq]
    # The inner friction diamond rotates with the material frame. Under
    # rigid gravity-axis rotation its force/joint problem is identical.
    final_contacts=material_force_contacts(p,spec.shape,final,plan.surfaces,force_limit=None,material_tangents=True)
    force_symmetry=symmetry and all('tangent' in c for c in final_contacts)
    final,envelope=endpoint_load_envelope(p,spec,final,plan.surfaces)
    if envelope['free']!='static_screen_pass' or envelope['load']!='static_screen_pass':
        raise ValueError('release load envelope unresolved')
    repairs=[];intermediate=[];nodes={};previous=None;raw=[];boundary_costs=[]
    gi=np.array([a['q'] for a in p.meta['actuators'] if a['role']=='vectoring'])
    from .planning_memo import check_output_bound
    from .valve_direction_refinement import refine
    for angle in reversed(dense):
        q=curve.pose(angle);qp=pin.neutral(p.pin);qp[pq]=q[mq]
        qp[gi]=final[gi] if previous is None else previous[gi]
        if angle==dense[0] and not force_symmetry:
            qp,envelope=endpoint_load_envelope(p,spec,qp,plan.surfaces)
            if envelope['free']!='static_screen_pass' or envelope['load']!='static_screen_pass':
                raise ValueError('acquisition load envelope unresolved')
        if force_symmetry and previous is not None:
            # Every joint angle, body-relative force map, gravity load and
            # hinge torque is unchanged by this exact rigid yaw. Reuse the
            # accepted common direction and force solution, not a rollout.
            fixed=qp;fixed[gi]=previous[gi];trim=raw[-1][2]
        else:fixed,trim=loaded_trim(p,spec,qp,plan.surfaces,None if angle==dense[0] else previous)
        if previous is None or angle==dense[0]:
            free=static_balance(p.pin,p.meta,fixed,payload_fraction=0.,direct_burden=True)
            if free['status']!='static_screen_pass':raise ValueError('release load envelope lost')
            boundary_costs.extend([hover_input_burden(free['thrust']),hover_input_burden(trim['thrust'])])
            # These endpoint costs are immutable parts of the final MAX.
            # Reject a dominated fixed-procedure output before continuing
            # the other 180 poses; this is not a partial-solution bound.
            check_output_bound(max(boundary_costs),dict(kind='finalized_load_transfer_endpoint',angle=float(angle)))
        # Continue the feasible branch using the original direction proposal.
        # Improvement never changes the proposal at another path sample.
        previous=fixed.copy()
        raw.append((float(angle),fixed,trim))
    # The MAX score already includes both free-flight endpoints. Improving
    # work poses below that immutable value cannot improve the final score.
    target_cost=max(boundary_costs);previous_output=None
    check_output_bound(target_cost,dict(kind='finalized_load_transfer_endpoints'))
    geometry_directions=set()
    for angle,fixed,trim in raw:
        burden=hover_input_burden(trim['thrust'])
        if angle not in [dense[0],dense[-1]] and burden>target_cost:
            proposed,improved=refine(p,spec,fixed,plan.surfaces,steps=3,target=target_cost,initial=previous_output)
            improved_cost=hover_input_burden(improved['thrust'])
            if improved_cost<burden-1e-6:
                try:geometry(p.configuration(proposed))
                except ValueError:pass
                else:fixed,trim,burden=proposed,improved,improved_cost
        previous_output=fixed.copy()
        nodes[float(angle)]=p.configuration(fixed)
        signature=fixed[gi].tobytes()
        if not geometry_symmetry or signature not in geometry_directions:
            geometry(nodes[float(angle)]);geometry_directions.add(signature)
        intermediate.append(dict(angle=float(angle),burden=burden))
        # Each finalized component is immutable. Its cost bounds the eventual
        # MAX output of this fixed procedure, even if later checks reject it.
        check_output_bound(burden,dict(angle=float(angle),kind='finalized_work_pose'))
    angles=np.array(sorted(nodes));reference=np.array([nodes[a] for a in angles])
    curve.set_directions(angles,reference)
    for index,field,trimfield in [(0,'opened','empty_trim'),(-1,'final_open','final_trim')]:
        qp=pin.neutral(p.pin);qp[pq]=reference[index][mq]
        world_normals=np.asarray(plan.normals)@pin.Quaternion(qp[p.meta['object']['q']+3:p.meta['object']['q']+7]).matrix().T
        away=separating_translation(world_normals,.05)
        if away is not None:
            opened=qp.copy();opened[:3]+=away
            trim=static_balance(p.pin,p.meta,opened,payload_fraction=0.,direct_burden=True)
        else:
            pose=spec.pose(float(angles[index]));source=np.r_[pose[:3],pose[4:],pose[3]]
            # The executed closed pose is already fixed and screened. Refit
            # only its opening, not a different closed material equilibrium.
            opened=open_material_posture(p,spec.shape,source,qp,plan.surfaces,
                np.asarray(plan.normals),{})
            opened,trim=free_flight_trim(p.pin,p.meta,opened)
        if trim['status']!='static_screen_pass':raise ValueError('endpoint flight direction unresolved')
        setattr(plan,field,opened);setattr(plan,trimfield,trim)
        plan.burden=max(plan.burden,hover_input_burden(trim['thrust']))
    plan.closed=pin.neutral(p.pin);plan.closed[pq]=reference[0][mq]
    plan.angles=angles;plan.reference_path=reference;plan.curve_record=curve.record()
    plan.direction_repairs=repairs
    for a,b in [(p.configuration(plan.opened),reference[0]),(reference[-1],p.configuration(plan.final_open))]:
        delta=np.zeros(p.mj.nv);mujoco.mj_differentiatePos(p.mj,delta,1.,a,b)
        for fraction in [0.,.25,.5,.75,1.]:
            q=a.copy();mujoco.mj_integratePos(p.mj,q,delta,fraction);geometry(q)
    plan.burden=max(target_cost,hover_input_burden(plan.empty_trim['thrust']),hover_input_burden(plan.final_trim['thrust']),max(r['burden'] for r in intermediate))
    intermediate.extend(dict(kind='load_transfer_endpoint',burden=value) for value in boundary_costs)
    plan.intermediate_loads=intermediate;plan.geometry_samples=checked
    return plan
