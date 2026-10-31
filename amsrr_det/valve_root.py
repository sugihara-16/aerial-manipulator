"""Task-load ranking of single-module root placements before tree growth.

The same contact-geometry continuation supplies three representative poses. Rotor
rays only provide a relaxed ranking; actual common directions are attempted
at load-transfer endpoints. Neither this ranking nor a failed nonlinear
search proves full-path feasibility or physical impossibility.
"""
from types import SimpleNamespace
import numpy as np
import pinocchio as pin
from .model import build_model
from .task_structure import part_frames
from .grasp_geometry import grasp_frame,aim_rotors_up,fit_contacts
from .grasp_task import material_force_contacts,refine_material_posture
from .rotor_distribution import rotor_action_basis,rotor_basis_actions
from .equilibrium import static_balance
from .valve_trim import load_span_trim
from .planning_memo import evaluate
from .structure_bounds import hover_input_burden
from .mujoco_model import CONTACT_CACHE


def task_root_score(spec,point,normal,port,angle_index,sign,rotation):
    inputs=dict(operation='valve_root_task_score',task=spec.__dict__,point=np.asarray(point).tolist(),
                normal=np.asarray(normal).tolist(),port=port,
                angle_index=angle_index,sign=sign,rotation=np.asarray(rotation).tolist())
    # This complete one-module model ranks one-module proposals only.
    # A partial tree cannot be required to support the whole task alone.
    return evaluate(inputs,lambda:_score(spec,
                        point,normal,port,angle_index,sign,rotation))


def _score(spec,point,normal,port,k,sign,rotation):
    from .valve_plan import bearing_support
    # The model is immutable; each call owns its kinematic data and angles.
    model,meta=build_model(1,[],spec.payload);p=SimpleNamespace(pin=model,meta=meta)
    meta['material_mesh_cache']=str(CONTACT_CACHE.resolve())
    ports,angles,_,_,names,*_=part_frames()
    a=next(a for a in meta['actuators'] if a['name']=='m0/'+names[port])
    vectoring=[a for a in meta['actuators'] if a['role']=='vectoring']
    qi=[a['q'] for a in vectoring];oq=meta['object']['q']
    sector=np.linspace([a['lower']+.02 for a in vectoring],[a['upper']-.02 for a in vectoring],9)
    midpoint=spec.target_angle/2;axis=np.asarray(spec.axis);surface=[(0,ports[port])]
    from .valve_plan import posture_path,mounting_obstacles
    _,_,_,pads,*_=part_frames()
    def geometry():
        pose=spec.pose(midpoint);source=np.r_[pose[:3],pose[4:],pose[3]]
        object_R=pin.Quaternion(source[3:]).matrix();target=pose[:3]+object_R@(np.asarray(point)+.001*np.asarray(normal))
        q=pin.neutral(model);q[oq:oq+7]=source;q[a['q']]=angles[k]
        q[:3]=target-rotation@pads[port,k,:3,3];q[3:7]=pin.Quaternion(rotation).coeffs()
        obstacles=mounting_obstacles(spec)
        q,_=fit_contacts(model,meta,q,surface,[target],[object_R@normal],object_shape=spec.shape,
            floor_clearance=.03,base_rotation=True,obstacles=obstacles,max_evaluations=45,avoid_object=True)
        _,q,_=refine_material_posture(p,spec.shape,source,q,surface,np.asarray([point]),np.asarray([normal]),open_posture=False,obstacles=obstacles)
        plan=SimpleNamespace(system=SimpleNamespace(control=p,spec=spec),closed=aim_rotors_up(model,meta,q),surfaces=surface)
        try:
            _,path,_=posture_path(plan,steps=13,force_screen=False)
            return path[[0,6,12]]
        except ValueError:return None
    geometry_inputs=dict(operation='root_contact_geometry',task={k:v for k,v in spec.__dict__.items() if k!='torque'},
        point=np.asarray(point).tolist(),normal=np.asarray(normal).tolist(),port=port,angle_index=k,rotation=np.asarray(rotation).tolist())
    def checked_geometry():
        try:return geometry()
        except ValueError:return None
    poses=evaluate(geometry_inputs,checked_geometry)
    if poses is None:return dict(score=24.,common_endpoint_unresolved=2,task_torque_Nm=spec.torque,
        scope='root_geometry_not_found_not_physical_impossibility')
    relaxed=[];actual=[];unresolved=0
    hover_unit=hover_input_burden(np.full(4,(meta['mass']-meta['object']['mass'])*9.80665/4))
    for theta,q in zip([0.,midpoint,spec.target_angle],poses):
        rays=[];basis=rotor_action_basis(model,meta,q)
        for gs in sector:
            rays.append(rotor_basis_actions(basis,gs))
        contacts=material_force_contacts(p,spec.shape,q,surface,force_limit=None,material_tangents=True)
        settings=dict(payload_fraction=0.,rotor_rays=np.asarray(rays),rotor_direction_cost=np.ones((4,9)),fast_lp=True)
        load=static_balance(model,meta,q,contacts,object_support=bearing_support(spec),
             object_wrench=np.r_[np.zeros(3),axis*np.sign(spec.target_angle)*spec.torque*1.1],**settings)
        def burden(result):
            coefficients=result.get('rotor_coefficients')
            return 20*hover_unit if coefficients is None else hover_input_burden(np.sum(coefficients,axis=1))
        relaxed.append(burden(load))
        if theta in [0.,spec.target_angle]:
            free=static_balance(model,meta,q,**settings);relaxed.append(burden(free))
            if 'rotor_coefficients' not in load or 'rotor_coefficients' not in free:
                unresolved+=1
                continue
            _,envelope=load_span_trim(p,spec,aim_rotors_up(model,meta,q),surface,0.,1.1,free_lower=True)
            if envelope['free']==envelope['load']=='static_screen_pass':
                actual.extend(hover_input_burden(t['thrust']) for t in envelope['trims'])
            else:unresolved+=1
    # One natural hover burden normalizes the dimensional input cost.
    # An unresolved common-direction endpoint is a finite preference penalty,
    # not a claim that extra modules or another solver cannot support it.
    return dict(score=max(relaxed+actual)/hover_unit+2*unresolved,
                relaxed_burden=max(relaxed),common_endpoint_unresolved=unresolved,
                task_torque_Nm=spec.torque,
                scope='three_pose_root_proposal_not_full_structure_certificate')
