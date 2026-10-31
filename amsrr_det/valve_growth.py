"""Task-load ranking for noncontact branches, before a completed tree exists.

Rank the worst motor capacity multiplier at three contact-path poses, including
free-flight ends. Rotor directions are relaxed to sampled rays; all transmitted
joint efforts and self weight remain. This directs generation, never accepts
a structure or proves a physical bound. Final actual-direction checks remain.
"""
from types import SimpleNamespace
from functools import lru_cache
import numpy as np
import pinocchio as pin
from .model import build_model
from .grasp_task import material_force_contacts
from .grasp_geometry import aim_rotors_up
from .rotor_distribution import rotor_action_basis,rotor_basis_actions
from .equilibrium import static_balance
from .planning_memo import evaluate
from .mujoco_model import CONTACT_CACHE


@lru_cache(maxsize=8)
def direction_facets(lower,upper):
    """The same five-ray convex hull, represented by its planar boundary."""
    from scipy.spatial import ConvexHull
    angles=np.linspace(lower,upper,5)
    points=np.vstack([np.c_[np.cos(angles),np.sin(angles)],[0.,0.]])
    return ConvexHull(points).equations


def support_score(spec,edges,surfaces,root,values):
    inputs=dict(operation='valve_support_growth',task=spec.__dict__,edges=edges,
        surfaces=surfaces,root=np.asarray(root).tolist(),values=values)
    return evaluate(inputs,lambda:_score(spec,edges,surfaces,root,values))


def _score(spec,edges,surfaces,root,values):
    from .valve_plan import bearing_support
    model,meta=build_model(len(values),edges,spec.payload)
    meta['material_mesh_cache']=str(CONTACT_CACHE.resolve())
    q=pin.neutral(model);q[:3]=root[:3,3];q[3:7]=pin.Quaternion(root[:3,:3]).coeffs()
    pose=spec.pose(spec.target_angle/2);q[meta['object']['q']:]=np.r_[pose[:3],pose[4:],pose[3]]
    for a in meta['actuators']:
        module,name=a['name'].split('/');q[a['q']]=values[int(module[1:])].get(name,0.)
    q=aim_rotors_up(model,meta,q);p=SimpleNamespace(pin=model,meta=meta)
    actuators=[meta['actuators'][r['gimbal']] for r in meta['rotors']]
    qi=[a['q'] for a in actuators]
    from .valve_plan import rotate_assembly,gravity_axis_symmetry
    mid_contacts=material_force_contacts(p,spec.shape,q,surfaces,force_limit=None,material_tangents=True)
    symmetry=(gravity_axis_symmetry(p,spec) and all('tangent' in c for c in mid_contacts))
    if symmetry:path=[q]
    else:
        # Rank a partial tree by its rigidly transported contact posture.
        # This preserves every material contact exactly, while exposing the
        # changing gravity load. It does not assume that the final motion
        # stays rigid: the full planner may deform and rechecks that path.
        path=[rotate_assembly(p,spec,q,angle,spec.target_angle/2)
              for angle in (0.,spec.target_angle/2,spec.target_angle)]
    reports=[];middle=len(path)//2
    for index in ([middle] if symmetry else [0,middle,len(path)-1]):
        q=path[index];rays=[]
        basis=rotor_action_basis(model,meta,q)
        intervals={(a['lower']+.02,a['upper']-.02) for a in actuators}
        if np.max(abs(basis[0]))<1e-14 and len(intervals)==1:
            directions=dict(rotor_polytope=(np.asarray(basis[1:]),direction_facets(*next(iter(intervals)))))
        else:
            for angles in np.linspace([a['lower']+.02 for a in actuators],[a['upper']-.02 for a in actuators],5):
                rays.append(rotor_basis_actions(basis,angles))
            directions=dict(rotor_rays=np.asarray(rays))
        contacts=mid_contacts if index==middle else material_force_contacts(p,spec.shape,q,surfaces,force_limit=None,material_tangents=True)
        for free in ([False,True] if symmetry or index in [0,len(path)-1] else [False]):
            report=static_balance(model,meta,q,() if free else contacts,payload_fraction=0.,
                object_wrench=None if free else np.r_[np.zeros(3),np.asarray(spec.axis)*np.sign(spec.target_angle)*spec.torque*1.1],
                object_support=None if free else bearing_support(spec),**directions,
                relaxed_actuator_scale=True,fast_lp=True)
            reports.append(dict(index=index,free=free,score=report.get('minimum_peak_effort_ratio',100.),
                                numerical_solution=report['status']=='direction_proposal'))
    return dict(score=float(max(r['score'] for r in reports)),scope='three_pose_partial_motor_scale_not_feasibility',
        states=reports,gravity_axis_symmetry=symmetry,partial_motion='rigid_contact_transport',
        numerical_solution=all(r['numerical_solution'] for r in reports),modules=len(values),torque_Nm=spec.torque)
