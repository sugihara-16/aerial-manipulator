"""Preparation of v0.2 C.2/C.4 maps, not a contact-success detector.

All coordinates and material points come from the candidate geometry. No true
contact force is an input. A full-rank map is necessary for the selected task
coordinates, but it neither establishes sticking contact nor actuator authority.
"""
from dataclasses import dataclass
import numpy as np
import pinocchio as pin
from .equilibrium import point_jacobian


@dataclass
class ContactMap:
    object_velocity: np.ndarray
    robot_velocity: np.ndarray
    task_from_robot: np.ndarray
    body_columns: int
    dock_names: list
    rank: int
    required_rank: int
    singular_values: np.ndarray
    joint_basis: np.ndarray
    joint_correction: np.ndarray

    @property
    def supports_task_directions(self):
        return self.rank==self.required_rank

    def compatible_velocity(self,robot_velocity):
        value=np.asarray(robot_velocity,dtype=float)
        y=self.task_from_robot@value
        return y,self.object_velocity@y-self.robot_velocity@value

    def correction(self,object_wrench):
        if not self.supports_task_directions:
            raise ValueError('required object motion directions are not determined by the sticking geometry')
        force=np.asarray(object_wrench,dtype=float)
        body=self.task_from_robot[:,:self.body_columns].T@force
        coefficients=self.joint_correction@force
        return body,coefficients


def prepare_contact_map(model,meta,q,contacts,task_axes=None,rank_relative_tolerance=1e-9):
    """Map world object twist about its frame origin to world body twist + dock rates.

    Each contact supplies frame/point and target_point in the object frame.
    Multiple material points describe a finite contact patch. task_axes has six
    rows and k independent columns; k=6 by default. It cannot be reduced after a
    rank failure to claim the same task is controllable.
    """
    if meta['object'] is None or not contacts:raise ValueError('object and material contact points required')
    axes=np.eye(6) if task_axes is None else np.asarray(task_axes,dtype=float)
    if axes.ndim!=2 or axes.shape[0]!=6 or not 1<=axes.shape[1]<=6 or np.linalg.matrix_rank(axes)!=axes.shape[1]:
        raise ValueError('independent task coordinate axes required')
    if not np.all(np.isfinite(axes)) or rank_relative_tolerance<=0:raise ValueError('invalid mapping inputs')
    data=model.createData();pin.framesForwardKinematics(model,data,q)
    obj=meta['object'];docks=[a for a in meta['actuators'] if a['role']=='dock']
    body_count=6*len(meta['bases']);lift=np.zeros((model.nv,body_count+len(docks)))
    for i,b in enumerate(meta['bases']):
        R=data.oMi[b['joint']].rotation
        lift[b['v']:b['v']+3,6*i:6*i+3]=R.T
        lift[b['v']+3:b['v']+6,6*i+3:6*i+6]=R.T
    for i,a in enumerate(docks):lift[a['v'],body_count+i]=1.
    object_axes=np.zeros((model.nv,6));R=data.oMf[obj['frame']].rotation
    object_axes[obj['v']:obj['v']+3,:3]=R.T
    object_axes[obj['v']+3:obj['v']+6,3:]=R.T
    O=[];P=[];normal_rows=[]
    for c in contacts:
        J=point_jacobian(model,data,q,meta['frames'][c['frame']],c['point'])
        JO=point_jacobian(model,data,q,obj['frame'],c['target_point'])
        O.append(JO@object_axes@axes);P.append(J@lift)
        n=np.asarray(c['normal'],dtype=float)
        if n.shape!=(3,) or abs(np.linalg.norm(n)-1)>1e-9:raise ValueError('unit world normal required')
        normal_rows.append(n@(J@lift)[:,body_count:])
    O=np.vstack(O);P=np.vstack(P)
    U,s,Vh=np.linalg.svd(O,full_matrices=False)
    threshold=s[0]*rank_relative_tolerance if s.size else 0.
    keep=s>threshold;rank=int(np.sum(keep))
    inverse=(Vh[keep].T/s[keep])@U[:,keep].T if rank else np.zeros((axes.shape[1],O.shape[0]))
    A=inverse@P
    stiffness=np.array([a['kp'] for a in docks])
    candidates=np.column_stack([np.array(normal_rows).T,A[:,body_count:].T/stiffness[:,None]])
    ub,sb,_=np.linalg.svd(candidates,full_matrices=False)
    basis=ub[:,sb>max(sb[0]*rank_relative_tolerance,1e-14)] if sb.size else np.zeros((len(docks),0))
    metric=basis.T@(stiffness[:,None]*basis)
    correction=np.linalg.solve(metric,basis.T@A[:,body_count:].T) if basis.shape[1] else np.zeros((0,axes.shape[1]))
    return ContactMap(O,P,A,body_count,[a['name'] for a in docks],rank,axes.shape[1],s,basis,correction)


def critical_damping_step(position,velocity,target,frequency,period):
    """Exact held-target update of z''+2*w*z'+w^2*(z-target)=0.

    The caller owns history, target bounds and the 100 Hz clock. Updating this
    at the physics-step rate would implement a different controller.
    """
    if frequency<=0 or period<=0:raise ValueError('positive frequency and control period required')
    delta=np.asarray(position)-target;velocity=np.asarray(velocity)
    decay=np.exp(-frequency*period)
    return (target+decay*((1+frequency*period)*delta+period*velocity),
            decay*((1-frequency*period)*velocity-frequency**2*period*delta))
