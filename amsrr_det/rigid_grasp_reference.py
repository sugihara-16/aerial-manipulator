"""Keep a qualified Dock reference and move the whole grasp by its base.

This is an experimental alternative to integrating each pad's closing motion.
Joints remain physical position-controlled joints with the original drive limits.
"""
import numpy as np
import pinocchio as pin
from .grasp_control import GraspReferences
from .numerics import advance_world_orientation
from .operating_transfer import pose


class RigidGraspReferences(GraspReferences):
    def configure_operating_point(self,parameters):
        self.operating={k:np.asarray(v) for k,v in parameters.items()}
        p=self.operating
        self.goal_star=pose(p['object_goal']);self.base_star=pose(p['base_pose'])
        obj_star=pose(p['object_pose'])
        error=np.r_[self.goal_star.translation-obj_star.translation,
            pin.log3(self.goal_star.rotation@obj_star.rotation.T)]
        correction=np.r_[[1.]*3,[.8]*3]*error-.3*p['object_world_velocity']
        lever=self.base_star.translation-obj_star.translation
        mapped=np.r_[correction[:3]+np.cross(correction[3:],lever),correction[3:]]
        # Express feedback about the same nonzero-velocity operating point.
        # This is determined by the planned state, not a new fitted bias.
        self.steady_velocity_offset=p['base_world_velocity']-mapped
        self.bridge_start=self.reference[self.qi].copy()
        self.bridge_velocity=self.last_velocity[6:].copy()
        self.elapsed=0.;self.previous_goal=None

    def update(self,q,v,object_target, *, desired_twist=None,preload=None,
               orientation_feedback=1.,object_feedback=1.,drive_guard=None,**unused):
        p=self.operating;dt=self.dt;oq,ov=self.meta['object']['q'],self.meta['object']['v']
        obj=pose(q[oq:oq+7]);goal=pose(object_target)
        velocity=np.r_[obj.rotation@v[ov:ov+3],obj.rotation@v[ov+3:ov+6]]
        error=np.r_[goal.translation-obj.translation,pin.log3(goal.rotation@obj.rotation.T)]
        previous_goal=goal if self.previous_goal is None else self.previous_goal
        vd=np.r_[(goal.translation-previous_goal.translation)/dt,
            pin.log3(goal.rotation@previous_goal.rotation.T)/dt]
        if desired_twist is not None:vd+=desired_twist
        transform=goal*self.goal_star.inverse();base_ff=transform*self.base_star
        common=np.r_[vd[:3]+np.cross(vd[3:],base_ff.translation-goal.translation),vd[3:]]
        feedback=np.r_[[1.]*3,[.8]*3]*error-.3*(velocity-vd)
        feedback*=object_feedback;feedback[3:]*=orientation_feedback
        common+=np.r_[feedback[:3]+np.cross(feedback[3:],self.reference[:3]-obj.translation),feedback[3:]]
        common+=np.r_[transform.rotation@self.steady_velocity_offset[:3],transform.rotation@self.steady_velocity_offset[3:]]
        limit=np.r_[[.1]*3,[.3]*3]*dt
        nu=np.clip(common,np.maximum(-self.scale[:6],self.last_velocity[:6]-limit),
                          np.minimum(self.scale[:6],self.last_velocity[:6]+limit))
        target=self.reference.copy();target[:3]+=nu[:3]*dt
        target[3:7]=advance_world_orientation(target[3:7],nu[3:]*dt)
        u=min(1.,self.elapsed/2.);s=u**3*(10+u*(-15+6*u))
        h=u-6*u**3+8*u**4-3*u**5
        target[self.qi]=(1-s)*self.bridge_start+s*p['joint_reference']+2*h*self.bridge_velocity
        joint_velocity=(p['joint_reference']-self.bridge_start)*(30*u*u*(1-u)**2/2.)
        joint_velocity+=(1-18*u*u+32*u**3-15*u**4)*self.bridge_velocity
        old_error=np.linalg.norm(self.reference[:3]-q[:3]);new_error=np.linalg.norm(target[:3]-q[:3])
        stalled=new_error>.05 and new_error>old_error+1e-10
        if stalled:target[:7]=self.reference[:7];nu[:]=0.
        valid=bool(np.all(abs(joint_velocity)<=self.scale[6:]+1e-10))
        valid &= bool(np.all(target[self.qi]>=[a['lower']+.025 for a in self.docks]) and
                      np.all(target[self.qi]<=[a['upper']-.025 for a in self.docks]))
        if drive_guard is not None:
            bias,previous_command=drive_guard
            kp=np.array([a['kp']for a in self.docks]);kd=np.array([a['kd']for a in self.docks])
            effort=np.array([a['effort']for a in self.docks]);old=kp*(previous_command-q[self.qi])-kd*v[self.vi]
            requested=kp*(target[self.qi]-q[self.qi])+bias+kd*(joint_velocity-v[self.vi])
            valid &= bool(np.all(requested>=np.minimum(-effort,old)-1e-9) and
                          np.all(requested<=np.maximum(effort,old)+1e-9))
        _,_,_,depth=self.geometry(q)
        self.last_diagnostics=dict(object_error=error.tolist(),compression=depth.tolist(),preload=self.preload.tolist(),
            reference_command_feasible=valid,stalled=bool(stalled),rigid_operating_reference=True,
            joint_reference=target[self.qi].tolist(),joint_observation=q[self.qi].tolist())
        if not valid:
            tv=np.zeros(self.model.nv);tv[:6]=self.last_velocity[:6];tv[self.vi]=self.last_velocity[6:]
            return self.reference.copy(),tv
        self.reference=target;self.last_velocity=np.r_[nu,joint_velocity]
        self.previous_goal=goal;self.elapsed+=dt
        tv=np.zeros(self.model.nv);tv[:6]=nu;tv[self.vi]=joint_velocity
        return target.copy(),tv
