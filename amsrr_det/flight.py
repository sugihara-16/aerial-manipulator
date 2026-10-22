"""Observation-only flight control, with physical rotor/gimbal allocation.

Payload feed-forward is a declared model input. No simulator contact result is
read here. The robot is propelled only by its four rotor forces per module.
"""
import numpy as np
import pinocchio as pin
from scipy.optimize import lsq_linear


def cross3(a,b):
    """Cross product for equal-shaped float arrays with last dimension three."""
    result=np.empty_like(a)
    result[...,0]=a[...,1]*b[...,2]-a[...,2]*b[...,1]
    result[...,1]=a[...,2]*b[...,0]-a[...,0]*b[...,2]
    result[...,2]=a[...,0]*b[...,1]-a[...,1]*b[...,0]
    return result


class FlightController:
    def __init__(self, model, meta, q0, period=.01, preserve_joint_action=False, guard_vectoring=False, measured_thrust=False, preserve_requested_action=False):
        self.model,self.meta,self.dt=model,meta,period
        self.data=model.createData()
        self.actuators=meta['actuators'];self.rotors=meta['rotors']
        self.nr=len(self.rotors)
        ancestors=[]
        for rotor in self.rotors:
            ancestor=self.actuators[rotor['gimbal']]['joint']
            while model.parents[ancestor]!=0:ancestor=model.parents[ancestor]
            ancestors.append(ancestor)
        self.rotor_groups={b['joint']:np.asarray(ancestors)==b['joint'] for b in meta['bases']}
        self.ai=np.array([r['gimbal'] for r in self.rotors])
        self.qi=np.array([a['q'] for a in self.actuators])
        self.vi=np.array([a['v'] for a in self.actuators])
        self.kp=np.array([a['kp'] for a in self.actuators])
        self.kd=np.array([a['kd'] for a in self.actuators])
        self.limits=np.array([[a['lower'],a['upper']] for a in self.actuators])
        self.speed=np.array([a['speed'] for a in self.actuators])
        self.gimbal_effort=np.array([self.actuators[i]['effort'] for i in self.ai])
        self.force_maximum=np.array([r['maximum'] for r in self.rotors])
        self.gimbal_margin=.01+self.gimbal_effort/self.kp[self.ai]
        self.allocation_groups={}
        for base,active in self.rotor_groups.items():
            combined=np.r_[active,active];ids=np.flatnonzero(combined)
            scale=np.r_[np.full(self.nr,5.),np.full(self.nr,.5)][ids]
            self.allocation_groups[base]=(combined,ids,scale,np.eye(len(ids))*.001)
        self.command=q0[self.qi].copy()
        self.force=np.full(self.nr,(meta['mass']-(meta['object'] or {}).get('mass',0))*9.80665/self.nr)
        self.dock=np.array([a['role']=='dock' for a in self.actuators])
        self.control_count=0
        self.trim_effort=None
        self.allocation_diagnostics=[]
        self.preserve_joint_action=preserve_joint_action
        self.preserve_requested_action=preserve_requested_action
        self.requested_action_diagnostics=None
        self.distribution_diagnostics=None
        self.guard_vectoring=guard_vectoring;self.allocation_failure=None
        self.measured_thrust=measured_thrust
        self.vectoring_guard_checks=0;self.maximum_vectoring_request_ratio=0.;self.vectoring_guard_diagnostics=None
        self.command_diagnostics=None

    def update(self,q,v,target, *, target_velocity=None, target_acceleration=None,
               payload_mass=0., payload_com=None, joint_bias=None, thrust_reference=None, gimbal_reference=None):
        m,d=self.model,self.data
        if self.preserve_requested_action:
            previous_command,previous_force=self.command.copy(),self.force.copy()
        self.allocation_diagnostics=[];self.allocation_failure=None
        self.baseline_thrust_before_measured=None
        vector_bounds=None
        if self.guard_vectoring:
            kp,kd=self.kp[self.ai],self.kd[self.ai]
            qg,vg=q[self.qi[self.ai]],v[self.vi[self.ai]]
            effort=self.gimbal_effort
            previous=kp*(self.command[self.ai]-qg)-kd*vg
            vector_bounds=(np.minimum(-effort,previous),np.maximum(effort,previous))
        tv=np.zeros(m.nv) if target_velocity is None else target_velocity
        ta=np.zeros(m.nv) if target_acceleration is None else target_acceleration
        pin.forwardKinematics(m,d,q,v);pin.updateFramePlacements(m,d)
        acceleration=ta.copy()
        for b in self.meta['bases']:
            k,j=b['q'],b['v'];R=d.oMi[b['joint']].rotation
            Rref=pin.Quaternion(target[k+3:k+7]).matrix()
            vw=R@v[j:j+3];omega=R@v[j+3:j+6]
            acceleration[j:j+3]=R.T@(ta[j:j+3]+4*(target[k:k+3]-q[k:k+3])+4*(tv[j:j+3]-vw))-cross3(v[j+3:j+6],v[j:j+3])
            acceleration[j+3:j+6]=R.T@(ta[j+3:j+6]+16*pin.log3(Rref@R.T)+8*(tv[j+3:j+6]-omega))
        required=pin.rnea(m,d,q,v,acceleration).copy()
        positions=np.array([d.oMf[r['frame']].translation for r in self.rotors])
        directions=np.array([d.oMf[r['frame']].rotation[:,2] for r in self.rotors])
        axes=np.array([d.oMi[self.actuators[r['gimbal']]['joint']].rotation[:,0] for r in self.rotors])
        pivots=np.array([d.oMi[self.actuators[r['gimbal']]['joint']].translation for r in self.rotors])
        inputs=np.r_[self.force,self.command[self.ai]]
        # Predictor, not hidden plant state. Same .05 s nominal parameter as C++.
        ag=1-.05/self.dt*(1-np.exp(-self.dt/.05))
        for base in self.meta['bases']:
            R=d.oMi[base['joint']].rotation;j=base['v'];origin=q[base['q']:base['q']+3]
            target_wrench=np.r_[R@required[j:j+3],R@required[j+3:j+6]]
            if payload_mass:
                f=np.array([0.,0.,payload_mass*9.80665])
                target_wrench+=np.r_[f,cross3(np.asarray(payload_com)-origin,f)]
            active=self.rotor_groups[base['joint']]
            combined,ids,scale,regularizer=self.allocation_groups[base['joint']]
            for _ in range(2):
                angle=ag*(inputs[self.nr:]-q[self.qi[self.ai]])
                c,s=np.cos(angle)[:,None],np.sin(angle)[:,None]
                rotate=lambda x:c*x+s*cross3(axes,x)+(1-c)*axes*np.sum(axes*x,axis=1)[:,None]
                direction=rotate(directions);relative=rotate(positions-pivots)
                arm=pivots+relative-origin
                fw=inputs[:self.nr,None]*direction
                wrench=np.r_[fw[active].sum(axis=0),cross3(arm,fw)[active].sum(axis=0)]
                df=ag*inputs[:self.nr,None]*cross3(axes,direction)
                dp=ag*cross3(axes,relative)
                B=np.vstack([np.hstack([direction.T,df.T]),
                             np.hstack([cross3(arm,direction).T,(cross3(dp,fw)+cross3(arm,df)).T])])
                B[:,~combined]=0
                # Solve with the actual nonnegative thrust and angle/rate
                # bounds. Clipping an unconstrained wrench allocation destroys
                # its balance, notably when a planned rotor trim is near zero.
                total_mass=self.meta['mass']-(self.meta['object'] or {}).get('mass',0)+payload_mass
                reference=np.r_[np.full(self.nr,total_mass*9.80665/self.nr) if thrust_reference is None else thrust_reference,np.zeros(self.nr) if gimbal_reference is None else gimbal_reference]
                margin=self.gimbal_margin
                angle_low=np.maximum(self.limits[self.ai,0]+margin,self.command[self.ai]-self.speed[self.ai]*self.dt)
                angle_high=np.minimum(self.limits[self.ai,1]-margin,self.command[self.ai]+self.speed[self.ai]*self.dt)
                if vector_bounds is not None:
                    angle_low=np.maximum(angle_low,qg+(vector_bounds[0]+kd*vg)/kp)
                    angle_high=np.minimum(angle_high,qg+(vector_bounds[1]+kd*vg)/kp)
                    if np.any(angle_low>angle_high):
                        self.allocation_failure='vectoring_command_bounds_incompatible'
                        return self.command.copy(),self.force.copy()
                lower=np.r_[np.zeros(self.nr),angle_low]
                upper=np.r_[self.force_maximum,angle_high]
                linear=B[:,ids];epsilon=.001
                A=np.vstack([linear*scale,regularizer])
                rhs=np.r_[target_wrench-wrench+linear@inputs[ids],epsilon*reference[ids]/scale]
                allocation=lsq_linear(A,rhs,bounds=(lower[ids]/scale,upper[ids]/scale),method='bvls',tol=1e-8,max_iter=50)
                inputs[ids]=allocation.x*scale
                self.allocation_diagnostics.append(dict(min_force=float(inputs[:self.nr].min()),max_force=float(inputs[:self.nr].max()),target=target_wrench.tolist(),prediction=wrench.tolist(),solver_status=int(allocation.status)))
            if self.measured_thrust:
                # Keep the same bounded gimbal update, but issue thrust using
                # the observed rotor directions. The predicted future angle
                # cannot supply force in the present control interval.
                self.baseline_thrust_before_measured=inputs[:self.nr].copy()
                ids=np.flatnonzero(active)
                current=np.vstack([directions[ids].T,cross3(positions[ids]-origin,directions[ids]).T])
                matrix=np.vstack([current*5.,np.eye(len(ids))*.001])
                rhs=np.r_[target_wrench,.001*reference[ids]/5.]
                allocation=lsq_linear(matrix,rhs,bounds=(np.zeros(len(ids)),np.array([self.rotors[i]['maximum'] for i in ids])/5.),method='bvls',tol=1e-8,max_iter=50)
                inputs[ids]=allocation.x*5.
                self.allocation_diagnostics[-1]['measured_thrust_solver_status']=int(allocation.status)
            # Record the final command's prediction, not the previous internal
            # iterate. The measured-angle action is a separate quantity.
            angle=ag*(inputs[self.nr:]-q[self.qi[self.ai]])
            c,s=np.cos(angle)[:,None],np.sin(angle)[:,None]
            direction=rotate(directions);relative=rotate(positions-pivots)
            fw=inputs[:self.nr,None]*direction
            predicted_wrench=np.r_[fw[active].sum(axis=0),cross3(pivots+relative-origin,fw)[active].sum(axis=0)]
            actual_fw=inputs[:self.nr,None]*directions
            measured_wrench=np.r_[actual_fw[active].sum(axis=0),cross3(positions-origin,actual_fw)[active].sum(axis=0)]
            self.allocation_diagnostics[-1].update(prediction=predicted_wrench.tolist(),residual=(predicted_wrench-target_wrench).tolist(),measured_angle_action=measured_wrench.tolist())
        if self.preserve_joint_action and thrust_reference is not None:
            from .rotor_distribution import rotor_actions,preserve_wrench_distribution
            # Leave the baseline gimbal commands unchanged. Preserve both the
            # current measured-angle wrench and the existing delayed-gimbal
            # predictor's wrench; neither is traded for lower joint deviation.
            action=rotor_actions(m,self.meta,q)
            predicted=q.copy();predicted[self.qi[self.ai]]+=ag*(inputs[self.nr:]-q[self.qi[self.ai]])
            predicted_action=rotor_actions(m,self.meta,predicted)
            base_indices=np.concatenate([np.arange(b['v'],b['v']+6) for b in self.meta['bases']])
            base_action=np.vstack([action[base_indices],predicted_action[base_indices]])
            inputs[:self.nr],self.distribution_diagnostics=preserve_wrench_distribution(
                base_action,action[self.vi[self.dock]],inputs[:self.nr],np.asarray(thrust_reference),
                np.array([r['maximum'] for r in self.rotors]),
                np.array([a['effort'] for a in self.actuators])[self.dock])
        if vector_bounds is not None:
            predicted=kp*(inputs[self.nr:]-qg)-kd*vg
            valid=bool(np.all(predicted>=vector_bounds[0]-1e-9) and np.all(predicted<=vector_bounds[1]+1e-9))
            self.vectoring_guard_diagnostics=dict(feasible=valid,previous_request=previous.tolist(),next_request=predicted.tolist())
            if not valid:
                self.allocation_failure='vectoring_issued_command_violation'
                return self.command.copy(),self.force.copy()
            self.vectoring_guard_checks+=1
            self.maximum_vectoring_request_ratio=max(self.maximum_vectoring_request_ratio,float(np.max(abs(predicted)/effort)))
        self.command_diagnostics=dict(
            observed_configuration=q.tolist(),observed_velocity=v.tolist(),
            previous_thrust=self.force.tolist(),target=target.tolist(),
            baseline_thrust_before_measured=None if self.baseline_thrust_before_measured is None else self.baseline_thrust_before_measured.tolist(),
            target_acceleration=acceleration.tolist(),
            joint_bias=None if joint_bias is None else np.asarray(joint_bias).tolist(),
            thrust_reference=None if thrust_reference is None else np.asarray(thrust_reference).tolist(),
            payload_mass=float(payload_mass),payload_com=None if payload_com is None else np.asarray(payload_com).tolist(),
            observed_gimbal_position=q[self.qi[self.ai]].tolist(),
            observed_gimbal_velocity=v[self.vi[self.ai]].tolist(),
            previous_gimbal_command=self.command[self.ai].tolist(),
            issued_gimbal_command=inputs[self.nr:].tolist(),
            raw_request_before_physics=(self.kp[self.ai]*(inputs[self.nr:]-q[self.qi[self.ai]])-self.kd[self.ai]*v[self.vi[self.ai]]).tolist(),
            allocation=self.allocation_diagnostics)
        self.force=inputs[:self.nr]
        self.command[self.ai]=inputs[self.nr:]
        # Initial stationary trim. Recomputing a bias from rapidly varying applied
        # rotor commands creates a second feedback loop; use planned loads instead.
        if self.trim_effort is None:
            ext=[pin.Force.Zero() for _ in range(m.njoints)]
            for r,f in zip(self.rotors,self.force):
                frame=m.frames[r['frame']]
                ext[frame.parentJoint]+=frame.placement.act(pin.Force(np.array([0.,0.,f]),np.zeros(3)))
            self.trim_effort=pin.rnea(m,d,q,np.zeros(m.nv),np.zeros(m.nv),ext)[self.vi].copy()
        effort=self.trim_effort if joint_bias is None else np.asarray(joint_bias)
        self.last_joint_bias=effort.copy();self.last_target_velocity=tv.copy()
        desired=target[self.qi]+(effort+self.kd*tv[self.vi])/self.kp
        # Drive saturation remains in the plant. References never bypass limits.
        self.command[self.dock]=np.clip(desired[self.dock],self.limits[self.dock,0],self.limits[self.dock,1])
        if self.preserve_requested_action:
            from .rotor_distribution import rotor_actions,preserve_requested_action
            action=rotor_actions(m,self.meta,q)
            base_indices=np.concatenate([np.arange(b['v'],b['v']+6) for b in self.meta['bases']])
            preserved=np.vstack([action[base_indices],action[self.vi[~self.dock]]])
            command,force,info=preserve_requested_action(preserved,action[self.vi[self.dock]],
                self.force,self.command[self.dock],q[self.qi[self.dock]],v[self.vi[self.dock]],
                self.kp[self.dock],self.kd[self.dock],
                np.array([a['effort'] for a in self.actuators])[self.dock],self.limits[self.dock],
                np.array([r['maximum'] for r in self.rotors]))
            self.requested_action_diagnostics=info
            if force is None:
                self.allocation_failure='requested_action_allocation_not_found'
                self.command,self.force=previous_command,previous_force
            else:
                self.command[self.dock]=command;self.force=force
        self.control_count+=1
        return self.command.copy(),self.force.copy()
