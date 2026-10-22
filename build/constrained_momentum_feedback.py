"""Whole-momentum input allocation with actual drive bounds in the solve.

The same physical minimum-relative-forcing objective is retained. Joint and
rotor limits are imposed before allocation, instead of clipping its columns
afterwards. Contact maintenance remains an independently audited property.
"""
import time
import numpy as np,mujoco,proxsuite
from scipy.linalg import cho_factor,cho_solve
from build.momentum_feedback import MomentumFeedback,rigid_motion_map
from build.allocation_optimality import box_objective_gap


class ConstrainedMomentumFeedback(MomentumFeedback):
    def __init__(self,*args,normal_tracking=False,internal_tracking=False,**kwargs):
        super().__init__(*args,**kwargs)
        self.normal_tracking=normal_tracking
        self.internal_tracking=internal_tracking
        pads=np.array([element[0] for element in self.local.elements])
        self.pad_average=np.array([(pads==i)/np.count_nonzero(pads==i) for i in range(len(self.local.links))])
        self.normal_reference=self.pad_average@self.local.contact()[0]
        self.active=np.r_[np.flatnonzero(self.local.dock),np.arange(self.local.nj,self.m.nu)]
        self.scale=np.r_[self.local.effort/self.local.gain,self.local.rotor_max]
        n=len(self.active);self.solver=proxsuite.proxqp.dense.QP(n,6,n)
        self.solver.settings.eps_abs=self.solver.settings.eps_rel=1e-11
        self.solver.settings.max_iter=100
        self.initialized=False;self.diagnostics={}
        self.info.update(method='bounded_whole_momentum_geometry',normal_tracking=normal_tracking,internal_tracking=internal_tracking)

    def update(self,q,v,reference,velocity=None,input_reference=None,*,path=None):
        began=time.perf_counter();m,l=self.m,self.local;ids=self.active
        if path is not None:reference,velocity=path.reference()
        velocity=np.zeros(m.nv)if velocity is None else velocity
        error=np.empty(6);oi=self.oi;error[:3]=q[oi:oi+3]-reference[oi:oi+3]
        angle=np.empty(3);rotation=np.empty(9)
        mujoco.mju_subQuat(angle,q[oi+3:oi+7],reference[oi+3:oi+7]);mujoco.mju_quat2Mat(rotation,reference[oi+3:oi+7])
        error[3:]=rotation.reshape(3,3)@angle
        momentum=self.momentum(q,v)-self.momentum(reference,velocity)
        desired=-self.outer_gain@np.r_[error,np.linalg.solve(self.inertia,momentum)]
        l.at(q,v);mass,B,_=l.matrices();T=rigid_motion_map(m,q)
        act=B[:,ids]*self.scale[ids];target=mass@T@desired
        normal_error=0.
        if self.internal_tracking:
            delta=np.empty(m.nv);mujoco.mj_differentiatePos(m,delta,1.,reference,q)
            restoring=-4.*delta-4.*(v-velocity)
            # Remove common rigid acceleration in the current inertia metric.
            # The remaining goal restores relative module/object/joint state;
            # actual actuator bounds and total-wrench constraints stay intact.
            restoring-=T@np.linalg.solve(T.T@mass@T,T.T@mass@restoring)
            target+=mass@restoring
        if self.normal_tracking:
            # A virtual restoring/damping action on observed material gaps.
            # It is an internal motion objective, not an observed contact force
            # or a modification of the evaluator's physical contact law.
            gap,_,Jn,_=l.contact();gap=self.pad_average@gap;Jn=self.pad_average@Jn
            # Set a critically damped return time of 0.5 s, the same response
            # time as acquisition. Convert this acceleration objective through
            # the current effective inertia; material stiffness is not a gain.
            normal_inertia=Jn@cho_solve(cho_factor(mass,lower=True),Jn.T)
            restoring=np.linalg.lstsq(normal_inertia,4.*(gap-self.normal_reference)+4.*(Jn@v),rcond=1e-10)[0]
            internal=-Jn.T@restoring
            if np.max(np.abs(T.T@internal))>1e-7:
                raise ValueError('normal tracking action is not internal')
            target+=internal
            normal_error=float(np.max(np.abs(gap-self.normal_reference)))
        inverse=cho_solve(cho_factor(mass,lower=True),act)
        H=act.T@inverse;gradient=-inverse.T@target
        wrench=T.T@act;requested=self.inertia@desired
        U,s,V=np.linalg.svd(wrench,full_matrices=False)
        if min(s)<max(s)*1e-10:raise ValueError('bounded momentum wrench rank')
        E=V;rhs=(U.T@requested)/s
        nominal=self.u if input_reference is None else input_reference
        offset=l.kp*q[l.qi]+l.kd*v[l.vi]
        lower=np.r_[(offset-l.effort)/l.gain,l.rotor_min]
        upper=np.r_[(offset+l.effort)/l.gain,l.rotor_max]
        limited=m.actuator_ctrllimited.astype(bool)
        lower[limited]=np.maximum(lower[limited],m.actuator_ctrlrange[limited,0])
        upper[limited]=np.minimum(upper[limited],m.actuator_ctrlrange[limited,1])
        gimbal=np.flatnonzero(~l.dock)
        if np.any(nominal[gimbal]<lower[gimbal]-1e-9)or np.any(nominal[gimbal]>upper[gimbal]+1e-9):
            raise ValueError('bounded momentum fixed gimbal outside input domain')
        lo=(lower[ids]-nominal[ids])/self.scale[ids];hi=(upper[ids]-nominal[ids])/self.scale[ids]
        path_acceleration=None
        if path is not None:
            from amsrr_det.affine_bounds import scalar_interval
            nd=int(np.count_nonzero(l.dock));direction=T.T@mass@T@path.direction6
            interval=scalar_interval(wrench[:,nd:],requested,direction,lo[nd:],hi[nd:])
            path_acceleration=path.observed_choice(interval,q,v,desired)
            target+=mass@T@path.direction6*path_acceleration
            gradient=-inverse.T@target
            requested+=direction*path_acceleration;rhs=(U.T@requested)/s
        objective_scale=max(float(H.diagonal().max()),1.)
        if self.initialized:self.solver.update(H=H/objective_scale,g=gradient/objective_scale,A=E,b=rhs,C=np.eye(len(ids)),l=lo,u=hi)
        else:
            self.solver.init(H/objective_scale,gradient/objective_scale,E,rhs,np.eye(len(ids)),lo,hi);self.initialized=True
        self.solver.solve();x=self.solver.results.x.copy()
        residual=max(float(abs(wrench@x-requested).max()),float((lo-x).max()),float((x-hi).max()))
        gap=box_objective_gap(H/objective_scale,gradient/objective_scale,E,rhs,lo,hi,x,self.solver.results.y)
        status=str(self.solver.results.info.status)
        # Check an independent convex objective-error bound on a stalled
        # iterate. The bound concerns normalized QP cost; physical tolerance
        # stays unchanged. Otherwise retain the original longer solve.
        bounded_approximation=status.endswith('PROXQP_MAX_ITER_REACHED') and np.isfinite(gap) and gap<=1e-8
        if (not status.endswith('PROXQP_SOLVED') and not bounded_approximation) or residual>1e-7:
            self.solver.settings.max_iter=10000
            self.solver.solve();x=self.solver.results.x.copy()
            self.solver.settings.max_iter=100
            residual=max(float(abs(wrench@x-requested).max()),float((lo-x).max()),float((x-hi).max()))
            gap=box_objective_gap(H/objective_scale,gradient/objective_scale,E,rhs,lo,hi,x,self.solver.results.y)
            status=str(self.solver.results.info.status)
            bounded_approximation=status.endswith('PROXQP_MAX_ITER_REACHED') and np.isfinite(gap) and gap<=1e-8
        correction_scale=1.
        if (not status.endswith('PROXQP_SOLVED') and not bounded_approximation) or not np.isfinite(residual) or residual>1e-7:
            if path is not None:
                raise ValueError('bounded momentum allocation unresolved: '+str(residual))
            from build.scaled_correction import scaled_correction
            feedback_gradient=-inverse.T@(mass@T@desired)/objective_scale
            x,correction_scale,scaled_info=scaled_correction(H/objective_scale,gradient/objective_scale,
                feedback_gradient,E,rhs,lo,hi)
            requested*=correction_scale;rhs*=correction_scale
            residual=max(float(abs(wrench@x-requested).max()),float((lo-x).max()),float((x-hi).max()))
            if not np.isfinite(residual) or residual>1e-7:
                raise ValueError('scaled momentum physical residual: '+str(residual))
            target-=mass@T@desired*(1.-correction_scale)
            gap=scaled_info['normalized_objective_gap_bound'];bounded_approximation=False
        command=nominal.copy();command[ids]+=self.scale[ids]*x
        requested_total=T.T@B@nominal+requested
        self.requested_world_wrench=requested_total.copy()
        self.requested_world_wrench[3:]+=np.cross(q[oi:oi+3],requested_total[:3])
        self.diagnostics=dict(allocation_residual=residual,relative_forcing=float((act@x-target)@cho_solve(cho_factor(mass,lower=True),act@x-target)),
            correction_scale=correction_scale,active_bounds=int(np.count_nonzero(np.minimum(x-lo,hi-x)<1e-7)),wall_s=time.perf_counter()-began,
            normalized_objective_gap_bound=gap,bounded_approximation=bounded_approximation,
            normal_tracking_error_m=normal_error)
        if path is not None:
            path.advance(path_acceleration)
            self.diagnostics['path']=dict(**path.info,acceleration_m_s2=path_acceleration,attainable_acceleration=interval)
        return command
