"""Experimental one-period allocation through all physical position drives.

Contact-free by default; a frozen generalized residual is diagnostic only. A reference is a motion objective; the final motor command
is the optimizer's actual held input, with no later bias or velocity addition.
"""
import numpy as np
import pinocchio as pin
import osqp
from scipy import sparse
from scipy.optimize import lsq_linear
import time
from .flight import FlightController
from .drive_prediction import DrivePrediction


def implicit_dock_velocity(mass,kp,kd,error,velocity,reference_velocity,reference_acceleration,period):
    """Discrete restoring target with the existing stiffness and damping.

    Error and reference derivatives use the interval's end time. Backward Euler avoids the
    unstable explicit update when damping/low inertia is fast relative to 10ms.
    This is a target filter, not a replacement for full-body prediction.
    """
    matrix=mass+np.diag(period*kd+period**2*kp)
    rhs=mass@velocity+period*(kp*error+kd*reference_velocity+mass@reference_acceleration)
    return np.linalg.solve(matrix,rhs)


class CoupledFlightController(FlightController):
    def __init__(self,model,meta,q0,period=.01,track_vectoring=False,base_residual_feedback=False,planned_input_reference=False):
        super().__init__(model,meta,q0,period)
        self.predictor=DrivePrediction(model,meta)
        self.input_scale=np.r_[self.speed*period,np.full(self.nr,5.)]
        self.track_joints=np.ones(len(self.actuators),dtype=bool) if track_vectoring else self.dock.copy()
        self.track_vi=self.vi[self.track_joints];self.track_qi=self.qi[self.track_joints]
        joint_scale=np.where(self.dock,.2,.3)[self.track_joints]
        self.track_scale=np.r_[np.tile(np.r_[np.full(3,.06),np.full(3,.15)],len(meta['bases'])),joint_scale]
        self.coupled_diagnostics=None
        self.prediction_samples=[]
        self.last_joint_bias=np.zeros(len(self.command));self.last_target_velocity=np.zeros(model.nv)
        self.base_residual_feedback=base_residual_feedback
        self.base_rows=6*len(meta['bases'])
        self.base_prediction_bias=np.zeros(self.base_rows)
        self.base_bias_alpha=1-np.exp(-period*np.tile([2.,2.,2.,4.,4.,4.],len(meta['bases'])))
        self.last_predicted_base=None
        self.planned_input_reference=planned_input_reference

    def tracking(self,q,v,info=None):
        values=[];rows=[]
        for base in self.meta['bases']:
            k,j=base['q'],base['v'];rotation=pin.Quaternion(q[k+3:k+7]).matrix()
            for start in [j,j+3]:
                values.extend(rotation@v[start:start+3])
                if info is not None:
                    rows.extend(rotation@(info['velocity_wrt_input'][start:start+3]-pin.skew(v[start:start+3])@info['position_wrt_input'][j+3:j+6]))
        values.extend(v[self.track_vi])
        if info is not None:rows.extend(info['velocity_wrt_input'][self.track_vi])
        return np.array(values),np.array(rows) if info is not None else None

    def update(self,q,v,target, *, target_velocity=None,target_acceleration=None,payload_mass=0.,payload_com=None,joint_bias=None,thrust_reference=None,gimbal_reference=None,generalized_load=None):
        if payload_mass:raise ValueError('coupled_controller_contact_model_not_validated')
        if generalized_load is not None and self.base_residual_feedback:
            raise ValueError('do not combine generalized and base prediction residuals')
        tv=np.zeros(self.model.nv) if target_velocity is None else np.asarray(target_velocity)
        ta=np.zeros(self.model.nv) if target_acceleration is None else np.asarray(target_acceleration)
        desired=[]
        for b in self.meta['bases']:
            k,j=b['q'],b['v'];R=pin.Quaternion(q[k+3:k+7]).matrix();Rd=pin.Quaternion(target[k+3:k+7]).matrix()
            linear,angular=R@v[j:j+3],R@v[j+3:j+6]
            desired.extend(linear+self.dt*(ta[j:j+3]+4*(target[k:k+3]-q[k:k+3])+4*(tv[j:j+3]-linear)))
            desired.extend(angular+self.dt*(ta[j+3:j+6]+16*pin.log3(Rd@R.T)+8*(tv[j+3:j+6]-angular)))
        joint_v,joint_q=self.track_vi,self.track_qi
        mass=pin.crba(self.model,self.data,q).copy()
        desired.extend(implicit_dock_velocity(mass[np.ix_(joint_v,joint_v)],self.kp[self.track_joints],self.kd[self.track_joints],target[joint_q]-q[joint_q],v[joint_v],tv[joint_v],ta[joint_v],self.dt));desired=np.array(desired)
        nominal_base_goal=desired[:self.base_rows].copy()
        base_innovation=np.zeros(self.base_rows)
        if self.base_residual_feedback:
            if self.last_predicted_base is not None:
                # Same world-axis velocity and same observation time. The
                # stored prediction is raw, before adding any correction.
                base_innovation=self.tracking(q,v)[0][:self.base_rows]-self.last_predicted_base
                self.base_prediction_bias+=self.base_bias_alpha*(base_innovation-self.base_prediction_bias)
            desired[:self.base_rows]-=self.base_prediction_bias
        previous=np.r_[self.command,self.force];size=len(self.command)
        lower=np.r_[np.maximum(self.limits[:,0],self.command-self.speed*self.dt),np.zeros(self.nr)]
        upper=np.r_[np.minimum(self.limits[:,1],self.command+self.speed*self.dt),[r['maximum'] for r in self.rotors]]
        margin=.01+np.array([self.actuators[i]['effort'] for i in self.ai])/self.kp[self.ai]
        lower[self.ai]=np.maximum(lower[self.ai],self.limits[self.ai,0]+margin)
        upper[self.ai]=np.minimum(upper[self.ai],self.limits[self.ai,1]-margin)
        inputs=previous.copy()
        if not self.control_count and joint_bias is not None and thrust_reference is not None:
            inputs=np.r_[target[self.qi]+(np.asarray(joint_bias)+self.kd*tv[self.vi])/self.kp,thrust_reference]
        inputs=np.clip(inputs,lower,upper)
        # Before the first issue, FlightController.force is an initialization
        # guess, not an applied-force history. Use the planned initial input as
        # the tie-breaking reference; actual command-rate bounds still use the
        # initial physical position targets above.
        regularization=inputs.copy() if not self.control_count else previous
        if self.planned_input_reference:
            if joint_bias is None or thrust_reference is None:
                raise ValueError('planned input reference requires matching bias and thrust')
            regularization=np.r_[target[self.qi]+(np.asarray(joint_bias)+self.kd*tv[self.vi])/self.kp,thrust_reference]
        epsilon=.001;iterations=[];initial_cost=None;accepted=None
        self.allocation_failure=None
        self.prediction_samples=[]
        def rollout(u,linearize):return self.predictor.interval(q,v,u[:size],u[size:],period=self.dt,linearize=linearize,history=True,generalized_load=generalized_load)
        def objective(pose,velocity,u):
            error=(self.tracking(pose,velocity)[0]-desired)/self.track_scale
            return float(error@error+epsilon**2*np.sum(((u-regularization)/self.input_scale)**2))
        def valid(info):
            positions=np.array([s['q'][self.qi] for s in info['samples']])
            velocities=np.array([s['v'][self.vi] for s in info['samples']])
            return bool(np.all(positions>=self.limits[:,0]-1e-7) and np.all(positions<=self.limits[:,1]+1e-7)
                        and np.all(abs(velocities)<=self.speed+1e-7))
        try:
            for iteration in range(2):
                pq,pv,info=rollout(inputs,True);value,jac=self.tracking(pq,pv,info)
                current_cost=objective(pq,pv,inputs)
                if initial_cost is None:initial_cost=current_cost
                if valid(info):accepted=(inputs.copy(),pq,pv,current_cost,info['samples'])
                A=jac*self.input_scale/self.track_scale[:,None];residual=(value-desired)/self.track_scale
                H=A.T@A+epsilon**2*np.eye(len(inputs));gradient=A.T@residual+epsilon**2*(inputs-regularization)/self.input_scale
                low=(lower-inputs)/self.input_scale;high=(upper-inputs)/self.input_scale
                constraints=[np.eye(len(inputs))];lows=[low];highs=[high]
                for sample in info['samples']:
                    for matrix,state,lo,hi,scale in [
                        (sample['position_wrt_input'][self.vi],sample['q'][self.qi],self.limits[:,0],self.limits[:,1],.01),
                        (sample['velocity_wrt_input'][self.vi],sample['v'][self.vi],-self.speed,self.speed,3.)]:
                        matrix=matrix*self.input_scale
                        smallest=np.maximum(matrix,0)@low+np.minimum(matrix,0)@high
                        largest=np.maximum(matrix,0)@high+np.minimum(matrix,0)@low
                        needed=(state+smallest<lo)|(state+largest>hi)
                        if np.any(needed):constraints.append(matrix[needed]/scale);lows.append((lo[needed]-state[needed])/scale);highs.append((hi[needed]-state[needed])/scale)
                matrix=np.vstack(constraints);clow=np.concatenate(lows);chigh=np.concatenate(highs)
                # Exact coordinate change, not a change in objective weights.
                # A rank-deficient motion map plus a small tie-breaker otherwise
                # leaves the numerical Hessian very poorly conditioned.
                self.problem_evidence=dict(H=H,gradient=gradient,matrix=matrix,lower=clow,upper=chigh,inputs=inputs.copy(),input_scale=self.input_scale,q=q.copy(),v=v.copy(),target=target.copy())
                if len(constraints)==1:
                    # With only input bounds, solve the original least-squares
                    # system directly. Normal equations square its condition
                    # number; a generic inequality solver is unnecessary here.
                    start=time.perf_counter()
                    augmented=np.vstack([A,epsilon*np.eye(len(inputs))])
                    rhs=-np.r_[residual,epsilon*(inputs-regularization)/self.input_scale]
                    solution=lsq_linear(augmented,rhs,bounds=(low,high),method='bvls',tol=1e-10,max_iter=100)
                    iterations.append(dict(status=solution.message,iterations=solution.nit,solver_s=time.perf_counter()-start,optimality=solution.optimality))
                    if not solution.success:raise ValueError('coupled_allocation_qp_not_solved')
                    change=solution.x
                else:
                    transform=np.linalg.solve(np.linalg.cholesky(H).T,np.eye(len(inputs)))
                    solver=osqp.OSQP();solver.setup(P=sparse.eye(len(inputs),format='csc'),q=transform.T@gradient,A=sparse.csc_matrix(matrix@transform),l=clow,u=chigh,verbose=False,eps_abs=1e-8,eps_rel=1e-8,max_iter=2000,polishing=False)
                    solution=solver.solve(raise_error=False)
                    iterations.append(dict(status=solution.info.status,iterations=solution.info.iter,solver_s=solution.info.run_time,primal_residual=solution.info.prim_res,dual_residual=solution.info.dual_res))
                    if solution.info.status_val!=1:raise ValueError('coupled_allocation_qp_not_solved')
                    change=transform@solution.x
                proposed=inputs+self.input_scale*change
                # The QP tolerance is in normalized coordinates. Project its
                # small bound residual before the nonlinear physical check,
                # so no negative thrust or excessive command is ever issued.
                bounded=np.clip(proposed,lower,upper)
                iterations[-1]['input_projection_max']=float(abs(bounded-proposed).max())
                direction=(bounded-inputs)/self.input_scale;slope=2*float(gradient@direction)
                alpha=1.;trials=[];improved=False
                # Saturation changes the derivative across a drive's torque
                # boundary. A full linearized step need not reduce the actual
                # objective. Backtrack against the nonlinear bounded model,
                # retaining only independently rechecked feasible inputs.
                for trial in range(4):
                    candidate=inputs+alpha*self.input_scale*direction
                    if np.any(candidate<lower-1e-7) or np.any(candidate>upper+1e-7):raise ValueError('coupled_allocation_input_bound')
                    pq,pv,checked=rollout(candidate,False);cost=objective(pq,pv,candidate)
                    feasible=valid(checked)
                    trials.append(dict(step_fraction=alpha,nonlinear_cost=cost,predicted_bounds_valid=feasible,peak_predicted_speed=max(float(abs(s['v'][self.vi]).max()) for s in checked['samples'])))
                    if feasible and cost<=current_cost+1e-4*alpha*min(slope,0.)+1e-12:
                        inputs=candidate;accepted=(inputs.copy(),pq,pv,cost,checked['samples']);improved=True;break
                    denominator=2*(cost-current_cost-alpha*slope)
                    estimate=-slope*alpha**2/denominator if denominator>0 and slope<0 else alpha/2
                    alpha=float(np.clip(estimate,alpha*.001,alpha*.5))
                iterations[-1].update(initial_cost=initial_cost,trials=trials,improved=improved)
                if not improved or alpha==1.:break
            if accepted is None:raise ValueError('coupled_allocation_nonlinear_recheck_failed')
            inputs,pq,pv,cost,self.prediction_samples=accepted
        except ValueError as error:
            self.allocation_failure=str(error);self.coupled_diagnostics=dict(accepted=False,reason=str(error),iterations=iterations,desired_tracking_velocity=desired.tolist())
            self.command_diagnostics=dict(observed_configuration=q.tolist(),observed_velocity=v.tolist(),issued_command=self.command.tolist(),issued_thrust=self.force.tolist(),target=target.tolist(),coupled=self.coupled_diagnostics)
            return self.command.copy(),self.force.copy()
        self.command=inputs[:size].copy();self.force=inputs[size:].copy();self.control_count+=1
        self.last_joint_bias=self.kp*(self.command-target[self.qi])-self.kd*tv[self.vi]
        self.last_target_velocity=tv.copy()
        self.last_predicted_base=self.tracking(pq,pv)[0][:self.base_rows].copy()
        self.coupled_diagnostics=dict(accepted=True,iterations=iterations,cost=cost,initial_cost=initial_cost,desired_tracking_velocity=desired.tolist(),predicted_tracking_velocity=self.tracking(pq,pv)[0].tolist(),predicted_q=pq.tolist(),predicted_v=pv.tolist())
        self.coupled_diagnostics.update(planned_input_reference=self.planned_input_reference,
            tracking_cost=float(np.sum(((self.tracking(pq,pv)[0]-desired)/self.track_scale)**2)),
            regularization_cost=float(epsilon**2*np.sum(((inputs-regularization)/self.input_scale)**2)))
        if self.base_residual_feedback:
            self.coupled_diagnostics['base_residual_feedback']=dict(
                innovation=base_innovation.tolist(),estimate=self.base_prediction_bias.tolist(),
                nominal_velocity_goal=nominal_base_goal.tolist(),corrected_velocity_goal=desired[:self.base_rows].tolist())
        self.command_diagnostics=dict(observed_configuration=q.tolist(),observed_velocity=v.tolist(),issued_command=self.command.tolist(),issued_thrust=self.force.tolist(),target=target.tolist(),coupled=self.coupled_diagnostics)
        return self.command.copy(),self.force.copy()
