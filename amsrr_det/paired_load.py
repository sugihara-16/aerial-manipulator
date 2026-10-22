"""Diagnostic transfer of observed motor load into rotor action.

The correction is fixed-capture or saturation-only feedback before transfer.
It is not an external-contact-force estimator.
The geometric reference generator stays active. Failure returns no new input.
"""
import numpy as np
from scipy.linalg import null_space
from scipy.optimize import minimize,LinearConstraint
from .rotor_distribution import rotor_actions,redistribute_observed_load


def progress(t):
    x=float(np.clip(t,0.,1.));return x*x*x*(10+x*(-15+6*x))


def unwind_saturated_request(command,previous,q,v,motor,kp,kd,effort,limits,step,dt,extrema=None):
    """Largest bounded correction only on observed, same-sign saturation.

    The observed load is not a prediction of the next implicit drive effort.
    Unsaturated axes retain their existing offset and restoring response.
    """
    if not np.isfinite(motor).all() or np.any(abs(motor)>effort+1e-4):
        return None,dict(reason='motor_observation_outside_qualified_range')
    raw=kp*(command-q)-kd*v;old=kp*(previous-q)-kd*v
    history_saturated=abs(motor)>=effort-1e-4
    if extrema is not None:
        minimum,maximum=extrema
        history_saturated=np.where(raw<0,minimum<=-effort+1e-4,maximum>=effort-1e-4)
    active=(abs(raw)>effort+1e-4)&history_saturated&(raw*motor>0)&(abs(motor)<abs(raw))
    lower=np.maximum.reduce([limits[:,0],previous-step,q+(np.minimum(-effort,old)+kd*v)/kp])
    upper=np.minimum.reduce([limits[:,1],previous+step,q+(np.maximum(effort,old)+kd*v)/kp])
    result=command.copy();fractions=np.zeros(len(command))
    for j in np.flatnonzero(active):
        change=(motor[j]-raw[j])/kp[j]
        ends=sorted([(lower[j]-command[j])/change,(upper[j]-command[j])/change])
        low=max(0.,ends[0]);high=min(1.,ends[1],.1*dt/abs(change))
        if high<low-1e-10:return None,dict(reason='dynamic_unwind_bounds_incompatible')
        fractions[j]=max(0.,high);result[j]+=fractions[j]*change
    valid=np.all(result>=lower-1e-8) and np.all(result<=upper+1e-8)
    return (result if valid else None),dict(active=active.tolist(),fractions=fractions.tolist(),
        raw_before_nm=raw.tolist(),raw_after_nm=(kp*(result-q)-kd*v).tolist())


def paired_progress_step(command,previous,q,v,kp,kd,effort,limits,command_step,
                         force,direction,joint_action,maximum,last_delta,s,rate,dt,duration):
    """Largest feasible scalar step, bounded by the previous quintic's peaks.

    This screens issued requests, not the next physical motor response.
    """
    vmax=1.875/duration;amax=(10*np.sqrt(3)/3)/duration**2
    discriminant=(amax*dt)**2-4*amax*dt*rate+8*amax*(1-s)
    if discriminant<0:return None,dict(reason='progress_cannot_brake')
    stop_rate=max(0.,(-amax*dt+np.sqrt(discriminant))/2)
    # Braking can finish before the end of the control interval.
    lo=s+(rate**2/(2*amax) if rate<amax*dt else dt*rate-.5*amax*dt**2)
    if rate>0 and rate**2/(2*amax)-1e-12<=1-s<=.5*dt*rate+1e-12:
        hi=1.
    else:hi=min(1.,s+.5*dt*(rate+min(vmax,rate+amax*dt,stop_rate)))
    raw=kp*(command-q)-kd*v;action=joint_action@direction
    force_step=1.875*dt/duration*maximum
    groups=[(command,-action/kp,np.maximum(limits[:,0],previous-command_step),np.minimum(limits[:,1],previous+command_step)),
            (raw,-action,-effort-1e-4,effort+1e-4),
            (force,direction,np.zeros_like(force),maximum),
            (np.zeros_like(force),direction,last_delta-force_step,last_delta+force_step)]
    for offset,slope,lower,upper in groups:
        moving=abs(slope)>1e-12
        if np.any(offset[~moving]<lower[~moving]-1e-8) or np.any(offset[~moving]>upper[~moving]+1e-8):
            return None,dict(reason='paired_constant_input_out_of_bounds')
        x=(lower[moving]-offset[moving])/slope[moving];y=(upper[moving]-offset[moving])/slope[moving]
        lo=max(lo,np.minimum(x,y).max(initial=-np.inf));hi=min(hi,np.maximum(x,y).min(initial=np.inf))
    valid=hi>=lo-1e-12 and hi>=s-1e-12
    if valid and abs(hi-1.)<1e-12:hi=1.
    return (float(hi) if valid else None),dict(lower=float(lo),upper=float(hi),maximum_rate=vmax,maximum_acceleration=amax)


class PairedLoadTransfer:
    def __init__(self,flight,dynamic_unwind=False,atomic_start=False):
        self.flight=flight;self.dock=flight.dock;self.kp=flight.kp[self.dock]
        self.limit=np.array([a['effort'] for a in flight.actuators])[self.dock]
        self.maximum=np.array([r['maximum'] for r in flight.rotors])
        self.offset=np.zeros(len(flight.actuators));self.started=None;self.failure=None
        self.diagnostics={};self.last_delta=np.zeros(flight.nr)
        self.command_step=.2*flight.dt+(flight.kd[self.dock]/self.kp)*.5*flight.dt
        self.transfer_time=2.
        self.dynamic_unwind=dynamic_unwind;self.ready_since=None;self.transfer_started=None
        self.transfer_deadline=None
        if atomic_start and not dynamic_unwind:raise ValueError('atomic start requires dynamic unwind')
        self.atomic_start=atomic_start;self.progress=0.;self.progress_rate=0.;self.start_allowed=True

    def action(self,q,command):
        f=self.flight;B=rotor_actions(f.model,f.meta,q);predicted=q.copy()
        predicted[f.qi[f.ai]]+=(1-5*(1-np.exp(-.2)))*(command[f.ai]-q[f.qi[f.ai]])
        P=rotor_actions(f.model,f.meta,predicted)
        # gimbal commands stay unmodified, so preserve their action too.
        return np.vstack([B[:6],P[:6],B[f.vi[f.ai]],P[f.vi[f.ai]]]),B[f.vi[self.dock]],P[f.vi[self.dock]]

    def capture(self,t,q,v,command,force,motor):
        if self.started is not None:raise ValueError('load correction must not be recaptured')
        f=self.flight;preserved,joint,_=self.action(q,command)
        target,screen=redistribute_observed_load(preserved,joint,np.asarray(motor)[self.dock],force,self.maximum,self.limit)
        self.diagnostics=dict(capture_time=t,capture_screen=screen)
        if target is None or screen['minimum_peak_ratio']>=screen['observed_peak_ratio']-1e-5:
            self.failure='observed_load_relief_not_found';return
        raw=self.kp*(command[self.dock]-q[f.qi[self.dock]])-f.kd[self.dock]*v[f.vi[self.dock]]
        self.a=np.asarray(motor)[self.dock]-raw
        self.goal=target-force;self.motor0=np.asarray(motor)[self.dock].copy();self.started=t
        self.wind_time=max(.2,1.875*float(max(abs(self.a/self.kp)))/.1)
        self.diagnostics.update(captured_offset_nm=self.a.tolist(),captured_motor_nm=self.motor0.tolist(),
            captured_raw_nm=raw.tolist(),wind_time_s=self.wind_time,transfer_time_s=self.transfer_time,
            captured_thrust_change_n=self.goal.tolist())
        if self.dynamic_unwind:
            self.a=np.zeros_like(self.a)
            self.diagnostics.update(dynamic_unwind=True)

    def apply(self,t,q,v,command,force,previous_command,motor,extrema=None):
        if self.started is None or self.failure:return None if self.failure else (command,force)
        if self.atomic_start:return self.apply_atomic(t,q,v,command,force,previous_command,motor,extrema)
        f=self.flight;elapsed=t-self.started;wind=progress(elapsed/self.wind_time)
        amount=progress((elapsed-self.wind_time-.3)/self.transfer_time)
        # command already contains last cycle's offset, as does the reference
        # generator's drive guard. Remove it only in this algebraic proposal.
        baseline_command=command.copy();baseline_command[self.dock]-=self.offset[self.dock]/self.kp
        base_raw=self.kp*(baseline_command[self.dock]-q[f.qi[self.dock]])-f.kd[self.dock]*v[f.vi[self.dock]]
        if self.dynamic_unwind:
            wind=1.
            if self.transfer_started is None:
                prior=baseline_command[self.dock]+self.a/self.kp
                revised,info=unwind_saturated_request(prior,previous_command[self.dock],
                    q[f.qi[self.dock]],v[f.vi[self.dock]],np.asarray(motor)[self.dock],
                    self.kp,f.kd[self.dock],self.limit,f.limits[self.dock],self.command_step,f.dt)
                self.diagnostics['unwind']=info
                if revised is None:self.failure=info.get('reason','dynamic_unwind_bounds_incompatible');return None
                self.a+=self.kp*(revised-prior)
                if np.all(abs(base_raw+self.a)<=self.limit+1e-4):
                    if self.ready_since is None:self.ready_since=t
                    if t-self.ready_since>=.3-1e-9:self.transfer_started=t
                else:self.ready_since=None
            amount=0. if self.transfer_started is None else progress((t-self.transfer_started)/self.transfer_time)
            self.diagnostics.update(unwind_offset_nm=self.a.tolist(),transfer_start_s=self.transfer_started,
                ready_since_s=self.ready_since)
            if self.transfer_deadline is not None and t>=self.transfer_deadline and amount<1.:
                self.failure='paired_transfer_missed_matched_hold';return None
        unwound=base_raw+self.a
        if not self.dynamic_unwind and wind==1. and np.any(abs(unwound)>self.limit+1e-4):
            self.failure='captured_correction_did_not_clear_saturation'
            self.diagnostics.update(failure=self.failure,t=t,unwound_request_nm=unwound.tolist(),measured_motor_nm=np.asarray(motor)[self.dock].tolist())
            return None
        delta=np.zeros(f.nr);preserved,joint,future_joint=self.action(q,command)
        if amount:
            basis=null_space(preserved,rcond=1e-9)
            if not basis.shape[1]:self.failure='paired_load_nullspace_lost';return None
            desired=amount*self.goal;initial=basis.T@desired
            solved=minimize(lambda z:.5*np.sum((basis@z-desired)**2),initial,
                jac=lambda z:basis.T@(basis@z-desired),constraints=[LinearConstraint(basis,-force,self.maximum-force)],
                method='SLSQP',options=dict(maxiter=40,ftol=1e-12))
            if not solved.success:self.failure='paired_load_projection_not_found';return None
            delta=basis@solved.x
        new_offset=wind*self.a-joint@delta
        proposed=baseline_command.copy();proposed[self.dock]+=new_offset/self.kp
        issued_force=force+delta
        raw=self.kp*(proposed[self.dock]-q[f.qi[self.dock]])-f.kd[self.dock]*v[f.vi[self.dock]]
        old_raw=self.kp*(previous_command[self.dock]-q[f.qi[self.dock]])-f.kd[self.dock]*v[f.vi[self.dock]]
        step=np.max(abs(proposed[self.dock]-previous_command[self.dock])/self.command_step)
        force_step=1.875*f.dt/self.transfer_time*self.maximum
        valid=(np.all(issued_force>=-1e-8) and np.all(issued_force<=self.maximum+1e-8)
            and np.all(proposed>=f.limits[:,0]) and np.all(proposed<=f.limits[:,1])
            and step<=1.+1e-8 and np.all(abs(delta-self.last_delta)<=force_step+1e-8)
            and np.max(abs(preserved@delta),initial=0.)<1e-7
            and np.all(raw>=np.minimum(-self.limit,old_raw)-1e-8)
            and np.all(raw<=np.maximum(self.limit,old_raw)+1e-8))
        self.diagnostics.update(t=t,wind_fraction=wind,transfer_fraction=amount,
            raw_request_nm=raw.tolist(),measured_motor_nm=np.asarray(motor)[self.dock].tolist(),
            request_without_paired_transfer_nm=unwound.tolist(),
            command_step_ratio=float(step),thrust_change_n=delta.tolist(),
            paired_request_change_nm=(-joint@delta).tolist(),
            predicted_angle_joint_action_difference_nm=((future_joint-joint)@delta).tolist(),
            paired_input_valid=bool(valid))
        if self.dynamic_unwind and self.transfer_started is not None and np.any(abs(raw)>self.limit+1e-4):
            self.failure='paired_transfer_final_request_saturated';return None
        if not valid:self.failure='paired_load_final_input_invalid';return None
        self.offset[self.dock]=new_offset;self.last_delta=delta.copy()
        return proposed,issued_force

    def apply_atomic(self,t,q,v,command,force,previous,motor,extrema):
        f=self.flight;baseline=command.copy();baseline[self.dock]-=self.offset[self.dock]/self.kp
        corrected=baseline[self.dock]+self.a/self.kp
        if self.transfer_started is None:
            observed=None if extrema is None else tuple(np.asarray(x)[self.dock] for x in extrema)
            revised,unwind=unwind_saturated_request(corrected,previous[self.dock],q[f.qi[self.dock]],v[f.vi[self.dock]],
                np.asarray(motor)[self.dock],self.kp,f.kd[self.dock],self.limit,f.limits[self.dock],self.command_step,f.dt,observed)
            self.diagnostics['unwind']=unwind
            if revised is None:self.failure=unwind.get('reason','dynamic_unwind_bounds_incompatible');return None
            self.a+=self.kp*(revised-corrected);corrected=revised
        E,J,P=self.action(q,command);basis=null_space(E,rcond=1e-9)
        if not basis.shape[1]:self.failure='paired_load_nullspace_lost';return None
        solved=minimize(lambda z:.5*np.sum((basis@z-self.goal)**2),basis.T@self.goal,
            jac=lambda z:basis.T@(basis@z-self.goal),constraints=[LinearConstraint(basis,-force,self.maximum-force)],
            method='SLSQP',options=dict(maxiter=40,ftol=1e-12))
        if not solved.success:self.failure='paired_load_projection_not_found';return None
        direction=basis@solved.x
        next_s,screen=paired_progress_step(corrected,previous[self.dock],q[f.qi[self.dock]],v[f.vi[self.dock]],
            self.kp,f.kd[self.dock],self.limit,f.limits[self.dock],self.command_step,
            force,direction,J,self.maximum,self.last_delta,self.progress,self.progress_rate,f.dt,self.transfer_time)
        self.diagnostics.update(t=t,atomic_start=True,entry_screen=screen,unwind_offset_nm=self.a.tolist())
        if self.transfer_started is None and (next_s is None or next_s<=1e-12 or not self.start_allowed):
            next_s=0.
        elif next_s is None:
            self.failure='paired_progress_not_feasible';return None
        elif self.transfer_started is None:
            self.transfer_started=t
        if self.transfer_deadline is not None and t>=self.transfer_deadline and next_s<1.-1e-10:
            self.failure='paired_transfer_missed_matched_hold';return None
        delta=next_s*direction;offset=self.a-J@delta
        proposed=baseline.copy();proposed[self.dock]+=offset/self.kp;issued=force+delta
        raw=self.kp*(proposed[self.dock]-q[f.qi[self.dock]])-f.kd[self.dock]*v[f.vi[self.dock]]
        step=float(np.max(abs(proposed[self.dock]-previous[self.dock])/self.command_step))
        valid=(step<=1+1e-8 and np.all(proposed>=f.limits[:,0]-1e-8) and np.all(proposed<=f.limits[:,1]+1e-8)
            and np.all(issued>=-1e-8) and np.all(issued<=self.maximum+1e-8) and abs(E@delta).max(initial=0.)<1e-7)
        if self.transfer_started is not None:valid=valid and np.all(abs(raw)<=self.limit+1e-4+1e-8)
        self.diagnostics.update(transfer_start_s=self.transfer_started,transfer_fraction=next_s,
            raw_request_nm=raw.tolist(),measured_motor_nm=np.asarray(motor)[self.dock].tolist(),
            request_without_paired_transfer_nm=(self.kp*(corrected-q[f.qi[self.dock]])-f.kd[self.dock]*v[f.vi[self.dock]]).tolist(),
            command_step_ratio=step,thrust_change_n=delta.tolist(),paired_request_change_nm=(-J@delta).tolist(),
            predicted_angle_joint_action_difference_nm=((P-J)@delta).tolist(),paired_input_valid=bool(valid))
        if not valid:self.failure='paired_load_final_input_invalid';return None
        self.progress_rate=max(0.,2*(next_s-self.progress)/f.dt-self.progress_rate)
        self.progress=next_s;self.offset[self.dock]=offset;self.last_delta=delta.copy()
        return proposed,issued
