"""Common rotor directions for two load endpoints at fixed contact geometry.

The two input vectors are independent, while the direction angles are shared.
Exact physical force screens accept the result, never the least-squares score.
Convex combinations cover intermediate loads only at this fixed geometry.
"""
import numpy as np
import pinocchio as pin
from scipy.optimize import least_squares
from .rotor_distribution import rotor_action_basis,rotor_basis_actions
from .equilibrium import point_jacobian,contact_basis,static_balance as _static_balance
from .free_flight_trim import free_flight_trim
from .grasp_task import material_force_contacts


def static_balance(*args,**kwargs):
    """Quadratic proposal; callers compare actual sum(T**1.5) afterwards."""
    kwargs["burden_power"]=2.
    return _static_balance(*args,**kwargs)


def load_span_trim(p,spec,q,surfaces,lower=.9,upper=1.1,*,free_lower=False):
    from .planning_memo import evaluate
    return evaluate(dict(operation='valve_load_span',specification=spec.__dict__,
        modules=p.meta['modules'],edges=p.meta['edges'],q=np.asarray(q).tolist(),
        surfaces=surfaces,lower=lower,upper=upper,free_lower=free_lower),
        lambda:_load_span_trim(p,spec,q,surfaces,lower,upper,free_lower))


def endpoint_load_envelope(p,spec,q,surfaces):
    return load_span_trim(p,spec,q,surfaces,0.,1.1,free_lower=True)


def _load_span_trim(p,spec,q,surfaces,lower,upper,free_lower):
    from .valve_plan import bearing_support
    model,meta=p.pin,p.meta;data=model.createData()
    contacts=material_force_contacts(p,spec.shape,q,surfaces,force_limit=None,material_tangents=True)
    sign=np.sign(spec.target_angle)
    def screen(pose,amount,free):
        return static_balance(model,meta,pose,() if free else contacts,payload_fraction=0.,
            object_wrench=None if free else np.r_[np.zeros(3),np.asarray(spec.axis)*sign*spec.torque*amount],
            object_support=None if free else bearing_support(spec),minimize_burden=True,fast_lp=True,direct_burden=True)
    def check(pose):
        reports=[screen(pose,lower,free_lower),screen(pose,upper,False)]
        return dict(free=reports[0]['status'],load=reports[1]['status'],trims=reports,
                    lower=lower,upper=upper,free_lower=free_lower)
    first=check(q)
    if first['free']==first['load']=='static_screen_pass':return q.copy(),first
    if free_lower:
        from .planning_memo import evaluate as reuse
        q0,initial=reuse(dict(operation='valve_free_flight_seed',modules=meta['modules'],edges=meta['edges'],
            payload=spec.payload,q=np.asarray(q).tolist()),lambda:free_flight_trim(model,meta,q,physical_check_interval=8))
    else:q0,initial=q.copy(),first['trims'][1]
    rotors=meta['rotors'];nr=len(rotors);act=meta['actuators']
    gimbals=[act[r['gimbal']] for r in rotors];qi=np.array([a['q'] for a in gimbals])
    av=np.array([a['v'] for a in act]);limits=np.array([a['effort'] for a in act]);nc=len(contacts)
    pin.framesForwardKinematics(model,data,q0);reaction=[];friction=[]
    for contact in contacts:
        R=contact_basis(contact)
        J=point_jacobian(model,data,q0,meta['frames'][contact['frame']],contact['point'])
        J-=point_jacobian(model,data,q0,meta['frames'][contact['target_frame']],contact['target_point'])
        reaction.append(J.T@R);friction.append(contact['mu'])
    C=np.column_stack(reaction);ov=meta['object']['v'];loaded_eq=np.r_[np.arange(6),ov+5]
    thrust=np.asarray(initial.get('thrust',np.ones(nr)*meta['mass']*9.80665/nr))
    x0=list(q0[qi]);lo=[a['lower']+.02 for a in gimbals];hi=[a['upper']-.02 for a in gimbals];stages=[]
    for amount,free in [(lower,free_lower),(upper,False)]:
        begin=len(x0);count=0 if free else nc
        x0.extend(thrust);lo.extend(r['minimum'] for r in rotors);hi.extend(r['maximum'] for r in rotors)
        x0.extend(np.tile([spec.torque*amount/(.4*max(nc,1)),0.,0.],count))
        lo.extend(np.tile([0.,-100.,-100.],count));hi.extend(np.tile([100.,100.,100.],count))
        stages.append((amount,free,begin,count))
    x0=np.asarray(x0);lo=np.asarray(lo);hi=np.asarray(hi);cached=None;pair=None
    rotor_basis=rotor_action_basis(model,meta,q0)
    evaluations=0
    class Feasible(Exception):
        def __init__(self,pose,report):self.pose=pose;self.report=report
    def evaluate(x):
        nonlocal cached,pair,evaluations
        if cached is not None and np.array_equal(cached,x):return pair
        pose=q0.copy();pose[qi]=x[:nr];B=rotor_basis_actions(rotor_basis,x[:nr])
        gravity=pin.computeGeneralizedGravity(model,data,pose).copy();gravity[ov:ov+6]=0.
        dg=pin.computeGeneralizedGravityDerivatives(model,data,pose).copy();dg[ov:ov+6]=0.
        derivatives=(-rotor_basis[1]*np.sin(x[:nr])+rotor_basis[2]*np.cos(x[:nr])).T
        residual=[];jacobian=[]
        for amount,free,begin,count in stages:
            force=x[begin:begin+nr];tau=gravity-B@force
            J=np.zeros((model.nv,len(x)));J[:,begin:begin+nr]=-B
            for i,a in enumerate(gimbals):J[:,i]=dg[:,a['v']]-derivatives[i]*force[i]
            if count:
                f=x[begin+nr:begin+nr+3*count].reshape(-1,3)
                tau[ov+5]+=sign*spec.torque*amount;tau-=C@f.ravel();J[:,begin+nr:begin+nr+3*count]=-C
            eq=np.arange(6) if free else loaded_eq
            residual.extend([tau[eq],np.maximum(abs(tau[av])/limits-.85,0.)])
            jacobian.extend([J[eq],J[av]*(np.sign(tau[av])*(abs(tau[av])/limits>.85)/limits)[:,None]])
            if count:
                cone=np.maximum(abs(f[:,1])+abs(f[:,2])-np.asarray(friction)*f[:,0],0.)
                cap=np.maximum(abs(f).sum(axis=1)-np.asarray([np.inf if c['max_force'] is None else c['max_force'] for c in contacts]),0.)
                Jc=np.zeros((nc,len(x)));Jcap=Jc.copy()
                for i in range(nc):
                    block=slice(begin+nr+3*i,begin+nr+3*i+3)
                    Jc[i,block]=[-friction[i],np.sign(f[i,1]),np.sign(f[i,2])];Jcap[i,block]=np.sign(f[i])
                residual.extend([cone,cap]);jacobian.extend([Jc*(cone>0)[:,None],Jcap*(cap>0)[:,None]])
        residual.append(1e-7*(x-x0));jacobian.append(1e-7*np.eye(len(x)))
        cached=x.copy();pair=np.concatenate(residual),np.vstack(jacobian);evaluations+=1
        # Least squares only proposes angles. Stop as soon as the exact
        # physical force problems accept them; tiny soft residuals need not
        # consume the remaining nonlinear iterations.
        # The soft .85 joint margin can remain nonzero even when the actual
        # bounded force problem is feasible. Do not make that soft score a
        # prerequisite for testing physical acceptance.
        if evaluations%8==0:
            accepted=check(pose)
            if accepted['free']==accepted['load']=='static_screen_pass':
                accepted.update(evaluations=evaluations,early_physical_acceptance=True)
                raise Feasible(pose.copy(),accepted)
        return pair
    try:
        solved=least_squares(lambda x:evaluate(x)[0],np.clip(x0,lo,hi),jac=lambda x:evaluate(x)[1],
            bounds=(lo,hi),tr_solver='lsmr',max_nfev=60,ftol=1e-10,gtol=1e-10,xtol=1e-10)
    except Feasible as accepted:
        return accepted.pose,accepted.report
    q0[qi]=solved.x[:nr];report=check(q0)
    report.update(evaluations=solved.nfev,residual=float(np.linalg.norm(solved.fun)))
    return q0,report
