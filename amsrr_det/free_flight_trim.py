"""Fixed Dock/base posture; only rotor direction and force are fitted.

This local numerical search is a necessary stationary screen, not flight proof.
Every independent component of a forest has its own six balance equations.
"""
import numpy as np
import pinocchio as pin
from scipy.optimize import least_squares
from .equilibrium import static_balance
from .rotor_distribution import rotor_actions


def _free_flight_objective(model,meta,q,x0):
    """Exact inverse-dynamics residual and its angle/force derivatives."""
    data=model.createData();rotors=meta['rotors'];nr=len(rotors)
    act=np.array([a['v'] for a in meta['actuators']]);limits=np.array([a['effort'] for a in meta['actuators']])
    floating=np.array([b['v']+i for b in meta['bases'] for i in range(6)])
    gimbal=[meta['actuators'][r['gimbal']] for r in rotors]
    qi=[a['q'] for a in gimbal];vi=[a['v'] for a in gimbal]
    zero=np.zeros(model.nv);cached=None;pair=None
    def evaluate(x):
        nonlocal cached,pair
        if cached is not None and np.array_equal(x,cached):return pair
        pose=q.copy();pose[qi]=x[nr:];ext=[pin.Force.Zero() for _ in range(model.njoints)]
        for r,f in zip(rotors,x[:nr]):
            frame=model.frames[r['frame']]
            ext[frame.parentJoint]+=frame.placement.act(pin.Force(np.array([0.,0.,f]),np.zeros(3)))
        tau=pin.rnea(model,data,pose,zero,zero,ext).copy()
        angle_derivative=pin.computeRNEADerivatives(model,data,pose,zero,zero,ext)[0][:,vi].copy()
        J=np.column_stack([-rotor_actions(model,meta,pose),angle_derivative])
        excess=np.maximum(abs(tau[act])/limits-.85,0.)
        residual=np.r_[tau[floating],excess,1e-6*(x-x0)]
        jac=np.vstack([J[floating],J[act]*(np.sign(tau[act])*(excess>0)/limits)[:,None],1e-6*np.eye(len(x))])
        cached=x.copy();pair=residual,jac
        return pair
    return evaluate


def check_fixed_thrust(model,meta,q,forces):
    """Re-express one physical rotor assignment in any connected forest."""
    d=model.createData();ext=[pin.Force.Zero() for _ in range(model.njoints)]
    for r,f in zip(meta['rotors'],forces):
        frame=model.frames[r['frame']]
        ext[frame.parentJoint]+=frame.placement.act(pin.Force(np.array([0.,0.,f]),np.zeros(3)))
    torque=pin.rnea(model,d,q,np.zeros(model.nv),np.zeros(model.nv),ext)
    floating=np.array([b['v']+i for b in meta['bases'] for i in range(6)])
    joints=[dict(name=a['name'],effort_nm=float(torque[a['v']]),limit_nm=a['effort'],normalized=abs(float(torque[a['v']]))/a['effort']) for a in meta['actuators']]
    residual=float(abs(torque[floating]).max(initial=0.));effort=max(j['normalized'] for j in joints)
    refs=[q[a['q']]+j['effort_nm']/a['kp'] for a,j in zip(meta['actuators'],joints)]
    refs_valid=all(a['lower']<=ref<=a['upper'] for a,ref in zip(meta['actuators'],refs))
    return dict(status='static_screen_pass' if residual<1e-7 and effort<=1.+1e-7 and refs_valid else 'static_screen_violation',
                scope='shared_independent_module_trim',equilibrium_residual=residual,max_normalized_joint_effort=effort,
                pd_reference_in_range=refs_valid,joints=joints,thrust=np.asarray(forces).tolist())


def free_flight_trim(model,meta,q, *, maximum_evaluations=60,physical_check_interval=0):
    initial=static_balance(model,meta,q,payload_fraction=0.,balance_thrust=True)
    if initial['status']=='static_screen_pass':return q.copy(),initial
    rotors=meta['rotors'];nr=len(rotors)
    gimbal=[meta['actuators'][r['gimbal']] for r in rotors];indices=np.array([a['q'] for a in gimbal])
    low=np.r_[np.zeros(nr),[a['lower']+.03 for a in gimbal]]
    high=np.r_[[r['maximum'] for r in rotors],[a['upper']-.03 for a in gimbal]]
    force=(meta['mass']-(meta['object'] or {}).get('mass',0))*9.80665/nr
    x0=np.r_[np.full(nr,force),q[indices]]
    objective=_free_flight_objective(model,meta,q,x0)
    evaluations=0
    class Feasible(Exception):
        def __init__(self,pose,report):self.pose=pose;self.report=report
    def residual(x):
        nonlocal evaluations
        value=objective(x)[0];evaluations+=1
        if physical_check_interval and evaluations%physical_check_interval==0:
            pose=q.copy();pose[indices]=x[nr:]
            checked=static_balance(model,meta,pose,payload_fraction=0.,balance_thrust=True)
            if checked['status']=='static_screen_pass':
                checked['direction_fit_evaluations']=evaluations
                raise Feasible(pose,checked)
        return value
    try:
        result=least_squares(residual,np.clip(x0,low,high),jac=lambda x:objective(x)[1],bounds=(low,high),max_nfev=maximum_evaluations,
                             tr_solver='lsmr' if physical_check_interval else 'exact',
                             ftol=1e-10,xtol=1e-10,gtol=1e-10)
    except Feasible as accepted:return accepted.pose,accepted.report
    pose=q.copy();pose[indices]=result.x[nr:]
    checked=static_balance(model,meta,pose,payload_fraction=0.,balance_thrust=True)
    checked['direction_fit_evaluations']=result.nfev
    return pose,checked


def support_schedule(model,meta,first,last,start_trim,end_trim, *, tolerance=.02,maximum_depth=7):
    """Sample and refine force/bias interpolation along a fixed pose path.

    Endpoints with feasible loads do not imply feasible interpolated loads.
    Check quarter/midpoints, refining only the force schedule; the geometric
    path and its duration do not change. This is still a sampled static screen.
    """
    nodes={0.:start_trim,1.:end_trim}
    limits=np.array([a['effort'] for a in meta['actuators']])
    def refine(lo,hi,depth):
        a,b=nodes[lo],nodes[hi]
        fa,fb=np.array(a['thrust']),np.array(b['thrust'])
        ja,jb=[np.array([j['effort_nm'] for j in t['joints']]) for t in [a,b]]
        valid=True
        for u in [.25,.5,.75]:
            s=lo+u*(hi-lo);q=pin.interpolate(model,first,last,float(s))
            check=check_fixed_thrust(model,meta,q,(1-u)*fa+u*fb)
            torque=np.array([j['effort_nm'] for j in check['joints']])
            error=float(np.max(abs((torque-((1-u)*ja+u*jb))/limits)))
            if (check['max_normalized_joint_effort']>1.+1e-7 or
                    check['equilibrium_residual']>tolerance or error>tolerance):valid=False;break
        if valid:return
        if depth>=maximum_depth:raise ValueError('morph_support_interpolation_unresolved')
        mid=(lo+hi)/2;q=pin.interpolate(model,first,last,float(mid))
        trim=static_balance(model,meta,q,payload_fraction=0.,balance_thrust=True)
        if trim['status']!='static_screen_pass':
            error=ValueError('morph_intermediate_support_not_found')
            error.evidence=dict(fraction=float(mid),configuration=q.tolist())
            raise error
        nodes[mid]=trim;refine(lo,mid,depth+1);refine(mid,hi,depth+1)
    refine(0.,1.,0)
    return [(s,nodes[s]) for s in sorted(nodes)]


def scheduled_support(schedule,fraction):
    """Continuous interpolation of a checked, task-derived force schedule."""
    fractions=[s for s,_ in schedule]
    index=max(0,min(len(schedule)-2,int(np.searchsorted(fractions,fraction,side='right')-1)))
    lo,a=schedule[index];hi,b=schedule[index+1];u=float(np.clip((fraction-lo)/(hi-lo),0.,1.))
    force=(1-u)*np.array(a['thrust'])+u*np.array(b['thrust'])
    joint=(1-u)*np.array([j['effort_nm'] for j in a['joints']])+u*np.array([j['effort_nm'] for j in b['joints']])
    return joint,force


def refine_support_path(model,meta,path,distances,trims, *, maximum_insertions=8):
    """Repair an unsupported intermediate pose by fitting its rotor directions.

    Dock/base geometry stays on the original interpolation. New rotor poses
    require a separate collision check by the caller before execution.
    """
    path=list(path);distances=list(distances);trims=list(trims);schedules=[];repairs=[]
    index=0
    while index<len(path)-1:
        try:
            schedule=support_schedule(model,meta,path[index],path[index+1],trims[index],trims[index+1])
        except ValueError as error:
            if str(error)!='morph_intermediate_support_not_found' or len(repairs)>=maximum_insertions:raise
            evidence=error.evidence;fraction=evidence['fraction']
            q,trim=free_flight_trim(model,meta,np.array(evidence['configuration']))
            if trim['status']!='static_screen_pass':raise
            distance=(1-fraction)*distances[index]+fraction*distances[index+1]
            path.insert(index+1,q);distances.insert(index+1,distance);trims.insert(index+1,trim)
            repairs.append(dict(distance_m=float(distance),direction_fit_evaluations=trim.get('direction_fit_evaluations',0)))
            continue
        schedules.append(schedule);index+=1
    return path,np.asarray(distances),schedules,repairs
