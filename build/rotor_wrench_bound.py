"""Analytical dual lower bound on hover input burden, without task rollouts.

Scope: fixed Dock/base/object geometry, all allowed gimbal angles. The dual
allows all real force directions; omitting angle limits relaxes to the plane.
Gimbal-dependent mass positions are enclosed, not frozen at the initial angle.
This does not bound all Dock/base postures or certify a complete structure.
"""
import numpy as np
import pinocchio as pin
from scipy.optimize import minimize


def sinusoid_interval(constant, cosine, sine, lower, upper):
    """Coordinate-wise extrema of c+a*cos(theta)+b*sin(theta)."""
    lo=[]; hi=[]
    for c,a,b in zip(constant,cosine,sine):
        angles=[lower,upper]
        peak=np.arctan2(b,a)
        for k in range(int(np.ceil((lower-peak)/np.pi)),int(np.floor((upper-peak)/np.pi))+1):
            angle=peak+k*np.pi
            if lower<=angle<=upper: angles.append(angle)
        values=c+a*np.cos(angles)+b*np.sin(angles)
        lo.append(float(min(values))-1e-12); hi.append(float(max(values))+1e-12)
    return np.asarray(lo),np.asarray(hi)


def rotor_plane_problem(model,meta,q):
    data=model.createData(); pin.computeJointJacobians(model,data,q);pin.updateFramePlacements(model,data)
    origin=data.oMi[meta['bases'][0]['joint']].translation
    weight=-np.asarray(model.gravity.linear)
    moving={a['joint']:a for a in meta['actuators'] if a['role']=='vectoring'}
    moment_lo=np.zeros(3);moment_hi=np.zeros(3);mass=0.
    for j,inertia in enumerate(model.inertias):
        if not inertia.mass: continue
        mass+=inertia.mass;T=data.oMi[j]
        lever=T.rotation@inertia.lever
        if j in moving:
            a=moving[j]
            J=pin.getJointJacobian(model,data,j,pin.LOCAL_WORLD_ALIGNED)
            axis=J[3:,a['v']]
            initial=pin.exp3(-axis*q[a['q']])@lever
            parallel=axis*(axis@initial); perpendicular=initial-parallel
            lo,hi=sinusoid_interval(T.translation-origin+parallel,perpendicular,
                np.cross(axis,perpendicular),a['lower'],a['upper'])
        else: lo=hi=T.translation-origin+lever
        matrix=-pin.skew(inertia.mass*weight)
        middle=matrix@((lo+hi)/2);radius=abs(matrix)@((hi-lo)/2)
        moment_lo+=middle-radius; moment_hi+=middle+radius
    maps=[]; caps=[]; angles=[]
    for rotor in meta['rotors']:
        a=meta['actuators'][rotor['gimbal']];T=data.oMf[rotor['frame']]
        J=pin.getJointJacobian(model,data,a['joint'],pin.LOCAL_WORLD_ALIGNED)
        axis=J[3:,a['v']];direction=T.rotation[:,2]
        center=data.oMi[a['joint']].translation
        if abs(axis@direction)>1e-10: raise ValueError('rotor direction not perpendicular to gimbal axis')
        if np.linalg.norm(np.cross(T.translation-center,direction))>1e-9:
            raise ValueError('rotor thrust line does not pass through its gimbal origin')
        zero_direction=pin.exp3(-axis*q[a['q']])@direction
        basis=np.c_[zero_direction,np.cross(axis,zero_direction)]
        maps.append(np.vstack([basis,pin.skew(center-origin)@basis]));caps.append(rotor['maximum'])
        angles.append([a['lower'],a['upper']])
    middle=np.r_[mass*weight,(moment_lo+moment_hi)/2]
    radius=np.r_[np.zeros(3),(moment_hi-moment_lo)/2]
    return dict(maps=np.asarray(maps),caps=np.asarray(caps),middle=middle,radius=radius,
                angle_ranges=np.asarray(angles))


def wrench_input_lower_bound(maps,caps,middle,radius,angle_ranges=None):
    """Any returned dual vector yields a bound, even if optimization stops early.

    For lambda, min_w lambda.w minus sum sup_f(lambda.A.f-||f||^1.5)
    bounds every feasible input from below. With ||f||<=cap, the supremum
    is a one-dimensional radial problem with an explicit solution.
    """
    maps=np.asarray(maps);caps=np.asarray(caps);middle=np.asarray(middle);radius=np.asarray(radius)
    if maps.shape!=(len(caps),6,2) or middle.shape!=(6,) or radius.shape!=(6,):
        raise ValueError('invalid rotor maps or wrench interval')
    if (np.any(caps<=0) or np.any(radius<0)
            or not all(np.isfinite(a).all() for a in (maps,caps,middle,radius))):
        raise ValueError('invalid limits or non-finite inputs')
    angles=np.tile([-np.pi,np.pi],(len(caps),1)) if angle_ranges is None else np.asarray(angle_ranges)
    if angles.shape!=(len(caps),2) or not np.isfinite(angles).all() or np.any(angles[:,0]>angles[:,1]):
        raise ValueError('invalid gimbal ranges')
    def dual(vector):
        projections=np.einsum('ijk,j->ik',maps,vector)
        peak=np.arctan2(projections[:,1],projections[:,0])
        peak+=2*np.pi*np.round((angles.mean(axis=1)-peak)/(2*np.pi))
        trials=np.c_[angles,np.clip(peak,angles[:,0],angles[:,1])]
        scores=projections[:,0,None]*np.cos(trials)+projections[:,1,None]*np.sin(trials)
        choice=np.argmax(scores,axis=1);best=trials[np.arange(len(caps)),choice]
        support=np.maximum(0.,scores[np.arange(len(caps)),choice])
        thrust=np.minimum((support/1.5)**2,caps)
        conjugates=support*thrust-thrust**1.5
        forces=thrust[:,None]*np.c_[np.cos(best),np.sin(best)]
        value=vector@middle-abs(vector)@radius-conjugates.sum()
        gradient=middle-np.sign(vector)*radius-np.einsum('ijk,ik->j',maps,forces)
        return -float(value),-gradient
    start=np.zeros(6)
    size=np.linalg.norm(middle[:3])
    if size: start[:3]=1.5*np.sqrt(size/len(caps))*middle[:3]/size
    # Bounding the dual search changes only its tightness, not its validity.
    # It avoids unbounded numerical iterates when the relaxation is infeasible.
    result=minimize(dual,start,jac=True,method='L-BFGS-B',bounds=[(-1000.,1000.)]*6,
                    options=dict(maxiter=80,gtol=1e-8,ftol=1e-12))
    candidates=[np.zeros(6),start,result.x]
    vector=min(candidates,key=lambda x:dual(x)[0]); value=-dual(vector)[0]
    # Floating-point allowance; this is not interval-arithmetic certification.
    allowance=1e-8*(1+abs(value)+np.linalg.norm(vector)*(np.linalg.norm(middle)+np.linalg.norm(radius)))
    return dict(lower_bound=max(0.,value-allowance),dual_vector=vector.tolist(),
                raw_bound=value,rounding_allowance=allowance,iterations=int(result.nit),
                optimizer_converged=bool(result.success),scope=__doc__)
