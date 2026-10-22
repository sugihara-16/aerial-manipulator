"""Unconstrained quadratic optimum, accepted only inside ALL local bounds."""
import numpy as np


def direct_step(control,P,c,response,D,current,lower,upper):
    try:step=np.linalg.solve(P,-c)
    except np.linalg.LinAlgError:return None
    if not np.isfinite(step).all() or np.any(step<lower-1e-10) or np.any(step>upper+1e-10):return None
    m=control.m;states=response['states'];qs=states[:,1:1+m.nq];vs=states[:,1+m.nq:]
    displacement=D@step
    joint_q=qs[:,control.qi]+displacement[:,:m.nv][:,control.vi]
    joint_v=vs[:,control.vi]+displacement[:,m.nv:][:,control.vi]
    if (np.any(joint_q<control.qlo-1e-10) or np.any(joint_q>control.qhi+1e-10)
            or np.max(abs(joint_v),initial=0.)>3.+1e-10):return None
    loads=response['loads']+response['load_derivatives']@(control.input_scale*step)
    if np.min(loads[-1,:,0],initial=np.inf)<getattr(control,'minimum_normal_load',0.)-1e-10:return None
    if np.max(np.linalg.norm(loads[:,:,1:4],axis=2),initial=0.)>30.+1e-10:return None
    if np.max(np.linalg.norm(loads[:,:,4:7],axis=2),initial=0.)>5.+1e-10:return None
    return step
