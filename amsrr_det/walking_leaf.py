"""Finite continuous motion of the three real hinges on a new leaf.

This is a generation preference under a frozen parent-body approximation.
It is not a whole-body support or dynamic certificate. Each phase starts at
the preceding joint state and the last must return to the initial state.
"""
from functools import lru_cache
import numpy as np
from amsrr_det.task_structure import part_frames,port_axes,port_origins


@lru_cache(maxsize=1)
def constants():
    _,_,seams,pads,*_=part_frames(True)
    axes=port_axes();origins=port_origins();S=np.zeros((4,3,3))
    for k,(x,y,z)in enumerate(axes):S[k]=[[0,-z,y],[z,0,-x],[-y,x,0]]
    return seams[:,2],pads[:,2],axes,origins,S


def evaluate(parent,out,incoming,pad,x):
    seams,pads,axes,origins,S=constants();frames=[]
    for port,angle,frame in [(out,x[0],seams[out]),(incoming,x[1],seams[incoming]),(pad,x[2],pads[pad])]:
        R=np.eye(3)+np.sin(angle)*S[port]+(1-np.cos(angle))*(S[port]@S[port]);F=np.eye(4);F[:3,:3]=R;F[:3,3]=origins[port]-R@origins[port];frames.append(F@frame)
    inv=np.eye(4);inv[:3,:3]=frames[1][:3,:3].T;inv[:3,3]=-inv[:3,:3]@frames[1][:3,3]
    child=parent@frames[0]@np.diag([-1.,-1.,1.,1.])@inv;foot=child@frames[2];p=foot[:3,3]
    axis=np.array([parent[:3,:3]@axes[out],-child[:3,:3]@axes[incoming],child[:3,:3]@axes[pad]])
    origins=np.array([parent[:3,3]+parent[:3,:3]@origins[out],child[:3,3]+child[:3,:3]@origins[incoming],child[:3,3]+child[:3,:3]@origins[pad]])
    J=np.cross(axis,p-origins).T
    return p,J,child


def reference_score(parent,out,incoming,pad,x,velocity,period,duty,*,height=.03,samples=12):
    x=np.array(x,dtype=float);initial=x.copy();p0,_,_=evaluate(parent,out,incoming,pad,x);errors=[];values=[]
    for step in range(samples+1):
        phase=(.5*duty+step/samples)%1.;s=np.clip((phase-duty)/(1-duty),0.,1.)
        blend=s**3*(10-15*s+6*s*s);target=p0+velocity*period*(.5*duty-phase+blend);target[2]+=64*height*s**3*(1-s)**3
        for _ in range(12):
            p,J,_=evaluate(parent,out,incoming,pad,x);err=p-target
            delta=np.linalg.solve(J.T@J+1e-5*np.eye(3),-J.T@err-1e-5*(x-initial))
            x=np.clip(x+delta*min(1.,.12/max(abs(delta).max(),1e-9)),-np.pi/2+.025,np.pi/2-.025)
            if max(abs(delta))<1e-5:break
        p,_,_=evaluate(parent,out,incoming,pad,x);errors.append(float(np.linalg.norm(p-target)));values.append(x.copy())
    values=np.array(values);closure=float(np.linalg.norm(values[-1]-values[0]));speed=float(abs(np.diff(values,axis=0)).max()*samples/period)
    # Error, finite closure and travel effort guide actual port selection.
    # No arbitrary successful/failed label or full-task simulation is used.
    cost=4*max(errors)**2+np.mean(np.array(errors)**2)+.03**2*closure**2+.002**2*np.mean(np.diff(values,axis=0)**2)
    return dict(cost=float(cost),max_error=max(errors),closure=closure,peak_grid_speed=speed,configurations=values.tolist())


def score(parent,out,incoming,pad,x,velocity,period,duty,*,height=.03,samples=12):
    import ctypes as ct
    from .walking_native import library
    fn=library().walking_leaf_motion
    fn.argtypes=[ct.c_void_p]*5+[ct.c_int]*3+[ct.c_void_p]*2+[ct.c_double]*3+[ct.c_int]+[ct.c_void_p]*2
    fn.restype=None
    seams,pads,axes,origins,_=constants()
    arrays=[np.ascontiguousarray(a,dtype=float) for a in [seams,pads,axes,origins,parent,x,velocity]]
    result=np.empty(4);values=np.empty((samples+1,3))
    fn(*[a.ctypes.data for a in arrays[:5]],out,incoming,pad,*[a.ctypes.data for a in arrays[5:]],period,duty,height,samples,result.ctypes.data,values.ctypes.data)
    return dict(cost=float(result[0]),max_error=float(result[1]),closure=float(result[2]),peak_grid_speed=float(result[3]),configurations=values.tolist())
