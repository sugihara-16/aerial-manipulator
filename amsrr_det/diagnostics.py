"""Explicit diagnostic inputs; none is a full task acceptance fixture."""
from pathlib import Path
import json
import numpy as np
import pinocchio as pin
from .model import ROOT, export_model


def fixture(directory, n=1, payload=None):
    # A construction seed for dynamics testing, not a task-dependent lookup.
    edges=[dict(a=i-1, port_a="pitch1", b=i, port_b="yaw2") for i in range(1,n)]
    model, metadata=export_model(directory,n,edges,payload)
    q=pin.neutral(model);q[2]=1.
    if payload is not None:
        q[metadata['object']['q']:metadata['object']['q']+3]=[0.,0.,0.5]
    data=model.createData();pin.forwardKinematics(model,data,q);pin.updateFramePlacements(model,data)
    g=pin.computeGeneralizedGravity(model,data,q)
    # Rotor force and moment allocation at the actual neutral geometry.
    A=np.zeros((6,4*n))
    for i,r in enumerate(metadata['rotors']):
        frame=data.oMf[r['frame']];axis=frame.rotation[:,2]
        A[:3,i]=axis;A[3:,i]=np.cross(frame.translation-q[:3],axis)
    force=np.linalg.lstsq(A,g[:6],rcond=None)[0]
    ext=[pin.Force.Zero() for _ in range(model.njoints)]
    for r,f in zip(metadata['rotors'],force):
        frame=model.frames[r['frame']]
        ext[frame.parentJoint]+=frame.placement.act(pin.Force(np.array([0.,0.,f]),np.zeros(3)))
    effort=pin.rnea(model,data,q,np.zeros(model.nv),np.zeros(model.nv),ext)
    preload=[float(q[a['q']]+effort[a['v']]/a['kp']) for a in metadata['actuators']]
    scenario=dict(schema="amsrr_det.scenario.v1",name=f"neutral_hover_n{n}",duration_s=100.,control_period_s=.01,
        q0=q.tolist(),f0=force.tolist(),tau0=[float(effort[a['v']]) for a in metadata['actuators']],
        initial_thrust_estimate=force.tolist(),
        command={'thrust':force.tolist(),'position':preload},
        thrust_lag_s=.03,gimbal_torque_lag_s=.01,dock_torque_lag_s=.0022,
        allocation_gimbal_predictor_s=.05,stop_on_limit_violation=True,
        assumptions=["nominal unmeasured thrust and torque response", "initial thrust estimate equals nominal hover, a diagnostic assumption rather than a sensor", "prepared structure; assembly excluded", "diagnostic only; not task acceptance"])
    directory=Path(directory);(directory/'scenario.json').write_text(json.dumps(scenario,indent=2)+'\n')
    return model,metadata,scenario


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--modules',type=int,default=1);p.add_argument('--directory',type=Path,required=True)
    args=p.parse_args();m,d,s=fixture(args.directory,args.modules)
    print(json.dumps(dict(directory=str(args.directory),nq=m.nq,nv=m.nv,mass=d['mass'],thrust=s['f0'])))
