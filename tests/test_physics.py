import json
import subprocess
from pathlib import Path
import numpy as np
import pinocchio as pin
import pytest
from amsrr_det.model import ROOT,export_model

BINARY=ROOT/'build/det_rollout'

def execute(tmp,model,meta,scenario,h=.001,closed=False):
    path=tmp/'input.json';out=tmp/'output.json'
    path.write_text(json.dumps(scenario))
    args=[str(BINARY),str(tmp),str(path),str(h),str(out)]
    if closed:args.append('closed')
    result=subprocess.run(args,capture_output=True,text=True,timeout=30)
    if result.returncode not in [0,3]:raise AssertionError(result.stderr)
    return json.loads(out.read_text())

def test_freefall_acceleration(tmp_path):
    model,meta=export_model(tmp_path)
    r=execute(tmp_path,model,meta,dict(probe=True))
    expected=np.zeros(model.nv);expected[:3]=model.gravity.linear
    np.testing.assert_allclose(r['acceleration'],expected,atol=2e-12)

def test_native_forces_match_inverse_dynamics(tmp_path):
    model,meta=export_model(tmp_path,3,[dict(a=0,port_a='pitch1',b=1,port_b='yaw2'),dict(a=1,port_a='pitch1',b=2,port_b='yaw2')])
    rng=np.random.default_rng(41);q=pin.neutral(model)
    for a in meta['actuators']:q[a['q']]=rng.uniform(-.3,.3)
    v=rng.normal(0,.15,model.nv);f=rng.uniform(0,8,len(meta['rotors']));tau=rng.uniform(-.1,.1,len(meta['actuators']))
    r=execute(tmp_path,model,meta,dict(probe=True,q0=q.tolist(),v0=v.tolist(),f0=f.tolist(),tau0=tau.tolist(),gimbal_torque_lag_s=.01,dock_torque_lag_s=.0022))
    data=model.createData();pin.forwardKinematics(model,data,q,v);pin.updateFramePlacements(model,data)
    ext=[pin.Force.Zero() for _ in range(model.njoints)]
    for rotor,fi in zip(meta['rotors'],f):
        frame=model.frames[rotor['frame']]
        ext[frame.parentJoint]+=frame.placement.act(pin.Force(np.array([0.,0.,fi]),np.zeros(3)))
    required=pin.rnea(model,data,q,v,np.array(r['acceleration']),ext)
    applied=np.zeros(model.nv)
    for a,t in zip(meta['actuators'],tau):applied[a['v']]=t
    np.testing.assert_allclose(required,applied,atol=2e-11)

def patch(frame='m0/pitch_grasp_contact_frame_1',target=None):
    z=dict(frame=frame,points=[[0.,y,z] for y,z in [(-.005,-.005),(-.005,.005),(.005,-.005),(.005,.005)]],
           normal=[1.,0.,0.],offset=0.,kn=7500.,cn=75.,kt=7500.,ct=1.,mu=4.5,
           lower=[-100.,-100.,-100.],upper=[100.,100.,100.])
    if target:z['target_frame']=target
    return z

def test_contact_force_and_moment_match_jacobian_virtual_work(tmp_path):
    model,meta=export_model(tmp_path)
    q=pin.neutral(model);data=model.createData();pin.framesForwardKinematics(model,data,q)
    p=patch();fid=meta['frames'][p['frame']]
    # Plane follows the pad's normal at this fixed pose; four equal depths.
    normal=data.oMf[fid].rotation[:,0]
    p['normal']=normal.tolist();p['offset']=float(normal@data.oMf[fid].translation+.001)
    r=execute(tmp_path,model,meta,dict(probe=True,q0=q.tolist(),patches=[p]))
    applied=pin.rnea(model,data,q,np.zeros(model.nv),np.array(r['acceleration']))
    J=pin.computeFrameJacobian(model,data,q,fid,pin.LOCAL_WORLD_ALIGNED)
    expected=J[:3].T@(7.5*normal)
    np.testing.assert_allclose(applied,expected,atol=2e-11)

def test_contact_action_reaction_conserves_total_momentum(tmp_path):
    model,meta=export_model(tmp_path,payload=dict(mass=1.,size=[.3,.4,.15]))
    q=pin.neutral(model);data=model.createData();pin.framesForwardKinematics(model,data,q)
    p=patch(target='object');fid=meta['frames'][p['frame']]
    normal=data.oMf[fid].rotation[:,0];p['normal']=normal.tolist();p['offset']=float(normal@data.oMf[fid].translation+.001)
    r=execute(tmp_path,model,meta,dict(probe=True,q0=q.tolist(),patches=[p]))
    A=pin.computeCentroidalMap(model,data,q)
    rate=A@np.array(r['acceleration'])
    expected=np.zeros(6);expected[:3]=model.gravity.linear*meta['mass']
    np.testing.assert_allclose(rate,expected,atol=3e-11)

def test_freefall_trajectory_and_work_counts(tmp_path):
    model,meta=export_model(tmp_path)
    for h in [.002,.001]:
        r=execute(tmp_path,model,meta,dict(duration_s=.1,control_period_s=.01,gimbal_torque_lag_s=.01,dock_torque_lag_s=.0022),h)
        assert r['full_horizon'];assert not r['task_accepted']
        assert r['counts']['rhs']==2*r['counts']['steps']
        assert r['counts']['control']==10
        assert r['counts']['steps']==round(.1/h)
        assert r['final']['q'][2]==pytest.approx(1.+.5*model.gravity.linear[2]*.1**2,abs=2e-13)
        assert r['final']['v'][2]==pytest.approx(model.gravity.linear[2]*.1,abs=2e-13)

def test_limits_reject_unsafe_initial_commands(tmp_path):
    model,meta=export_model(tmp_path)
    source=tmp_path/'input.json';source.write_text(json.dumps(dict(duration_s=.1,command={'thrust':[21.]*4})))
    r=subprocess.run([str(BINARY),str(tmp_path),str(source),'.001',str(tmp_path/'out.json')],capture_output=True,text=True)
    assert r.returncode==2
    assert 'thrust command limit' in r.stderr
