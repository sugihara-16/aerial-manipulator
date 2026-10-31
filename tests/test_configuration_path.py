"""Curve derivatives agree with the simulator's position differences."""
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from amsrr_det.configuration_path import ConfigurationPath


def test_exact_axis_rotation_derivatives_and_record_replay():
    from types import SimpleNamespace
    from amsrr_det.configuration_path import AxisRotationPath,TabulatedContactPath
    m=mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body><freejoint/>
    <geom size=".1" mass="1"/></body></worldbody></mujoco>''')
    p=SimpleNamespace(mj=m,meta=dict(actuators=[]));q=m.qpos0.copy();q[:3]=[.4,-.2,1.2]
    q[3:7]=Rotation.from_rotvec([.3,-.2,.4]).as_quat()[[3,0,1,2]]
    for axis in [[0,0,1],[0,.6,-.8]]:
        path=AxisRotationPath(p,[0.,np.pi],q,axis,[.1,.2,.7],np.pi/2)
        restored=TabulatedContactPath.from_record(p,path.record())
        nodes=np.linspace(0,np.pi,725)
        np.testing.assert_allclose(path.velocities(nodes),[path.at(s)[1] for s in nodes],atol=1e-14)
        for s in [.1,.7,1.4,2.7]:
            pose,v,a=path.at(s);h=1e-5;qm,vm,_=path.at(s-h);qp,vp,_=path.at(s+h)
            fd=np.zeros(m.nv);mujoco.mj_differentiatePos(m,fd,2*h,qm,qp)
            np.testing.assert_allclose(v,fd,atol=1e-10)
            np.testing.assert_allclose(a,(vp-vm)/(2*h),atol=1e-10)
            for original,loaded in zip(path.at(s),restored.at(s)):np.testing.assert_array_equal(original,loaded)


def test_local_clock_slows_a_short_joint_turn_without_slowing_the_whole_path():
    from types import SimpleNamespace
    from amsrr_det.configuration_path import PathClock
    m=mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body><freejoint/>
    <geom size=".1" mass="1"/><body><joint name="joint" axis="0 1 0"/>
    <geom size=".03" mass=".1"/></body></body></worldbody></mujoco>''')
    nodes=np.array([0.,1.4,1.5,1.6,3.]);poses=np.tile(m.qpos0,(len(nodes),1))
    poses[:,-1]=[0.,0.,.5,1.,1.]
    curve=ConfigurationPath(m,nodes,poses)
    plant=SimpleNamespace(mj=m,meta=dict(actuators=[dict(name='joint',speed=1.)]))
    clock=PathClock(plant,curve)
    global_duration=1.875*3*max(abs(curve.at(x)[1][-1])/.5 for x in np.linspace(0,3,601))
    assert clock.duration<global_duration/2
    assert clock.at(0.)==(0.,0.,0.)
    np.testing.assert_allclose(clock.at(clock.duration),[3.,0.,0.],atol=1e-12)
    for time in np.linspace(.01,clock.duration-.01,301):
        angle,speed,acceleration=clock.at(time)
        assert speed>=0.
        assert abs(curve.at(angle)[1][-1]*speed)<=.5+1e-4
        h=1e-6
        numerical=(clock.at(time+h)[0]-clock.at(time-h)[0])/(2*h)
        assert abs(speed-numerical)<1e-7


def test_rotation_and_shape_derivatives_match_numerical_motion():
    m=mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body><freejoint/>
    <geom size=".1" mass="1"/><body><joint axis="0 1 0"/>
    <geom size=".03" mass=".1"/></body></body></worldbody></mujoco>''')
    nodes=np.linspace(0,3,13);poses=[]
    for s in nodes:
        q=m.qpos0.copy();q[:3]=[np.sin(s),.2*s*s,1+.1*np.cos(s)]
        q[3:7]=Rotation.from_rotvec([.3*np.sin(s),.2*s,.5*s]).as_quat()[[3,0,1,2]]
        q[7]=.4*np.sin(2*s);poses.append(q)
    curve=ConfigurationPath(m,nodes,poses);h=1e-5
    for s in [.13,.73,1.41,2.65]:
        q,v,a=curve.at(s);qa,va,_=curve.at(s-h);qb,vb,_=curve.at(s+h)
        fd=np.zeros(m.nv);mujoco.mj_differentiatePos(m,fd,2*h,qa,qb)
        np.testing.assert_allclose(v,fd,atol=1e-7)
        np.testing.assert_allclose(a,(vb-va)/(2*h),atol=1e-7)
    for s,q in zip(nodes,poses):np.testing.assert_allclose(curve.at(s)[0],q,atol=1e-13)


def test_material_frame_projection_and_velocity_use_the_same_curve():
    import pinocchio as pin
    from amsrr_det.valve import ValvePlant,ValveSpec
    from amsrr_det.grasp_geometry import grasp_frame
    from amsrr_det.configuration_path import ContactConfigurationPath
    plant=ValvePlant(1,[],ValveSpec(),meshes=False).control
    angles=np.linspace(0,1,5);poses=[]
    for s in angles:
        q=plant.mj.qpos0.copy();q[:3]=[.4,.1*np.sin(s),1.5]
        q[3:7]=Rotation.from_rotvec([.1*s,.2*s,.3*s]).as_quat()[[3,0,1,2]]
        oi=plant.mj.jnt_qposadr[plant.mj.joint('object').id]
        q[oi:oi+7]=ValveSpec().pose(s);poses.append(q)
    curve=ContactConfigurationPath(plant,angles,poses,[(0,'yaw1')],[[0.,1.,0.]])
    data=plant.pin.createData();pq,mq,*_=plant.coordinate_maps
    from amsrr_det.configuration_path import TabulatedContactPath
    table=TabulatedContactPath(plant,angles,poses,[(0,'yaw1')],[[0.,1.,0.]])
    for s in [.13,.37,.81]:
        q,v,a=curve.at(s);qp=pin.neutral(plant.pin);qp[pq]=q[mq]
        pin.framesForwardKinematics(plant.pin,data,qp)
        local=data.oMf[plant.meta['object']['frame']].inverse()*data.oMf[plant.meta['frames'][grasp_frame(0,'yaw1')]]
        np.testing.assert_allclose(local.rotation[:,0],[0.,1.,0.],atol=1e-12)
        h=2e-5;fd=np.zeros(plant.mj.nv)
        mujoco.mj_differentiatePos(plant.mj,fd,2*h,curve.pose(s-h),curve.pose(s+h))
        np.testing.assert_allclose(v,fd,atol=1e-6)
        tq,tv,ta=table.at(s)
        np.testing.assert_allclose(tq,q,atol=2e-6)
        mujoco.mj_differentiatePos(plant.mj,fd,2*h,table.pose(s-h),table.pose(s+h))
        np.testing.assert_allclose(tv,fd,atol=1e-7)
        np.testing.assert_allclose(ta,(table.at(s+h)[1]-table.at(s-h)[1])/(2*h),atol=1e-5)


def test_direction_repairs_preserve_fitted_geometry_and_its_derivatives():
    from amsrr_det.valve import ValvePlant,ValveSpec
    from amsrr_det.configuration_path import TabulatedContactPath
    plant=ValvePlant(1,[],ValveSpec(),meshes=False).control
    angles=np.linspace(0,1,5);poses=[]
    for angle in angles:
        q=plant.mj.qpos0.copy();q[:3]=[.4,.1*np.sin(angle),1.5]
        q[3:7]=Rotation.from_rotvec([.1*angle,.2*angle,.3*angle]).as_quat()[[3,0,1,2]]
        oi=plant.mj.jnt_qposadr[plant.mj.joint('object').id]
        q[oi:oi+7]=ValveSpec().pose(angle);poses.append(q)
    curve=TabulatedContactPath(plant,angles,poses,[(0,'yaw1')],[[0.,1.,0.]])
    probes=[.13,.37,.81];before=[curve.at(x) for x in probes]
    nodes=np.sort(np.r_[angles,.33,.69]);fixed=np.array([curve.pose(x) for x in nodes])
    fixed[:,curve.gq]=np.sin(nodes[:,None]*4)*.4
    curve.set_directions(nodes,fixed)
    qmask=np.ones(plant.mj.nq,bool);qmask[curve.gq]=False
    vmask=np.ones(plant.mj.nv,bool);vmask[curve.gv]=False
    for x,old in zip(probes,before):
        q,v,a=curve.at(x)
        for new,previous,mask in zip((q,v,a),old,(qmask,vmask,vmask)):
            np.testing.assert_array_equal(new[mask],previous[mask])
        h=1e-6;fd=np.zeros(plant.mj.nv)
        mujoco.mj_differentiatePos(plant.mj,fd,2*h,curve.pose(x-h),curve.pose(x+h))
        np.testing.assert_allclose(v,fd,atol=1e-7)
        np.testing.assert_allclose(a,(curve.at(x+h)[1]-curve.at(x-h)[1])/(2*h),atol=1e-5)


def test_projected_valve_path_keeps_real_material_in_contact():
    import json
    from pathlib import Path
    from amsrr_det.valve import ValveSpec,ValvePlant
    from amsrr_det.configuration_path import TabulatedContactPath
    saved=json.loads(Path(__file__).with_name('fixtures').joinpath('valve_path_input.json').read_text())
    spec=ValveSpec(**saved['request']);system=ValvePlant(1,[],spec);p=system.control
    system.configure_contacts(['m0/yaw_dock_mech1'])
    curve=TabulatedContactPath(p,saved['angles'],saved['path'],saved['surfaces'],saved['normals'],object_shape=spec.shape)
    assert np.diff(curve.spline.x).min()>1e-6
    import pinocchio as pin
    from amsrr_det.grasp_geometry import grasp_frame
    pdata=p.pin.createData();pq,mq,*_=p.coordinate_maps
    material=[]
    for angle in np.linspace(0,np.pi,361):
        qp=pin.neutral(p.pin);qp[pq]=curve.pose(angle)[mq]
        pin.framesForwardKinematics(p.pin,pdata,qp)
        T=pdata.oMf[p.meta['object']['frame']].inverse()*pdata.oMf[p.meta['frames'][grasp_frame(*saved['surfaces'][0])]]
        material.append(T.copy())
    origin=material[0]
    assert max(np.linalg.norm(T.translation-origin.translation) for T in material)<1e-5
    assert max(np.linalg.norm(pin.log3(origin.rotation.T@T.rotation)) for T in material)<1e-5
    d=mujoco.MjData(p.mj);oid=p.mj.geom('object').id
    for angle in np.linspace(0,np.pi,13):
        d.qpos[:]=curve.pose(angle);mujoco.mj_fwdPosition(p.mj,d)
        active=[c for c in d.contact if oid in c.geom and c.dist<0]
        assert active
        assert min(c.dist for c in active)>-.0003
    restored=TabulatedContactPath.from_record(p,json.loads(json.dumps(curve.record(),default=lambda v:v.tolist())))
    for angle in [.01,.25,1.23,3.1]:
        for left,right in zip(curve.at(angle),restored.at(angle)):
            np.testing.assert_array_equal(left,right)

    nodes=np.linspace(0,np.pi,725)
    np.testing.assert_allclose(restored.velocities(nodes),[restored.at(x)[1] for x in nodes],atol=1e-13)
    from types import SimpleNamespace
    from amsrr_det.configuration_path import PathClock
    batched=PathClock(p,restored)
    original=PathClock(p,SimpleNamespace(parameters=restored.parameters,at=restored.at))
    np.testing.assert_allclose(batched.duration,original.duration,atol=1e-12,rtol=1e-13)
    for t in np.linspace(0,batched.duration,101):
        np.testing.assert_allclose(batched.at(t),original.at(t),atol=1e-12,rtol=1e-12)
