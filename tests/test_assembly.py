import numpy as np
import pinocchio as pin
import mujoco
from amsrr_det.assembly import make_assembly_plant,connected_view,forest_posture,actuator_indices,dock_error,ready_to_attach


def test_same_physical_state_and_commands_survive_controller_topology_change():
    edges=[dict(a=0,port_a='pitch1',b=1,port_b='yaw1')]
    p=make_assembly_plant(2,edges,meshes=False);view=connected_view(p,edges)
    q=pin.neutral(view.pin);q[2]=1.
    for i,a in enumerate(view.meta['actuators']):q[a['q']]=.03*np.sin(i)
    fq=forest_posture(p,view,q);d=mujoco.MjData(p.mj);d.qpos[:]=p.configuration(fq)
    mujoco.mj_forward(p.mj,d);observed,_=view.observation(d)
    np.testing.assert_allclose(observed,q,atol=1e-12)
    assert ready_to_attach(dock_error(p,fq,np.zeros(p.pin.nv),edges[0]))
    assert not d.eq_active.any()
    initial=d.qpos.copy();d.eq_active[0]=1
    np.testing.assert_array_equal(d.qpos,initial)
    indices=actuator_indices(view)
    for a,i in zip(view.meta['actuators'],indices):assert p.mj.actuator(i).name==a['name']
    # Nonzero relative speed must prevent an early latch even at perfect pose.
    v=np.zeros(p.pin.nv);v[p.meta['bases'][1]['v']]=.02
    assert not ready_to_attach(dock_error(p,fq,v,edges[0]))


def test_connected_inertia_bias_and_rotor_force_are_the_forest_projection():
    edges=[dict(a=0,port_a='pitch2',b=1,port_b='yaw2')]
    p=make_assembly_plant(2,edges,meshes=False);c=connected_view(p,edges)
    q=pin.neutral(c.pin);q[2]=1.;q[3:7]=pin.Quaternion(pin.exp3(np.array([.08,-.04,.2]))).coeffs()
    for i,a in enumerate(c.meta['actuators']):q[a['q']]=.1*np.sin(i)
    vf=np.zeros(p.pin.nv);v=np.random.default_rng(194).normal(0,.1,c.pin.nv)
    def velocity_map(configuration):
        data=c.pin.createData();pin.computeJointJacobians(c.pin,data,configuration);pin.updateFramePlacements(c.pin,data)
        T=np.zeros((p.pin.nv,c.pin.nv))
        for b in p.meta['bases']:
            T[b['v']:b['v']+6]=pin.getFrameJacobian(c.pin,data,c.meta['frames'][f"m{b['module']}/root"],pin.LOCAL)
        active={a['name']:a for a in c.meta['actuators']}
        for a in p.meta['actuators']:T[a['v'],active[a['name']]['v']]=1.
        return T
    T=velocity_map(q);fq=forest_posture(p,c,q);vf=T@v
    Mf=pin.crba(p.pin,p.pin.createData(),fq);Mt=pin.crba(c.pin,c.pin.createData(),q)
    np.testing.assert_allclose(T.T@Mf@T,Mt,atol=1e-11)
    epsilon=1e-6
    Tdot=(velocity_map(pin.integrate(c.pin,q,epsilon*v))-velocity_map(pin.integrate(c.pin,q,-epsilon*v)))/(2*epsilon)
    hf=pin.rnea(p.pin,p.pin.createData(),fq,vf,np.zeros(p.pin.nv))
    ht=pin.rnea(c.pin,c.pin.createData(),q,v,np.zeros(c.pin.nv))
    np.testing.assert_allclose(T.T@(hf+Mf@Tdot@v),ht,atol=1e-8)
    forces=np.linspace(1.,4.,len(p.meta['rotors']))
    def rotor_generalized(view,pose):
        ext=[pin.Force.Zero() for _ in range(view.pin.njoints)]
        for r,f in zip(view.meta['rotors'],forces):
            frame=view.pin.frames[r['frame']]
            ext[frame.parentJoint]+=frame.placement.act(pin.Force(np.array([0.,0.,f]),np.zeros(3)))
        data=view.pin.createData();zero=np.zeros(view.pin.nv)
        return pin.rnea(view.pin,data,pose,zero,zero).copy()-pin.rnea(view.pin,data,pose,zero,zero,ext)
    np.testing.assert_allclose(T.T@rotor_generalized(p,fq),rotor_generalized(c,q),atol=1e-11)


def test_alignment_target_tracks_observed_joint_deflections():
    from amsrr_det.assembly import following_base_target,make_assembly_plant,dock_error
    edge=dict(a=0,port_a='pitch2',b=1,port_b='yaw2')
    plant=make_assembly_plant(2,[edge],meshes=False)
    q=pin.neutral(plant.pin)
    for base in plant.meta['bases']:q[base['q']:base['q']+3]=[.3,-.2,1.1]
    for a in plant.meta['actuators']:
        if a['role']=='dock':q[a['q']]=.07 if a['name'].startswith('m0') else -.04
    target=following_base_target(plant,q,edge,.15);base=plant.meta['bases'][1];q[base['q']:base['q']+7]=target
    error=dock_error(plant,q,np.zeros(plant.pin.nv),edge)
    assert abs(error['axial_m']-.15)<1e-12
    assert error['transverse_m']<1e-12 and error['attitude_rad']<1e-12
