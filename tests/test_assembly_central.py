import numpy as np
import mujoco

from amsrr_det.assembly import make_assembly_plant
from amsrr_det.assembly_central import SmoothHandoff
from amsrr_det.assembly_fsm import AssemblyFSM
from amsrr_det.assembly_plan import prepare


def test_handoff_preserves_first_output_and_live_feedback_with_physical_bounds():
    edges=[dict(a=0,b=1,port_a='pitch1',port_b='yaw1')]
    plant=make_assembly_plant(2,edges)
    smooth=SmoothHandoff(plant,edges)
    nj=len(plant.meta['actuators'])
    old=np.r_[np.zeros(nj),np.full(len(plant.meta['rotors']),4.)]
    requested=old.copy();requested[nj:]=6.
    smooth.note([True],old,1.)
    first=smooth.command(requested,old,1.02)
    np.testing.assert_allclose(first,old)
    # A live change is not suppressed while the topology bias fades.
    second_request=requested.copy();second_request[nj:]+=1.
    second=smooth.command(second_request,first,1.02)
    np.testing.assert_allclose(second[nj:],old[nj:]+1.)
    final=smooth.command(second_request,second,1.63)
    np.testing.assert_allclose(final,second_request)
    assert smooth.ready(1.63,dict(central_mask=1))
    assert not smooth.ready(1.63,dict(central_mask=0))
    requested[nj:]=-20.
    bounded=smooth.command(requested,final,1.64)
    assert np.all(bounded>=smooth.lower) and np.all(bounded<=smooth.upper)


def test_final_hover_requires_completed_control_handoff_and_elapsed_observations():
    plan=prepare(dict(modules=1,edges=[],initial_seed=10))
    fsm=AssemblyFSM(plan);data=mujoco.MjData(plan['plant'].mj)
    target=plan['initial'].copy();target[plan['qi']+2]=1.5
    data.qpos[:]=target;data.qvel[:]=0.
    fsm.clock=0.;fsm.stage='VERIFY';fsm.start=target.copy();fsm.end=target.copy()
    fsm.control_ready=False
    data.time=2.;fsm.update(data)
    assert fsm.stable_since is None and not fsm.success
    fsm.control_ready=True
    fsm.update(data)
    for _ in range(150):fsm.update(data)
    assert not fsm.success
    data.time=2.99;fsm.update(data)
    assert not fsm.success
    data.time=3.;fsm.update(data)
    assert fsm.success


def test_connected_allocation_couples_member_inputs():
    from amsrr_det.assembly_central import CentralAssemblyControl
    from amsrr_det.airborne_control import AirborneControl
    edges=[dict(a=0,b=1,port_a='pitch1',port_b='yaw1')]
    plan=prepare(dict(modules=2,edges=edges,initial_seed=10))
    shared=CentralAssemblyControl(plan);physical=plan['plant']
    q=plan['closed'];v=np.zeros(physical.mj.nv)
    nominal=shared.single.update(q,v,q,v,v,np.array([True]),observation_time=0.)
    view,_,indices,_,_,mapping=shared.views[1]
    qs,qd,vs,vd=mapping
    reduced=view.mj.qpos0.copy();reduced[qd]=q[qs];zero=np.zeros(view.mj.nv)
    old=nominal[indices];nj=len(view.meta['actuators'])
    results=[]
    for extra in [0.,.4]:
        controller=AirborneControl(view);controller.seed(reduced,zero,old)
        preferred=old.copy()
        desired_acc={i:a.copy()for i,a in shared.single.acceleration_requests.items()}
        desired_acc[0][2]+=extra
        results.append(controller.update(reduced,zero,reduced,zero,zero,0.,nominal_input=preferred,acceleration_requests=desired_acc))
    other=[nj+i for i,r in enumerate(view.meta['rotors'])if r['module']==1]
    # An independent allocator for module 1 would see identical inputs and
    # return identical commands. Shared dynamics changes its allocation.
    assert np.linalg.norm(results[1][other]-results[0][other])>1e-4


def test_recovery_reference_is_continuous_and_does_not_reset_physics():
    edges=[dict(a=0,b=1,port_a='pitch1',port_b='yaw1')]
    plan=prepare(dict(modules=2,edges=edges,initial_seed=10))
    fsm=AssemblyFSM(plan);m=plan['plant'].mj;data=mujoco.MjData(m)
    target=plan['poses'][0].copy();data.qpos[:]=target
    velocity=np.zeros(m.nv);velocity[plan['vi'][1]]=.1
    acceleration=np.zeros(m.nv);acceleration[plan['vi'][1]]=-.2
    data.qvel[:]=velocity;before=(data.qpos.copy(),data.qvel.copy())
    fsm.errors={0:dict(axial_m=.12)}
    fsm.begin_braking(data,target,velocity,acceleration,2.)
    q,v,a=fsm.interpolate(2.)
    np.testing.assert_allclose(q,target,atol=1e-12)
    np.testing.assert_allclose(v,velocity,atol=1e-12)
    np.testing.assert_allclose(a,acceleration,atol=1e-12)
    q,v,a=fsm.interpolate(2.+fsm.duration+.1)
    np.testing.assert_allclose(q,fsm.end,atol=1e-12)
    np.testing.assert_allclose(v,0.,atol=1e-12)
    np.testing.assert_allclose(a,0.,atol=1e-12)
    np.testing.assert_array_equal(data.qpos,before[0])
    np.testing.assert_array_equal(data.qvel,before[1])
