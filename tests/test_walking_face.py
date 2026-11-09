"""Ground-facing feet must be checked in geometry and actual contact evidence."""
import pytest
import numpy as np
import mujoco
import pinocchio as pin
from amsrr_det.model import build_model,PORTS
from amsrr_det.grasp_geometry import grasp_frame
from amsrr_det.walking import foot_link,WalkSpec


def test_all_dock_face_normals_are_positive_link_x():
    model,meta=build_model();d=model.createData()
    pin.framesForwardKinematics(model,d,pin.neutral(model))
    for port in PORTS:
        frame=d.oMf[meta['frames'][foot_link(0,port)]].inverse()*d.oMf[meta['frames'][grasp_frame(0,port)]]
        np.testing.assert_allclose(frame.rotation[:,0],[1,0,0],atol=1e-12)


@pytest.mark.parametrize("port,required",[("pitch1",True),("yaw1",False)])
def test_periodic_screen_rejects_tilt_even_without_collisions(port,required):
    from amsrr_det.walking_plan import plant_for
    from amsrr_det.walking_gait import validate_cycle
    p=plant_for(dict(modules=1,edges=[]));m=p.mj;spec=WalkSpec(modules=3,speed=.22)
    bid=m.body(foot_link(0,port)).id
    q=m.qpos0.copy();q[2]=3.
    d=mujoco.MjData(m);d.qpos[:]=q;mujoco.mj_kinematics(m,d)
    R=d.xmat[bid].reshape(3,3);normal=R[:,0]
    axis=np.cross(normal,[0,0,-1]);angle=np.arccos(-normal[2])
    rot=pin.exp3(axis/np.linalg.norm(axis)*angle)
    mujoco.mju_mat2Quat(q[3:7],rot.ravel())
    def screen(tilt,tracked=False):
        poses=[];r=pin.exp3(np.array([tilt,0.,0.]))@rot
        for t in np.linspace(0,1,5):
            v=q.copy();v[0]+=.22*t;mujoco.mju_mat2Quat(v[3:7],r.ravel());poses.append(v)
        gait=dict(configurations=poses,period=1.,origin=q[:3],height=.03,feet=np.zeros((1,3)),errors=[0.],closure_error=0.,foot_bodies=[bid])
        if tracked:gait['foot_points']=np.zeros((1,3))
        return validate_cycle(p,gait,spec=spec)
    straight=screen(0.);tilted=screen(np.deg2rad(10))
    assert straight['maximum_face_angle_rad']<1e-6
    if required:
        assert abs(tilted['maximum_face_angle_rad']-np.deg2rad(10))<1e-7
        assert not tilted['passed']
    else:assert tilted['maximum_face_angle_rad']==0.
    # A transformed/re-timed curve must not inherit a stale zero error from
    # its source. The physical foot is several metres above its target here.
    misplaced=screen(0.,tracked=True)
    assert misplaced['dense_position_error']>2.
    assert not misplaced['passed']


@pytest.mark.parametrize("port,required",[("pitch1",True),("yaw1",False)])
def test_actual_loaded_contacts_reject_tilt_without_changing_contact_physics(port,required):
    from types import SimpleNamespace
    from amsrr_det.walking_plan import plant_for
    from amsrr_det.walking_audit import WalkingAudit
    from amsrr_det.walking import foot_material
    p=plant_for(dict(modules=1,edges=[]));m=p.mj;link=foot_link(0,port)
    bid,vertices=foot_material(p,[link])[0];floor=m.geom('floor').id
    # Isolate one real Dock against the unchanged floor/contact material.
    for g in range(m.ngeom):
        if g!=floor and m.geom_bodyid[g]!=bid:m.geom_contype[g]=m.geom_conaffinity[g]=0
    d=mujoco.MjData(m);mujoco.mj_kinematics(m,d);n=d.xmat[bid].reshape(3,3)[:,0]
    axis=np.cross(n,[0,0,-1]);base=pin.exp3(axis/np.linalg.norm(axis)*np.arccos(-n[2]))
    results=[]
    for tilt in [0.,np.deg2rad(10.)]:
        mujoco.mj_resetData(m,d);R=pin.exp3(np.array([tilt,0.,0.]))@base
        mujoco.mju_mat2Quat(d.qpos[3:7],R.ravel());mujoco.mj_kinematics(m,d)
        world=vertices@d.xmat[bid].reshape(3,3).T+d.xpos[bid]
        d.qpos[2]-=world[:,2].min()+.001;d.qvel[2]=-.1;mujoco.mj_forward(m,d)
        audit=WalkingAudit(p,[link],WalkSpec(),SimpleNamespace(period=1.));audit.physics(d)
        assert audit.last_load[0]>.02*audit.weight
        results.append(audit)
    assert results[0].face_angle_max<1e-7
    assert results[0].face_bad_impulse==0.
    if required:
        assert abs(results[1].face_angle_max-np.deg2rad(10))<1e-7
        assert results[1].face_bad_impulse>0.
    else:
        assert results[1].face_angle_max==0.
        assert results[1].face_bad_impulse==0.


def test_spline_velocity_bounds_cover_between_knot_extrema():
    from scipy.interpolate import CubicSpline
    from amsrr_det.walking_vectoring import spline_velocity_hull
    rng=np.random.default_rng(612)
    times=np.r_[0.,np.sort(rng.uniform(.02,.98,12))];values=rng.normal(size=len(times))
    controls=(spline_velocity_hull(times,1.)@values).reshape(-1,3)
    curve=CubicSpline(np.r_[times,1.],np.r_[values,values[0]],bc_type='periodic')
    for i,(a,b) in enumerate(zip(times,np.r_[times[1:],1.])):
        derivative=curve(np.linspace(a,b,101),1)
        assert derivative.min()>=controls[i].min()-1e-9
        assert derivative.max()<=controls[i].max()+1e-9
    assert np.max(abs(curve(np.linspace(0,1,1001),1)))>np.max(abs(np.diff(np.r_[values,values[0]])/np.diff(np.r_[times,1.])))


def test_trial_deadline_returns_without_truncating_result_pipe(monkeypatch):
    import time
    from amsrr_det import walking_search as search
    monkeypatch.setattr(search,'_planning_deadline',time.perf_counter()+.03)
    monkeypatch.setattr(search,'_trial_stage',lambda payload:time.sleep(.5))
    start=time.perf_counter();row=search.trial_stage((None,1,2,None,None))
    assert row['reason']=='planning_deadline' and row['index']==2
    assert time.perf_counter()-start<.2



def test_spline_position_bounds_cover_between_knot_overshoot():
    from scipy.interpolate import CubicSpline
    from amsrr_det.walking_vectoring import spline_position_hull
    times=np.array([0.,.1,.3,.7]);values=np.array([0.,1.,-.2,-.3])
    hull=(spline_position_hull(times,1.)@values).reshape(-1,4)
    curve=CubicSpline(np.r_[times,1.],np.r_[values,values[0]],bc_type='periodic')
    for i,(a,b) in enumerate(zip(times,np.r_[times[1:],1.])):
        samples=curve(np.linspace(a,b,101))
        assert samples.min()>=hull[i].min()-1e-12
        assert samples.max()<=hull[i].max()+1e-12
    assert curve(np.linspace(0,1,1001)).max()>values.max()+.01


def test_local_periodic_coordinates_preserve_curves_and_continuous_bounds():
    from scipy.interpolate import BSpline,CubicSpline
    from amsrr_det.walking_vectoring import local_spline_maps,spline_velocity_hull,spline_position_hull
    rng=np.random.default_rng(192)
    for count in [8,21,27]:
        times=np.r_[0.,np.sort(rng.uniform(.001,.999,count-1))]
        basis,velocity,position=local_spline_maps(times,1.)
        np.testing.assert_allclose(velocity.toarray(),(spline_velocity_hull(times,1.)@basis).toarray(),atol=2e-10)
        np.testing.assert_allclose(position.toarray(),(spline_position_hull(times,1.)@basis).toarray(),atol=2e-10)
        coefficients=rng.normal(size=(count,3));values=basis@coefficients
        local=BSpline(np.r_[times[-3:]-1.,times,times[:4]+1.],
                      np.vstack([coefficients,coefficients[:3]]),3)
        interpolated=CubicSpline(np.r_[times,1.],np.vstack([values,values[:1]]),bc_type='periodic')
        for derivative in [0,1,2]:
            np.testing.assert_allclose(local(np.linspace(0,1,193),derivative),
                interpolated(np.linspace(0,1,193),derivative),atol=1e-7,rtol=1e-10)
        assert basis.nnz<=4*count
        assert velocity.nnz<=12*count and position.nnz<=16*count



def test_ground_allocation_uses_emitted_gimbal_effort(monkeypatch):
    """Motor saturation must agree with the force that other actuators see."""
    import time
    from amsrr_det import walking_control as control
    from amsrr_det.walking_plan import plant_for
    p=plant_for(dict(modules=3,edges=[dict(a=0,port_a='pitch1',b=1,port_b='yaw1'),dict(a=0,port_a='pitch2',b=2,port_b='yaw2')]))
    q=p.mj.qpos0.copy();q[2]=2.
    c=control.GroundControl(p,q,[foot_link(1,'pitch1'),foot_link(2,'pitch2')])
    dr=c.drive;gi=np.flatnonzero(~dr.dock);v=np.zeros(p.mj.nv);v[dr.vi[gi]]=np.linspace(-1.,1.,len(gi))
    ref=q.copy();ref[dr.qi[gi]]=np.linspace(-.15,.15,len(gi))
    c.drive.observe(q,v);feet=c.positions(dr.d)
    # End the solve immediately to also exercise its feasible deadline output.
    u=c.update(q,v,q[:7],np.zeros(6),np.zeros(6),feet,np.zeros_like(feet),np.zeros_like(feet),np.zeros(2,bool),shape_reference=ref,deadline=time.perf_counter()-1.)
    physical=np.clip(dr.gain[gi]*u[gi]-dr.kp[gi]*q[dr.qi[gi]]-dr.kd[gi]*v[dr.vi[gi]],-dr.limit[gi],dr.limit[gi])
    np.testing.assert_allclose(c.previous[gi]*dr.limit[gi],physical,atol=1e-12)
    assert c.last['fallback']



def test_face_rule_uses_declared_dock_sex():
    from amsrr_det.walking import requires_face_alignment
    for port in PORTS:
        assert requires_face_alignment(port)==port.startswith('pitch')
        assert requires_face_alignment(foot_link(3,port))==port.startswith('pitch')
    with pytest.raises(ValueError):requires_face_alignment('unlabelled')


def test_ground_proposals_keep_real_lowest_material_and_finite_friction():
    from amsrr_det.walking_plan import plant_for
    from amsrr_det.walking_control import GroundControl
    from amsrr_det.walking_load import ground_contacts
    p=plant_for(dict(modules=1,edges=[]));q=p.mj.qpos0.copy();q[2]=1.
    c=GroundControl(p,q,[foot_link(0,'yaw1')]);v=np.zeros(p.mj.nv);v[5]=.3
    c.drive.observe(q,v);bid,vertices=c.material[0];d=c.drive.d;R=d.xmat[bid].reshape(3,3)
    lowest=np.min(vertices@R[2]+d.xpos[bid,2])
    for contact in ground_contacts(c,[True]):
        point=np.array(contact['point'])
        assert np.min(np.linalg.norm(vertices-point,axis=1))<1e-12
        assert (d.xpos[bid]+R@point)[2]<=lowest+.0005+1e-12
        assert contact['mu']==.8
        assert not contact.get('sliding',False)


def test_support_duration_weights_integrate_time_not_number_of_sets():
    from amsrr_det.walking_posture import support_durations
    from amsrr_det.walking_load import support_quadrature
    phase=np.array([0.,.5,0.,.5]);duration=support_durations(phase,.6)
    assert duration[(0,1,2,3)]==pytest.approx(.2)
    assert duration[(0,2)]==pytest.approx(.4)
    assert duration[(1,3)]==pytest.approx(.4)
    phase=np.array([0.,.31,.71]);duration=support_durations(phase,.73)
    times,weights=support_quadrature(1.,3,.73,phase)
    expected={}
    for t,w in zip(times,weights):
        active=tuple(np.flatnonzero((t+phase)%1.<.73))
        expected[active]=expected.get(active,0.)+w
    assert duration==pytest.approx(expected)
    assert sum(duration.values())==pytest.approx(1.)


def test_removed_periodic_hull_rows_are_duplicates_or_node_bounds():
    from amsrr_det.walking_vectoring import local_spline_maps
    rng=np.random.default_rng(573)
    for n in [8,21]:
        times=np.r_[0.,np.sort(rng.uniform(.001,.999,n-1))]
        values,velocity,position=local_spline_maps(times,1.)
        velocity=velocity.toarray().reshape(n,3,n)
        position=position.toarray().reshape(n,4,n)
        np.testing.assert_allclose(velocity[:,2],np.roll(velocity[:,0],-1,axis=0),atol=1e-10)
        np.testing.assert_allclose(position[:,0],values.toarray(),atol=1e-12)
        np.testing.assert_allclose(position[:,3],np.roll(values.toarray(),-1,axis=0),atol=1e-12)


def test_phase_workers_keep_order_and_propagate_rejection():
    import os
    from amsrr_det.walking_screen import _map_rows
    cpus=set(sorted(os.sched_getaffinity(0))[:4])
    assert _map_rows(lambda x:x*x,[4,1,7,3],cpus)==[16,1,49,9]
    def fail(x):
        if x==7:raise ValueError('physical_check_failed')
        return x
    with pytest.raises(ValueError,match='physical_check_failed'):
        _map_rows(fail,[4,1,7,3],cpus)


def test_joint_memory_is_ground_only_and_does_not_reuse_observation_time():
    import time
    from amsrr_det.walking_plan import plant_for
    from amsrr_det.walking_control import GroundControl
    p=plant_for(dict(modules=1,edges=[]));q=p.mj.qpos0.copy();q[2]=1.
    c=GroundControl(p,q,[foot_link(0,'yaw1')]);dr=c.drive
    reference=q.copy();reference[dr.qi]=.2
    feet=c.positions(dr.d);zero=np.zeros(p.mj.nv)
    def update(t,flight):
        return c.update(q,zero,q[:7],zero[:6],zero[:6],feet,
            np.zeros_like(feet),np.zeros_like(feet),np.array([False]),shape_reference=reference,
            observation_time=t,flight_mode=flight,deadline=time.perf_counter()-1.)
    update(0.,True);np.testing.assert_array_equal(c.joint_bias,0.)
    update(.01,False);memory=c.joint_bias.copy()
    assert np.all(memory[dr.dock]>0)
    np.testing.assert_array_equal(memory[~dr.dock],0.)
    update(.01,False);np.testing.assert_array_equal(c.joint_bias,memory)
    update(.02,True);np.testing.assert_array_equal(c.joint_bias,memory)
    c.joint_bias[dr.dock]=.0999;u=update(.07,False)
    assert np.max(abs(c.joint_bias))<=.1
    assert np.isfinite(u).all() and c.last['fallback']
