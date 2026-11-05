import numpy as np
import pytest
import mujoco
from amsrr_det.walking import WalkSpec
from amsrr_det.walking_gait import foot_targets
from amsrr_det.mujoco_model import make_plant
from build.relative_task_control import RobotDrive


@pytest.mark.parametrize('failure_stage',['prepare','finish'])
def test_posture_repair_restarts_from_original_structure(monkeypatch,failure_stage):
    from types import SimpleNamespace
    import amsrr_det.walking_plan as plan
    original=dict(q=np.array([1.,2.]),_generation_s=.1)
    attempts=[]
    def prepare(spec,rng,record,*,support_weight):
        np.testing.assert_array_equal(record['q'],[1.,2.])
        attempts.append(support_weight);record['q'][:]=-99
        if support_weight==5. and failure_stage=='prepare':raise ValueError('geometry')
        return dict(weight=support_weight,cost=.4)
    def finish(spec,record):
        if record['weight']==5.:raise ValueError('moving_load')
        return record
    monkeypatch.setattr(plan,'prepare_candidate',prepare)
    monkeypatch.setattr(plan,'finish_candidate',finish)
    result=plan.candidate(SimpleNamespace(modules=4),None,original)
    assert attempts==[5.,20.] and result['weight']==20.
    assert len(result['proposal_attempts'])==1
    np.testing.assert_array_equal(original['q'],[1.,2.])


def test_completed_trial_keeps_first_fully_screened_posture(monkeypatch):
    from types import SimpleNamespace
    import amsrr_det.walking_plan as plan
    attempts=[]
    def prepare(spec,rng,record,*,support_weight):
        attempts.append(support_weight);return dict(cost=.3)
    monkeypatch.setattr(plan,'prepare_candidate',prepare)
    monkeypatch.setattr(plan,'finish_candidate',lambda spec,record:record)
    result=plan.candidate(SimpleNamespace(modules=3),None,dict(_generation_s=0.))
    assert attempts==[5.] and result['proposal_attempts']==[]


@pytest.mark.parametrize('curvature',[0.,.5,-.7])
def test_requested_speed_and_curvature_are_retained(curvature):
    task=WalkSpec(modules=7,speed=.28,curvature=curvature)
    t=np.linspace(0,30,101);p,yaw,v,a=task.path(t)
    np.testing.assert_allclose(np.linalg.norm(v,axis=1),.28)
    np.testing.assert_allclose(np.linalg.norm(a,axis=1),.28**2*abs(curvature),atol=1e-15)
    h=1e-5
    np.testing.assert_allclose((task.path(t+h)[0]-task.path(t-h)[0])/(2*h),v,atol=1e-10)
    assert task.modules==7 and task.duration==pytest.approx(5/.28)


@pytest.mark.parametrize('kwargs',[dict(modules=0),dict(modules=9),dict(modules=2.5),dict(speed=0),dict(speed=float('nan')),dict(distance=0.),dict(speed=.19)])
def test_invalid_task_is_not_silently_repaired(kwargs):
    with pytest.raises(ValueError):WalkSpec(**kwargs)


@pytest.mark.parametrize('count',[2,3,4,5,6,7,8])
def test_stance_stays_fixed_and_internal_step_is_periodic(count):
    nominal=np.c_[np.zeros(count),np.linspace(-.5,.5,count),np.zeros(count)]
    for now in np.linspace(0,2,101):
        p,v,a,stance=foot_targets(now,1.,.05,nominal)
        np.testing.assert_array_equal(v[stance],0.)
        np.testing.assert_array_equal(a[stance],0.)
        p1,v1,a1,s1=foot_targets(now+1,1.,.05,nominal)
        np.testing.assert_allclose(p1-p,np.tile([.05,0,0],(count,1)),atol=1e-15)
        np.testing.assert_allclose(v1,v,atol=1e-14)
        assert np.count_nonzero(stance)>=int(np.floor(.62*count))
        assert np.min(p[:,2])>=0


def test_shared_drive_has_no_fictitious_object_for_ground_task():
    plant=make_plant(meshes=False,floor=True)
    drive=RobotDrive(plant)
    assert drive.oi is None and drive.ov is None
    np.testing.assert_array_equal(drive.robot,np.arange(plant.mj.nv))
    data=mujoco.MjData(plant.mj)
    drive.observe(data.qpos,data.qvel)
    assert np.linalg.eigvalsh(drive.mass).min()>0
    assert drive.B.shape==(plant.mj.nv,plant.mj.nu)


@pytest.mark.parametrize('curvature',[-.5,.5])
def test_curved_stance_stays_fixed_and_cycle_rotates_as_a_whole(curvature):
    from amsrr_det.walking import rotation_z
    from amsrr_det.walking_gait import foot_headings
    origin=np.array([.3,-.2,.15]);feet=np.array([[.8,.2,0.],[-.2,-.6,0.]])
    task=WalkSpec(speed=.25,curvature=curvature);period=1.3
    delta,yaw,*_=task.path(period);R=rotation_z(yaw)
    for now in np.linspace(0,period,31):
        p,v,a,stance=foot_targets(now,period,task.speed,feet,curvature=curvature,origin=origin)
        nextp,*_=foot_targets(now+period,period,task.speed,feet,curvature=curvature,origin=origin)
        np.testing.assert_allclose(nextp,origin+(p-origin)@R.T+np.r_[delta,0.],atol=1e-14)
        np.testing.assert_array_equal(v[stance],0.)
        h=1e-7
        rate=(foot_headings(now+h,period,task.speed,2,curvature)-foot_headings(now-h,period,task.speed,2,curvature))/(2*h)
        np.testing.assert_allclose(rate[stance],0.,atol=1e-8)


def test_generalized_load_override_matches_existing_gravity_balance():
    import pinocchio as pin
    from amsrr_det.model import build_model
    from amsrr_det.equilibrium import static_balance
    model,meta=build_model();q=pin.neutral(model)
    gravity=pin.computeGeneralizedGravity(model,model.createData(),q)
    baseline=static_balance(model,meta,q)
    explicit=static_balance(model,meta,q,generalized_load=gravity)
    assert explicit['status']==baseline['status']
    np.testing.assert_allclose(explicit['thrust'],baseline['thrust'],atol=1e-10)
    with pytest.raises(ValueError):static_balance(model,meta,q,generalized_load=np.zeros(3))
    with pytest.raises(ValueError):static_balance(model,meta,q,generalized_load=gravity,base_acceleration=np.zeros(6))


def test_reused_load_screen_matches_fresh_workspaces_after_pose_and_load_changes():
    import pinocchio as pin
    from amsrr_det.model import build_model
    from amsrr_det.equilibrium import static_balance
    model,meta=build_model();workspace={};compared=0
    for angle in [0.,.15,0.]:
        q=pin.neutral(model);q[meta['actuators'][0]['q']]=angle
        gravity=pin.computeGeneralizedGravity(model,model.createData(),q)
        for axis in [0,2,meta['actuators'][1]['v']]:
            load=gravity.copy();load[axis]+=.01
            common=dict(generalized_load=load,fast_lp=True)
            fresh=static_balance(model,meta,q,**common)
            reused=static_balance(model,meta,q,workspace=workspace,**common)
            assert reused['status']==fresh['status']
            assert ('thrust' in reused)==('thrust' in fresh)
            if 'thrust' not in fresh:continue
            compared+=1
            np.testing.assert_allclose(reused['thrust'],fresh['thrust'],atol=1e-10)
            assert reused['minimum_peak_effort_ratio']==pytest.approx(fresh['minimum_peak_effort_ratio'],abs=1e-10)
    assert compared>=3


def test_swing_derivatives_match_motion_and_end_without_acceleration_jump():
    feet=np.array([[.3,.2,0.],[-.3,-.2,0.]])
    h=1e-6
    for t in [.1,.25,.4,.56,.7,.9,1.]:
        p,v,a,_=foot_targets(t,1.,.04,feet)
        before=foot_targets(t-h,1.,.04,feet);after=foot_targets(t+h,1.,.04,feet)
        np.testing.assert_allclose((after[0]-before[0])/(2*h),v,atol=1e-8)
        np.testing.assert_allclose((after[1]-before[1])/(2*h),a,atol=5e-5)


@pytest.mark.parametrize('curvature',[0.,.5])
def test_cycle_reference_does_not_modify_saved_motion_or_jump_at_wrap(curvature):
    from amsrr_det.walking_cycle import CycleReference
    from amsrr_det.walking import rotation_z
    p=make_plant(meshes=False,floor=True);d=mujoco.MjData(p.mj)
    spec=WalkSpec(speed=.24,curvature=curvature);poses=[]
    for t in np.linspace(0,1,25):
        q=d.qpos.copy();xy,yaw,*_=spec.path(t);q[:2]+=xy
        mujoco.mju_mat2Quat(q[3:7],rotation_z(yaw).ravel());poses.append(q)
    poses=np.array(poses);original=poses.copy()
    gait=dict(configurations=poses,period=1.,origin=d.qpos[:3].copy(),height=.015,feet=np.zeros((2,3)))
    a=CycleReference(p,spec,gait);b=CycleReference(p,spec,gait)
    np.testing.assert_array_equal(poses,original)
    for t in [0.,.99999,1.,1.00001,29.5]:
        np.testing.assert_array_equal(a.pose(t),b.pose(t))
        q,v,acc=a.rates(t);xy,yaw,velocity,acceleration=spec.path(t)
        np.testing.assert_allclose(q[:2],xy,atol=1e-12)
        np.testing.assert_allclose(v[:2],velocity,atol=1e-8)
        np.testing.assert_allclose(acc[:2],acceleration,atol=1e-7)


def test_remaining_foot_assignment_does_not_reuse_the_same_port():
    from amsrr_det.task_structure import _remaining_contact_cost,part_frames
    pads=part_frames()[3];root=np.eye(4)
    point=pads[0,2,:3,3];normal=pads[0,2,:3,0]
    # One remaining free port can cover one role, never two simultaneous feet.
    occupied={(0,1),(0,2),(0,3)}
    assert _remaining_contact_cost([root],occupied,point[None],normal[None],None)<1e-12
    assert _remaining_contact_cost([root],occupied,np.array([point,point]),np.array([normal,normal]),None)>=100.


def test_fixed_count_selection_does_not_claim_evidence_for_unfinished_work():
    from amsrr_det.structure_stopping import fixed_count_evidence
    values=[float('inf')]*59;values[12]=.2
    proof=fixed_count_evidence(6,values,.2,protocol_id='frozen-test')
    assert proof['complete'] and proof['one_more_trial_improvement_probability_upper']<.05
    assert proof['modules']==6 and not proof['undiscovered_structure_existence_probability_claimed']
    more=fixed_count_evidence(2,values*4,.2,protocol_id='more-fixed-trials')
    assert more['completed_trials']==236 and more['one_more_trial_improvement_probability_upper']<.013
    assert not fixed_count_evidence(2,values+[None],.2,protocol_id='unfinished-extension')['complete']
    values[58]=None
    proof=fixed_count_evidence(6,values,.2,protocol_id='frozen-test')
    assert not proof['complete'] and proof['one_more_trial_improvement_probability_upper'] is None
    with pytest.raises(ValueError):fixed_count_evidence(6,[.1]*59,.2,protocol_id='frozen-test')


def test_correct_travel_without_ground_load_is_not_walking():
    import json
    from types import SimpleNamespace
    from amsrr_det.walking_audit import WalkingAudit
    from amsrr_det.mujoco_model import CONTACT_CACHE
    p=make_plant(mesh_cache=CONTACT_CACHE,floor=True,timestep=.01);m=p.mj;d=mujoco.MjData(m)
    d.qpos[2]=1.;mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
    spec=WalkSpec(modules=1,speed=.2);initial=d.qpos.copy();origin=d.subtree_com[0].copy()
    def pose(t):
        q=initial.copy();q[0]+=.2*t;return q
    ref=SimpleNamespace(period=1.,origin=origin,pose=pose)
    audit=WalkingAudit(p,['m0/pitch_dock_mech1','m0/pitch_dock_mech2'],spec,ref)
    # Deliberately perfect prescribed kinematics with no physical support.
    for k in range(1,3001):
        d.time=k*.01;d.qpos[:]=pose(d.time);d.qvel[0]=.2
        audit.physics(d);audit.observe(d)
    result=audit.result();json.dumps(result,allow_nan=False)
    assert result['checks']['distance'] and result['checks']['path'] and result['checks']['mean_speed']
    assert not result['task_success'] and not result['checks']['ground_support'] and not result['checks']['feet']


def test_sparse_operator_specialization_preserves_matrix_and_adjoint():
    from scipy.sparse import random as sparse_random
    from scipy.optimize._lsq.common import right_multiplied_operator,regularized_lsq_operator
    from amsrr_det.sparse_least_squares import _right_scaled,_regularized
    rng=np.random.default_rng(14);A=sparse_random(90,30,density=.15,random_state=rng).tocsr()
    scales=rng.uniform(.1,2.,30);diagonal=rng.uniform(.01,.2,30)
    old=regularized_lsq_operator(right_multiplied_operator(A,scales),diagonal)
    new=_regularized(_right_scaled(A,scales),diagonal)
    x=rng.normal(size=30);y=rng.normal(size=120)
    np.testing.assert_allclose(new@x,old@x,atol=1e-13)
    np.testing.assert_allclose(new.T@y,old.T@y,atol=1e-13)


def test_direct_sparse_step_matches_bounded_convex_solution_without_global_patch():
    from scipy.sparse import random as sparse_random
    from scipy.optimize import least_squares
    from scipy.optimize._lsq import trf
    from amsrr_det.sparse_least_squares import least_squares as sparse_solve
    rng=np.random.default_rng(5);A=sparse_random(150,40,density=.2,random_state=rng).tocsr();b=rng.normal(size=150)
    original=trf.regularized_lsq_operator
    kwargs=dict(jac=lambda x:A,bounds=(0,1),tr_solver='lsmr',gtol=1e-10,ftol=1e-10,xtol=1e-10)
    old=least_squares(lambda x:A@x-b,np.full(40,.1),**kwargs)
    new=sparse_solve(lambda x:A@x-b,np.full(40,.1),direct=True,**kwargs)
    assert trf.regularized_lsq_operator is original
    assert 0<=new.x.min() and new.x.max()<=1.
    from scipy.optimize import lsq_linear
    reference=lsq_linear(A.toarray(),b,bounds=(0,1),method='bvls',tol=1e-12)
    assert new.cost<=old.cost+1e-9
    assert abs(new.cost-reference.cost)<1e-9
    np.testing.assert_allclose(new.x,reference.x,atol=1e-7)


def test_ground_friction_and_support_corners_rotate_with_the_assembly():
    from amsrr_det.walking import ground_support,rotation_z
    from scipy.spatial.transform import Rotation
    R=Rotation.from_euler('xyz',[.17,-.38,.49]).as_matrix()
    vertices=np.array([[-.04,-.07,0],[.05,-.06,0],[.04,.06,0],[-.03,.08,0],[0,0,.01]])
    basis,ids=ground_support(R,vertices)
    np.testing.assert_allclose(basis.T@basis,np.eye(3),atol=1e-14)
    for angle in [.31,.8,1.62,2.8]:
        Z=rotation_z(angle);rotated,other=ground_support(Z@R,vertices@Z.T+np.array([1.,2.,0.]))
        np.testing.assert_allclose(rotated,Z@basis,atol=1e-14)
        np.testing.assert_array_equal(other,ids)


def test_periodic_banded_normal_solve_matches_dense_solution():
    from scipy.sparse import diags
    from amsrr_det.sparse_least_squares import _normal_solve
    # Include the cycle wrap, which is far from the diagonal before ordering.
    n=320;rng=np.random.default_rng(71)
    A=diags([-.3*np.ones(n-1),2+rng.random(n),-.3*np.ones(n-1)],[-1,0,1],format='lil')
    A[0,-1]=A[-1,0]=-.3;A=A.tocsc();normal=(A.T@A).tocsc();rhs=rng.normal(size=n)
    actual=_normal_solve(normal,rhs)
    np.testing.assert_allclose(actual,np.linalg.solve(normal.toarray(),rhs),rtol=1e-11,atol=1e-12)
    np.testing.assert_allclose(normal@actual,rhs,atol=1e-12)


def test_shared_posture_force_elimination_matches_full_equations():
    from scipy.sparse import csr_matrix
    from scipy.optimize import lsq_linear
    from amsrr_det.sparse_least_squares import _arrow_solve,least_squares
    rng=np.random.default_rng(936);shared=9;blocks=[shared,22,37,45]
    rows=[]
    for start,end in zip(blocks[:-1],blocks[1:]):
        row=np.zeros((30,blocks[-1]));row[:,:shared]=rng.normal(size=(30,shared))
        row[:,start:end]=rng.normal(size=(30,end-start));rows.append(row)
    A=np.vstack([*rows,.1*np.eye(blocks[-1])]);b=rng.normal(size=len(A))
    normal=csr_matrix(A.T@A);rhs=A.T@b
    np.testing.assert_allclose(_arrow_solve(normal,rhs,blocks),np.linalg.solve(A.T@A,rhs),rtol=1e-11,atol=1e-12)
    sparse=csr_matrix(A)
    actual=least_squares(lambda x:A@x-b,np.full(A.shape[1],.1),jac=lambda x:sparse,
        direct=True,blocks=blocks,bounds=(0.,1.),tr_solver='lsmr',gtol=1e-10,ftol=1e-10,xtol=1e-10)
    reference=lsq_linear(A,b,bounds=(0.,1.),method='bvls',tol=1e-12)
    np.testing.assert_allclose(actual.x,reference.x,atol=1e-7)
    assert abs(actual.cost-reference.cost)<1e-9


def test_ground_acquisition_bias_never_supplies_force_from_a_separated_foot():
    from amsrr_det.walking import foot_link
    from amsrr_det.walking_control import GroundControl
    from amsrr_det.walking_plan import plant_for
    plant=plant_for(dict(modules=1,edges=[]));m=plant.mj;q=m.qpos0.copy();q[2]=1.
    control=GroundControl(plant,q,[foot_link(0,'pitch1'),foot_link(0,'pitch2')])
    dr=control.drive;dr.observe(q,np.zeros(m.nv));feet=control.positions(dr.d);feet[:,2]=0.
    for stamp,stance in [(0.,np.ones(2,bool)),(.05,np.zeros(2,bool)),(.1,np.ones(2,bool))]:
        command=control.update(q,np.zeros(m.nv),q[:7],np.zeros(6),np.zeros(6),feet,
            np.zeros_like(feet),np.zeros_like(feet),stance,observation_time=stamp)
        assert control.last['active_feet']==[]
        np.testing.assert_array_equal(control.last['forces'],np.zeros((2,3)))
        assert np.max(control.contact_bias)<=.03
        efforts=command[:dr.nj]*dr.gain[:dr.nj]-dr.kp*q[dr.qi]
        assert np.max(abs(efforts)/dr.limit[:dr.nj])<=1.000001
        assert np.all(command[dr.nj:]>=0) and np.all(command[dr.nj:]<=dr.limit[dr.nj:])


def test_tracking_integrals_use_observation_time_and_release_swing_correction():
    from amsrr_det.walking import foot_link
    from amsrr_det.walking_control import GroundControl
    from amsrr_det.walking_plan import plant_for
    plant=plant_for(dict(modules=1,edges=[]));m=plant.mj;q=m.qpos0.copy();q[2]=1.
    c=GroundControl(plant,q,[foot_link(0,'pitch1'),foot_link(0,'pitch2')])
    feet=c.positions(c.drive.d);feet[:,2]+=1.
    com=c.drive.d.subtree_com[0,:2].copy()+1.
    body=q[:7].copy();body[2]+=.2
    body[3:]=[np.cos(.2),np.sin(.2),0.,0.]
    def update(t,stance):
        c.update(q,np.zeros(m.nv),body,np.zeros(6),np.zeros(6),feet,
            np.zeros_like(feet),np.zeros_like(feet),np.array(stance),observation_time=t,
            com_reference=com,com_velocity=np.zeros(2),com_acceleration=np.zeros(2))
        assert c.last['active_feet']==[]  # Integration cannot invent contact.
    update(0.,[False,False]);lift=c.lift_bias.copy();path=c.path_bias.copy();bias=c.attitude_bias.copy()
    assert np.all(lift>0) and np.all(path>0)
    assert bias[0]>0
    np.testing.assert_array_equal(bias[1:],[0.,0.])
    update(0.,[False,False])
    np.testing.assert_array_equal(c.lift_bias,lift)
    np.testing.assert_array_equal(c.path_bias,path)
    np.testing.assert_array_equal(c.attitude_bias,bias)
    for t in [.05,.1,.15]:update(t,[False,False])
    assert np.max(c.lift_bias)<=.015 and np.max(abs(c.path_bias))<=.05
    assert np.all(abs(c.attitude_bias)<=.15)
    update(.2,[True,False]);assert c.lift_bias[0]==0 and c.lift_bias[1]>0
    update(.25,[True,True]);np.testing.assert_array_equal(c.lift_bias,[0.,0.])


@pytest.mark.parametrize('count',[2,3,4,5,6,7,8])
def test_load_samples_include_both_contact_sets_at_every_support_change(count):
    from amsrr_det.walking_load import support_switch_times
    period=1.31;times=support_switch_times(period,count);feet=np.zeros((count,3))
    assert np.all(times>=0) and np.all(times<period)
    for i in range(count):
        for phase in [(-i/count)%1.,(.62-i/count)%1.]:
            before=((phase-1e-6)%1.)*period;after=((phase+1e-6)%1.)*period
            assert min(abs(times-before))<1e-12 and min(abs(times-after))<1e-12
            a=foot_targets(before,period,.06,feet)[3]
            b=foot_targets(after,period,.06,feet)[3]
            assert a[i]!=b[i]


def test_com_force_identity_matches_geometric_derivatives():
    plant=make_plant(meshes=False,floor=True);m=plant.mj
    drive=RobotDrive(plant);q=m.qpos0.copy();q[2]=1.
    delta=np.linspace(-.1,.1,m.nv);mujoco.mj_integratePos(m,q,delta,1.)
    v=np.linspace(-.3,.4,m.nv);drive.observe(q,v)
    mass=m.body_mass.sum();geometric=np.zeros((3,m.nv))
    mujoco.mj_jacSubtreeCom(m,drive.d,geometric,0)
    np.testing.assert_allclose(drive.mass[:3]/mass,geometric,atol=1e-12)
    expected_bias=drive.d.qfrc_bias[:3]/mass+m.opt.gravity
    derivatives=[];h=1e-5
    for sign in [-1,1]:
        pose=q.copy();mujoco.mj_integratePos(m,pose,v,sign*h)
        d=mujoco.MjData(m);d.qpos[:]=pose
        mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
        J=np.zeros_like(geometric);mujoco.mj_jacSubtreeCom(m,d,J,0);derivatives.append(J@v)
    np.testing.assert_allclose((derivatives[1]-derivatives[0])/(2*h),expected_bias,atol=1e-8)


def test_support_pose_gradient_matches_changes_in_real_geometry(monkeypatch):
    from types import SimpleNamespace
    import amsrr_det.walking_posture as posture
    from amsrr_det.walking_plan import plant_for
    from amsrr_det.walking_control import GroundControl
    from amsrr_det.walking import foot_link
    p=plant_for(dict(modules=1,edges=[]));q=p.mj.qpos0.copy();q[2]=1.
    c=GroundControl(p,q,[foot_link(0,'pitch1'),foot_link(0,'yaw2')])
    def check(fun,x,jac,bounds,**kwargs):
        x=x.copy();x[:6]+=np.array([.02,-.01,.03,.07,-.09,.11])
        x[6:6+len(p.meta['actuators'])]+=.013*np.sin(np.arange(len(p.meta['actuators'])))
        # Move off the support-deficit hinge: its residual has no unique
        # derivative exactly at the initialized 60% normal load.
        x[6+len(p.meta['actuators']):]*=.9
        lo,hi=map(np.asarray,bounds);x=np.clip(x,lo+1e-4,hi-1e-4)
        analytic=jac(x).toarray();h=1e-6;numeric=[]
        for i in range(len(x)):
            step=np.zeros(len(x));step[i]=h
            numeric.append((fun(x+step)-fun(x-step))/(2*h))
        np.testing.assert_allclose(analytic,np.array(numeric).T,atol=3e-5,rtol=3e-5)
        return SimpleNamespace(x=x,nfev=1,cost=.5*fun(x)@fun(x))
    monkeypatch.setattr(posture,'least_squares',check)
    posture.refine_support_pose(p,c,q,iterations=1)


def test_paired_four_foot_phases_have_only_the_selected_pairs_or_all_feet():
    from amsrr_det.walking_load import support_switch_times
    phases=np.array([0.,.5,0.,.5]);period=1.6;feet=np.zeros((4,3))
    observed=set()
    times=np.r_[np.linspace(0,period,101),support_switch_times(period,4,phases=phases)]
    for t in times:
        stance=foot_targets(t,period,.25,feet,phases=phases)[3]
        observed.add(tuple(np.flatnonzero(stance)))
    assert observed=={(0,2),(1,3),(0,1,2,3)}


def test_compiled_cycle_geometry_jacobian_matches_independent_differences():
    from amsrr_det.walking_native import CycleGeometry
    from amsrr_det.walking_plan import plant_for
    from amsrr_det.walking_control import GroundControl
    from amsrr_det.walking_gait import CycleKinematics
    p=plant_for(dict(modules=1,edges=[]));q=p.mj.qpos0.copy();q[2]=.24
    c=GroundControl(p,q,['m0/pitch_dock_mech1','m0/pitch_dock_mech2'])
    k=CycleKinematics(p,c,q);geometry=CycleGeometry(k);x=np.random.default_rng(89).normal(0,.03,len(k.indices))
    points=k.points()[0];rotations=k.rotations;support=np.array([True,False]);com=np.array([.03,-.02])
    r,J=geometry.evaluate(x,q,points,rotations,support,com)
    h=1e-6;fd=np.column_stack([(geometry.evaluate(x+h*np.eye(len(x))[i],q,points,rotations,support,com)[0]-
        geometry.evaluate(x-h*np.eye(len(x))[i],q,points,rotations,support,com)[0])/(2*h) for i in range(len(x))])
    np.testing.assert_allclose(J,fd,atol=2e-6,rtol=1e-6)
    assert np.isfinite(r).all()


def test_foot_tracking_points_remain_fixed_when_initial_pose_changes():
    from amsrr_det.walking_plan import plant_for
    from amsrr_det.walking_control import GroundControl
    p=plant_for(dict(modules=1,edges=[]));m=p.mj;q=m.qpos0.copy();q[2]=.4
    links=['m0/pitch_dock_mech1','m0/pitch_dock_mech2']
    planned=GroundControl(p,q,links)
    tilted=q.copy();rotation=np.zeros(m.nv);rotation[3:6]=[.12,.21,-.1]
    mujoco.mj_integratePos(m,tilted,rotation,1.)
    regenerated=GroundControl(p,tilted,links)
    assert np.max(np.abs(regenerated.foot_points-planned.foot_points))>.005
    execution=GroundControl(p,tilted,links,foot_points=planned.foot_points.tolist())
    planned.drive.observe(tilted,np.zeros(m.nv))
    np.testing.assert_allclose(execution.positions(execution.drive.d),planned.positions(planned.drive.d),atol=1e-14)
    # Storage is independent of the caller's mutable array.
    execution.foot_points[0,0]+=.01
    assert execution.foot_points[0,0]!=planned.foot_points[0,0]


def test_support_timing_requires_time_not_only_mean_capacity(monkeypatch):
    from types import SimpleNamespace
    import amsrr_det.walking_load as load
    plant=SimpleNamespace(meta={'mass':1/9.80665},mj=SimpleNamespace(nv=1),pin=None,
                          observation=lambda d:(None,None))
    controller=SimpleNamespace(drive=SimpleNamespace(d=None,observe=lambda q,v:None))
    monkeypatch.setattr(load,'support_phases',lambda *args:(np.array([0.,.5]),{}))
    monkeypatch.setattr(load,'ground_contacts',lambda control,mask:mask)
    def balance(model,meta,q,mask,**kwargs):
        return {'contacts':[{'normal_force':1. if sum(mask)==2 else .45}]}
    monkeypatch.setattr(load,'static_balance',balance)
    phase,report=load.support_timing(plant,controller,None)
    # At .62 duty mean capacity is .582, but double support lasts only .24.
    # Both feet must stay down for .60 of the cycle, requiring .80 duty.
    np.testing.assert_allclose(phase,[0.,.5])
    assert abs(report['duty']-.8)<1e-12
    assert abs(report['capacity_time_55']-.6)<1e-12
    assert abs(report['capacity_mean']-.78)<1e-12
    # An unhelpful initial direction is not a proof that the structure cannot
    # work. Subsequent motion/direction checks still get the original seed.
    monkeypatch.setattr(load,'static_balance',lambda *args,**kwargs:{'contacts':[]})
    _,fallback=load.support_timing(plant,controller,None)
    assert fallback['duty']==.62 and fallback['capacity_proposal_unavailable']


def test_motion_keeps_the_support_order_used_by_posture(monkeypatch):
    from amsrr_det.walking_control import GroundControl
    from amsrr_det.walking_gait import CycleKinematics
    import amsrr_det.walking_load as load
    from amsrr_det.mujoco_model import CONTACT_CACHE
    plant=make_plant(mesh_cache=CONTACT_CACHE,floor=True)
    pose=plant.mj.qpos0.copy();pose[2]=.6
    links=['m0/pitch_dock_mech1','m0/pitch_dock_mech2']
    controller=GroundControl(plant,pose,links)
    kin=CycleKinematics(plant,controller,pose)
    def unexpected(*args):
        raise AssertionError('must not replace a previously synthesized support order')
    monkeypatch.setattr(load,'support_phases',unexpected)
    monkeypatch.setattr(load,'support_timing',unexpected)
    monkeypatch.setattr(kin,'fit',lambda seed,*args,**kwargs:(seed.copy(),0.))
    phases=np.array([.1,.6]);duty=.71
    gait=kin.generate(WalkSpec(modules=3,speed=.22),samples=3,
                      fixed_duty=duty,fixed_phases=phases)
    np.testing.assert_array_equal(gait['phases'],phases)
    assert gait['duty']==duty


def test_compiled_foot_fit_matches_independent_linearized_motion():
    from amsrr_det.walking_plan import plant_for
    from amsrr_det.walking_control import GroundControl
    from amsrr_det.walking_gait import CycleKinematics
    p=plant_for(dict(modules=1,edges=[]));m=p.mj;q=m.qpos0.copy();q[2]=1.
    c=GroundControl(p,q,['m0/pitch_dock_mech1','m0/pitch_dock_mech2']);kin=CycleKinematics(p,c,q)
    d=kin.d;d.qpos[:]=q;mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d);mujoco.mj_collision(m,d)
    assert not any(x.dist<.002 for x in d.contact)
    points,lowest=kin.points();feet=points+np.array([.02,-.01,.005]);body=q[:7].copy();body[0]+=.01
    J=[];angular=[];jp=np.zeros((3,m.nv));jr=jp.copy()
    for i,((bid,_),point) in enumerate(zip(c.material,lowest)):
        center=d.xpos[bid]+d.xmat[bid].reshape(3,3)@c.foot_points[i]
        mujoco.mj_jac(m,d,jp,jr,center,bid);j=jp.copy();angular.extend(jr[:,kin.indices])
        mujoco.mj_jac(m,d,jp,None,point,bid);j[2]=jp[2];J.extend(j[:,kin.indices])
    weights=np.array([1.,.5,.25,.12,.12,.12]);ref=q.copy();ref[:7]=body;delta=np.empty(m.nv)
    mujoco.mj_differentiatePos(m,delta,1.,q,ref);nd=len(kin.indices)
    A=np.vstack([10*np.array(J),.2*np.array(angular),np.c_[np.diag(weights),np.zeros((6,nd-6))],.008*np.eye(nd)])
    b=np.r_[10*(feet-points).ravel(),np.zeros(3*len(feet)),weights*delta[:6],.008*delta[kin.indices]]
    step=np.linalg.lstsq(A,b,rcond=1e-8)[0];step*=min(1.,.15/max(abs(step).max(),1e-9))
    v=np.zeros(m.nv);v[kin.indices]=step;expected=q.copy();mujoco.mj_integratePos(m,expected,v,1.)
    expected[kin.qi]=np.clip(expected[kin.qi],kin.limits[:,0],kin.limits[:,1])
    actual,error=kin.fit(q,body,feet,iterations=1)
    np.testing.assert_allclose(actual,expected,atol=2e-10,rtol=2e-10)
    assert error<np.max(np.linalg.norm(feet-points,axis=1))


def test_periodic_screen_checks_interpolated_angles_and_speeds():
    from amsrr_det.walking_gait import validate_cycle
    from amsrr_det.walking_cycle import CycleReference
    p=make_plant(meshes=False,floor=True);m=p.mj;d=mujoco.MjData(m);spec=WalkSpec(speed=.22)
    actuator=next(a for a in p.meta['actuators'] if a['role']!='dock')
    joint=m.joint(actuator['name']);qi=joint.qposadr[0];vi=joint.dofadr[0]
    q=d.qpos.copy();q[2]=3.;poses=[]
    # Every saved knot respects its angle limit. The interpolating cubic
    # nevertheless overshoots between the two equal high knots.
    upper=joint.range[1];angles=[0,upper-.01,upper-.01,0,0]
    for t,angle in zip(np.linspace(0,4,5),angles):
        pose=q.copy();pose[0]+=.22*t;pose[qi]=angle;poses.append(pose)
    gait=dict(configurations=np.array(poses),period=4.,origin=q[:3].copy(),height=.03,feet=np.zeros((2,3)),errors=[0.],closure_error=0.)
    report=validate_cycle(p,gait,spec=spec)
    assert report['angle_excess_rad']>.01 and not report['passed']
    # Isolate rate overshoot: small-amplitude triangular knots have secant
    # speed below the configured limit, but cubic peak speed exceeds it.
    limit=actuator['speed'];gait['period']=.4
    for t,pose,angle in zip(np.linspace(0,.4,5),gait['configurations'],[0,.09*limit,0,-.09*limit,0]):
        pose[:3]=q[:3]+[.22*t,0,0];pose[qi]=angle
    secants=np.diff(gait['configurations'][:,qi])/.1
    assert np.max(abs(secants))<limit
    report=validate_cycle(p,gait,spec=spec)
    assert report['velocity_ratio']>1. and not report['passed']


def test_independent_foot_growth_does_not_share_a_nonroot_branch():
    from amsrr_det.task_structure import grow_to_contacts
    points=np.array([[.7,.5,0.],[.7,-.5,0.],[-.7,.5,0.],[-.7,-.5,0.]])
    root=np.eye(4);root[2,3]=.4
    edges,surfaces,_,_,trace=grow_to_contacts(7,points,np.tile([0.,0.,-1.],(4,1)),
        np.random.default_rng(17),root_pose=root,independent_branches=True,
        contact_forces=np.tile([0.,0.,20.],(4,1)),contact_strokes=np.tile([.3,0.,0.],(4,1)))
    parents={edge['b']:edge['a'] for edge in edges}
    paths=[]
    for module,_ in surfaces:
        path=set()
        while module:
            path.add(module);module=parents[module]
        paths.append(path)
    assert len(surfaces)==4
    assert all(not (paths[i]&paths[j]) for i in range(4) for j in range(i))
    assert len(edges)==6 and all(t['target_role'] in range(4) for t in trace)


def test_vector_inspection_never_accepts_before_complete_physics_check(monkeypatch):
    from types import SimpleNamespace
    import amsrr_det.walking_vectoring as vector
    monkeypatch.setattr(vector,'validate_cycle',lambda *a,**k:dict(passed=True))
    class Reference:
        period=1.
        def __init__(self,*args):pass
        def rates(self,t):return np.zeros(1),np.zeros(1),np.zeros(1)
        def feet(self,t):return None,None,None,np.array([True])
    monkeypatch.setattr(vector,'CycleReference',Reference)
    monkeypatch.setattr(vector,'motion_load',lambda *a:(np.zeros(1),np.zeros(1)))
    monkeypatch.setattr(vector,'ground_contacts',lambda *a:[])
    monkeypatch.setattr(vector,'static_balance',lambda *a,**k:dict(minimum_peak_effort_ratio=.5))
    plant=SimpleNamespace(pin=None,meta=None);control=SimpleNamespace(drive=None);calls=[]
    def screen(*args):calls.append(True);return dict(cost=.4)
    report,_,reason=vector.inspect_proposal(plant,control,None,{},screen,complete_check=False)
    assert report is None and not calls
    report,_,_=vector.inspect_proposal(plant,control,None,{},screen)
    assert report==dict(cost=.4) and calls==[True]


def test_force_certificate_is_checked_against_a_fresh_load_solve():
    import pinocchio as pin
    from amsrr_det.model import build_model
    from amsrr_det.equilibrium import static_balance,certify_load_changes
    from amsrr_det.grasp_geometry import aim_rotors_up
    model,meta=build_model();q=aim_rotors_up(model,meta,pin.neutral(model))
    gravity=pin.computeGeneralizedGravity(model,model.createData(),q).copy();workspace={}
    base=static_balance(model,meta,q,generalized_load=gravity,fast_lp=True,workspace=workspace)
    assert base['status']=='static_screen_pass'
    changes=np.zeros((3,model.nv));changes[:,2]=[.01,-.01,1000.]
    good,upper=certify_load_changes(workspace,changes)
    assert good[:2].all() and not good[2]
    for i in range(2):
        fresh=static_balance(model,meta,q,generalized_load=gravity+changes[i],fast_lp=True)
        assert fresh['minimum_peak_effort_ratio']<=upper[i]+1e-7
    changes[0,meta['actuators'][0]['v']]=.1
    assert certify_load_changes(workspace,changes) is None


def test_reused_simplex_checks_changed_loads_and_matrix_bounds():
    from amsrr_det.linear_program import reused_simplex_linprog,linprog
    reusable={};objective=np.array([1.,2.]);E=np.array([[1.,1.]]);A=np.array([[-1.,0.]])
    for total in [1.,2.,.25,1.]:
        kw=dict(A_eq=E,b_eq=np.array([total]),A_ub=A,b_ub=np.array([-.1]),bounds=[(0.,1.),(0.,None)])
        got=reused_simplex_linprog(objective,**kw,reusable=reusable,options={'maxiter':500})
        ref=linprog(objective,**kw)
        assert got.success and got.fun==pytest.approx(ref.fun,abs=1e-9)
        np.testing.assert_allclose(E@got.x,[total],atol=1e-9)
    assert len(reusable)==1


def test_coarse_load_screen_cannot_accept_an_unchecked_between_sample_peak(monkeypatch):
    from types import SimpleNamespace
    import amsrr_det.walking_vectoring as module
    plant=SimpleNamespace(pin=None,meta=None)
    controller=SimpleNamespace(drive=None)
    monkeypatch.setattr(module,'validate_cycle',lambda *a,**k:dict(passed=True))
    monkeypatch.setattr(module,'CycleReference',lambda *a:SimpleNamespace(period=1.,rates=lambda t:(t,None,None),feet=lambda t:(None,None,None,None)))
    monkeypatch.setattr(module,'motion_load',lambda p,d,q,v,a:(q,None))
    monkeypatch.setattr(module,'ground_contacts',lambda *a:[])
    for bad,expected_calls in [(3,32),(1,96),(None,96)]:
        visited=[];accepted=[]
        def balance(p,m,q,c,**kw):
            index=round(q*96);visited.append(index)
            return dict(minimum_peak_effort_ratio=1.1 if index==bad else .5)
        monkeypatch.setattr(module,'static_balance',balance)
        def screen(*a):
            assert a[-1].shape==(96,) and np.isfinite(a[-1]).all()
            assert np.max(a[-1])<=.9
            accepted.append(True);return dict(passed=True)
        report,cuts,diagnostic=module.inspect_proposal(plant,controller,None,{},screen)
        assert len(visited)==expected_calls
        assert bool(accepted)==(bad is None)
        assert (report is not None)==(bad is None)
        if bad is not None:assert bad/96 in cuts


def test_proposal_floor_filter_preserves_geometry_and_physical_contacts():
    import copy
    from amsrr_det.walking_plan import plant_for
    from amsrr_det.walking_control import GroundControl
    from amsrr_det.walking_gait import CycleKinematics
    from amsrr_det.walking_native import CycleGeometry
    p=plant_for(dict(modules=1,edges=[]));m=p.mj;q=m.qpos0.copy();q[2]=.01
    c=GroundControl(p,q,['m0/pitch_dock_mech1','m0/pitch_dock_mech2'])
    kin=CycleKinematics(p,c,q);filtered=CycleGeometry(kin);full=CycleGeometry(kin)
    full.geometry_model=copy.copy(m)
    targets=c.positions(c.drive.d);com=np.zeros(2)
    floor=m.geom('floor').id
    for tilt in [0.,.12,-.09]:
        x=np.zeros(filtered.nd);x[3:6]=[tilt,-.5*tilt,.3*tilt]
        a=filtered.evaluate(x,q,targets,kin.rotations,[True,False],com)
        b=full.evaluate(x,q,targets,kin.rotations,[True,False],com)
        for got,want in zip(a,b):np.testing.assert_allclose(got,want,atol=2e-10,rtol=2e-10)
    assert m.geom_contype[floor] and m.geom_conaffinity[floor]
    kin.d.qpos[:]=q;mujoco.mj_kinematics(m,kin.d);mujoco.mj_collision(m,kin.d)
    assert any(floor in contact.geom for contact in kin.d.contact)


def test_cached_foot_material_is_independent_and_tracks_model_replacement():
    import copy
    from amsrr_det.walking import foot_material
    from amsrr_det.walking_plan import plant_for
    p=plant_for(dict(modules=1,edges=[]));links=['m0/pitch_dock_mech1']
    first=foot_material(p,links);expected=first[0][1].copy();first[0][1][:]=100.
    np.testing.assert_array_equal(foot_material(p,links)[0][1],expected)
    p.mj=copy.copy(p.mj)
    geom=next(g for g in range(p.mj.ngeom) if (p.mj.geom(g).name or '').startswith(links[0]+'/'))
    p.mj.geom_pos[geom,0]+=.01
    assert not np.array_equal(foot_material(p,links)[0][1],expected)


def test_energy_cost_matches_planner_gradient_and_scales_with_weight():
    from amsrr_det.walking_control import energy_regularizer
    nj=2;limit=np.array([6.,12.,20.,20.]);x=np.array([.2,-.3,.4,.7]);weight=40.
    def cost(u):
        return 30000.*(np.sum((u[nj:]*limit[nj:])**1.5)/weight**1.5+.05*np.mean(u[:nj]**2))
    h=1e-6
    gradient=np.array([(cost(x+np.eye(4)[i]*h)-cost(x-np.eye(4)[i]*h))/(2*h) for i in range(4)])
    np.testing.assert_allclose(energy_regularizer(x,limit,nj,weight)*x,gradient,rtol=1e-8)
    np.testing.assert_allclose(energy_regularizer(x,limit*2,nj,weight*2),energy_regularizer(x,limit,nj,weight))
    assert np.isfinite(energy_regularizer(np.zeros(4),limit,nj,weight)).all()


def test_energy_allocation_favors_ground_support_without_changing_required_force():
    import time
    from amsrr_det.bounded_qp import bounded_qp
    from amsrr_det.walking_control import energy_regularizer
    # One joint/rotor plus a supporting contact. Both can supply upward force.
    A=np.array([[0.,1.,1.,0.,0.]])*1000.;target=np.array([1000.])
    C=np.zeros((5,13));C[:2,:2]=np.eye(2);C[:2,2:4]=-np.eye(2);C[2,4]=1.
    theta=np.arange(8)*np.pi/4+np.pi/8
    C[2,5:]=.8*np.cos(np.pi/8);C[3:,5:]=-np.array([np.cos(theta),np.sin(theta)])
    lower=np.array([-1.,0.]);upper=np.ones(2);bound=np.r_[lower,-upper,np.zeros(9)]
    reg=np.r_[energy_regularizer(np.array([0.,.5]),np.ones(2),1,1.),np.ones(3)*1e-5]
    x,expired,_=bounded_qp(A.T@A+np.diag(reg),A.T@target,C,bound,lower,upper,np.array([0.,.5,.5,0.,0.]),deadline=time.perf_counter()+1.,friction=.8)
    assert not expired
    assert abs(x[1]+x[2]-1.)<1e-6 and x[1]<1e-5 and x[2]>.999
def test_energy_quadrature_resolves_support_time_alias():
    from amsrr_det.walking_load import support_quadrature
    duty=.667466658503673
    for shift in [0.,.031,.127]:
        phases=(np.arange(4)/4+shift)%1.
        times,weights=support_quadrature(1.7,4,duty,phases)
        count=(((times[:,None]/1.7+phases)%1.)<duty).sum(axis=1)
        assert np.isclose(weights.sum(),1.)
        assert np.isclose(weights@count,4*duty)
        assert np.isclose(weights@(count==2),3-4*duty)
        # Energy constant within each support set must use its duration.
        power=np.where(count==2,1.4,.4)
        assert np.isclose(weights@power,1.4*(3-4*duty)+.4*(4*duty-2))


def test_energy_quadrature_integrates_smooth_variation():
    from amsrr_det.walking_load import support_quadrature
    times,weights=support_quadrature(2.,3,.73,[0.,.31,.71])
    for degree in range(6):
        assert np.isclose(weights@(times/2.)**degree,1/(degree+1))




def test_native_leaf_motion_retains_finite_python_trajectory():
    from amsrr_det.walking_leaf import score,reference_score
    from amsrr_det.walking import rotation_z
    rng=np.random.default_rng(480)
    for _ in range(4):
        parent=np.eye(4);parent[:3,:3]=rotation_z(rng.uniform(-3,3));parent[2,3]=.4
        args=(parent,0,2,1,rng.uniform(-1,1,3),np.array([.22,.03,0.]),1.9,.66)
        actual=score(*args);expected=reference_score(*args)
        np.testing.assert_allclose(actual['configurations'],expected['configurations'],atol=1e-10,rtol=1e-10)
        for key in ['cost','max_error','closure','peak_grid_speed']:
            assert actual[key]==pytest.approx(expected[key],abs=1e-10)


def test_eight_support_roles_use_distinct_unoccupied_real_ports():
    from amsrr_det.task_structure import grow_to_contacts
    from amsrr_det.model import validate_graph,DockEdge
    phase=np.arange(8)*np.pi/4
    points=np.c_[.8*np.cos(phase),.8*np.sin(phase),np.zeros(8)]
    root=np.eye(4);root[2,3]=.4
    edges,surfaces,_,_,_=grow_to_contacts(8,points,np.tile([0.,0.,-1.],(8,1)),
        np.random.default_rng(43),root_pose=root,multiple_contacts=True,support=True,
        contact_forces=np.tile([0.,0.,20.],(8,1)),contact_strokes=np.tile([.3,0.,0.],(8,1)))
    validate_graph(8,[DockEdge(**edge) for edge in edges])
    occupied={(e['a'],e['port_a']) for e in edges}|{(e['b'],e['port_b']) for e in edges}
    assert len(surfaces)==len(set(surfaces))==8
    assert not occupied.intersection(surfaces)


def test_motion_load_matches_full_mass_bias_with_armature():
    import pinocchio as pin
    from amsrr_det.walking_load import motion_load
    from amsrr_det.walking_plan import plant_for
    plant=plant_for(dict(modules=3,edges=[
        dict(a=0,port_a='pitch1',b=1,port_b='yaw1'),
        dict(a=1,port_a='pitch2',b=2,port_b='yaw2')]))
    drive=RobotDrive(plant);rng=np.random.default_rng(7204)
    for _ in range(8):
        q=plant.mj.qpos0.copy();q[:3]=rng.normal(size=3)
        q[3:7]=rng.normal(size=4);mujoco.mj_normalizeQuat(plant.mj,q)
        q[drive.qi]=rng.uniform(-.3,.3,len(drive.qi))
        v=rng.normal(size=plant.mj.nv);a=rng.normal(size=plant.mj.nv)
        drive.observe(q,v);full=drive.mass@a+drive.d.qfrc_bias
        qp,actual=motion_load(plant,drive,q,v,a)
        _,_,pv,mv,bases=plant.coordinate_maps;expected=np.empty(plant.pin.nv);expected[pv]=full[mv]
        for pos,vel,source in bases:
            expected[vel:vel+3]=pin.Quaternion(qp[pos+3:pos+7]).matrix().T@full[source:source+3]
        np.testing.assert_allclose(actual,expected,atol=1e-11,rtol=1e-11)


def test_simultaneous_feet_do_not_duplicate_support_transitions():
    from amsrr_det.walking_load import support_switch_times
    times=support_switch_times(2.,4,.7,[0.,0.,.5,.5])
    expected=support_switch_times(2.,2,.7,[0.,.5])
    np.testing.assert_array_equal(times,expected)
    assert len(times)==len(np.unique(times))==8


def test_energy_average_cannot_hide_missing_successful_measurement():
    from scripts.report_walking_panel import energy_summary
    rows=[dict(success=True,rotor_burden_hover_ratio=.3),dict(success=True,rotor_burden_hover_ratio=.4),
          dict(success=False,rotor_burden_hover_ratio=.01)]
    report=energy_summary(rows)
    assert report['valid'] and report['mean']==pytest.approx(.35)
    rows[1].pop('rotor_burden_hover_ratio')
    assert not energy_summary(rows)['valid']


def test_turning_and_swing_acceleration_both_change_timing_proposal():
    from amsrr_det.walking_growth import motion_timing
    points=np.array([[0.,-.8,.03],[0.,.8,.03]])
    slow=WalkSpec(modules=5,speed=.22,curvature=0.)
    fast=WalkSpec(modules=5,speed=.3,curvature=.3)
    ts,ds=motion_timing(slow,points,2.,.76)
    tf,df=motion_timing(fast,points,2.,.76)
    assert tf<ts and df<ds
    assert slow.speed*ts>.15 and fast.speed*tf>.15


def test_stance_mobility_pose_and_velocity_derivatives_are_consistent():
    import pinocchio as pin
    from amsrr_det.walking_mobility import residual
    from amsrr_det.model import build_model
    model,meta=build_model();rng=np.random.default_rng(516)
    q=pin.integrate(model,pin.neutral(model),rng.normal(0,.1,model.nv))
    moving=np.r_[np.arange(6),[a['v'] for a in meta['actuators'] if a['role']=='dock']]
    v=np.zeros(model.nv);v[moving]=rng.normal(0,.1,len(moving))
    frames=[meta['frames']['m0/pitch_dock_mech1']];points=[np.array([.01,.02,.03])]
    data=model.createData()
    def evaluate(q,v):return residual(model,q,v,moving,frames,points,[0],[.22,0.,0.],[[.7,0.,.12]],data=data)
    for sign in [-1.,1.]:
        pose=pin.integrate(model,q,sign*.02*np.ones(model.nv))
        fresh=residual(model,pose,sign*v,moving,frames,points,[0],[.22,0.,0.],[[.7,0.,.12]])
        for actual,expected in zip(evaluate(pose,sign*v),fresh):
            np.testing.assert_array_equal(actual,expected)
    _,J,V=evaluate(q,v);h=1e-6
    pose=np.column_stack([(evaluate(pin.integrate(model,q,h*np.eye(model.nv)[i]),v)[0]-
                           evaluate(pin.integrate(model,q,-h*np.eye(model.nv)[i]),v)[0])/(2*h) for i in range(model.nv)])
    rates=np.column_stack([(evaluate(q,v+h*np.eye(model.nv)[i])[0]-evaluate(q,v-h*np.eye(model.nv)[i])[0])/(2*h) for i in moving])
    np.testing.assert_allclose(J,pose,atol=1e-7,rtol=1e-7)
    np.testing.assert_allclose(V,rates,atol=1e-7,rtol=1e-7)


def test_periodic_direction_polynomial_envelope():
    from scipy.interpolate import CubicSpline
    from amsrr_det.walking_direction import spline_envelope
    rng=np.random.default_rng(147)
    spline=CubicSpline(np.linspace(0,2,14),rng.normal(size=(14,7)))
    columns=np.array([0,2,5])
    for derivative in [0,1]:
        curve=spline.derivative(derivative)
        for left,right in [(0.,.17),(.18,1.34),(1.89,2.)]:
            low,high=spline_envelope(curve,columns,left,right)
            for j,col in enumerate(columns):
                values=[float(curve(left)[col]),float(curve(right)[col])]
                for k in range(len(curve.x)-1):
                    lo=max(left,curve.x[k]);hi=min(right,curve.x[k+1])
                    if hi<lo:continue
                    values.extend([curve(lo)[col],curve(hi)[col]])
                    for root in np.roots(np.polyder(curve.c[:,k,col])):
                        if abs(root.imag)<1e-10 and lo-curve.x[k]<=root.real<=hi-curve.x[k]:
                            values.append(np.polyval(curve.c[:,k,col],root.real))
                np.testing.assert_allclose([low[j],high[j]],[min(values),max(values)],atol=2e-13)


@pytest.mark.parametrize("worker_count",[1,2])
def test_auxiliary_deadline_keeps_last_checked_plan(monkeypatch,worker_count):
    import time
    import multiprocessing as mp
    from amsrr_det import walking_refine
    initial={'cost':.5,'gait':'original'}
    children={p.pid for p in mp.active_children()}
    def unfinished(pipe,plant,plan,spec,deadline,cpu,cpus,mode="both"):
        import os
        os.setsid()
        assert type(cpu) is int and all(type(c) is int for c in cpus)
        pipe.send(dict(cost=.4,gait='checked',iterations=2))
        pipe.send(dict(cost=.6,gait='worse',iterations=4))
        time.sleep(10)
    monkeypatch.setattr(walking_refine,'_improve',unfinished)
    start=time.perf_counter()
    selected,report=walking_refine.refine(None,initial,None,deadline=start+1.,cpus=set(range(worker_count)))
    assert time.perf_counter()-start<1.5
    assert selected['gait']=='checked' and selected['cost']==.4
    assert initial=={'cost':.5,'gait':'original'}
    assert len(report['updates'])==1
    assert {p.pid for p in mp.active_children()}==children
