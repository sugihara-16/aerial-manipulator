import numpy as np
import pinocchio as pin
from amsrr_det.model import build_model,DockEdge
from amsrr_det.equilibrium import static_balance


def test_fast_lp_matches_hover_optimum_and_unsupported_payload_failure():
    for modules,edges in [(1,[]),(3,[DockEdge(0,'pitch1',1,'yaw2'),DockEdge(1,'pitch1',2,'yaw2')])]:
        model,meta=build_model(modules,edges);q=pin.neutral(model)
        expected=static_balance(model,meta,q,minimize_burden=True)
        actual=static_balance(model,meta,q,minimize_burden=True,fast_lp=True)
        assert actual['status']==expected['status']=='static_screen_pass'
        assert abs(actual['minimum_peak_effort_ratio']-expected['minimum_peak_effort_ratio'])<1e-7
        assert actual['equilibrium_residual']<1e-7
        assert actual['inequality_residual']<1e-7
        assert abs(sum(np.array(actual['thrust'])**1.5)-sum(np.array(expected['thrust'])**1.5))<1e-5
    model,meta=build_model(payload=dict(mass=1.,size=[.3,.4,.15]))
    assert static_balance(model,meta,pin.neutral(model),fast_lp=True)['status']=='unresolved_or_inner_model_infeasible'


def test_burden_objective_respects_physical_balance_and_convex_weight_bound():
    model,meta=build_model();q=pin.neutral(model)
    plain=static_balance(model,meta,q)
    result=static_balance(model,meta,q,minimize_burden=True)
    assert result['status']=='static_screen_pass' and result['burden_refinement_success']
    thrust=np.array(result['thrust'])
    cost=np.sum(thrust**1.5)
    lower=(meta['mass']*9.80665)**1.5/np.sqrt(len(thrust))
    assert lower-1e-7<=cost<1.001*lower
    assert cost<np.sum(np.array(plain['thrust'])**1.5)
    assert result['equilibrium_residual']<1e-7
    assert np.min(thrust)>=0. and np.max(thrust)<=20.


def test_quadratic_force_proposal_and_power_proposal_do_not_share_solver_cache():
    from amsrr_det.equilibrium import _minimum_burden
    # Same nonnegative force contract, distinct strictly convex objectives.
    # For T0+2*T1=6, the QP gives (1.2,2.4); power 1.5 gives (2/3,8/3).
    E=np.array([[1.,2.,0.]]);bounds=[(0.,20.),(0.,20.)];cache={}
    for power,expected in [(2.,[1.2,2.4]),(1.5,[2/3,8/3]),(2.,[1.2,2.4])]:
        result=_minimum_burden(E,np.array([6.]),np.empty((0,3)),np.empty(0),bounds,2,1.,reusable=cache,power=power,strict=True)
        assert result.success
        np.testing.assert_allclose(result.x[:2],expected,atol=2e-5)
        assert abs(E@result.x-6)<1e-8
    assert len(cache)==2


def test_shared_pad_budget_allows_unequal_load_and_is_invariant_to_sampling():
    model,meta=build_model(payload=dict(size=[.2,.2,.2],mass=.2))
    q=pin.neutral(model);q[meta['object']['q']+1]=.09
    contacts=[dict(frame='m0/root',point=[0,y,0],target_frame='object',
                   target_point=[0,y-.09,0],normal=[0,0,-1],mu=.6,
                   max_force=30.,pad_force_limit=2.1) for y in [-.1,.1]]
    shared=static_balance(model,meta,q,contacts)
    assert shared['status']=='static_screen_pass'
    force=[c['normal_force'] for c in shared['contacts']]
    assert max(force)>1.8 and sum(force)<2.1
    doubled=static_balance(model,meta,q,contacts+contacts)
    assert doubled['status']=='static_screen_pass'
    assert abs(shared['minimum_peak_effort_ratio']-doubled['minimum_peak_effort_ratio'])<1e-8
    equal=[{k:v for k,v in c.items() if k!='pad_force_limit'} for c in contacts]
    for c in equal:c['max_force']=1.05
    assert static_balance(model,meta,q,equal)['status']=='unresolved_or_inner_model_infeasible'
    overloaded=[dict(c,pad_force_limit=1.8) for c in contacts]
    assert static_balance(model,meta,q,overloaded+overloaded)['status']=='unresolved_or_inner_model_infeasible'


def test_hover_screen_balances_gravity_within_actual_limits():
    model,meta=build_model(3,[DockEdge(0,'pitch1',1,'yaw2'),DockEdge(1,'pitch1',2,'yaw2')])
    r=static_balance(model,meta,pin.neutral(model))
    assert r['status']=='static_screen_pass'
    assert abs(sum(r['thrust'])+model.gravity.linear[2]*meta['mass'])<1e-9
    assert all(0<=f<=20 for f in r['thrust'])
    assert all(x['normalized']<=1 for x in r['joints'])
    assert not r['task_accepted']


def test_free_payload_cannot_be_supported_by_robot_thrust_without_contact():
    model,meta=build_model(payload=dict(mass=1.,size=[.3,.4,.15]))
    r=static_balance(model,meta,pin.neutral(model))
    assert r['status']=='unresolved_or_inner_model_infeasible'
    assert not r['physical_impossibility_proven']


def test_secondary_thrust_balancing_preserves_minimum_peak_when_requested():
    model,meta=build_model(3,[DockEdge(0,'pitch1',1,'yaw2'),DockEdge(1,'pitch1',2,'yaw2')])
    q=pin.neutral(model)
    result=static_balance(model,meta,q,balance_thrust=True,peak_slack=1e-6)
    assert result['status']=='static_screen_pass'
    assert result['max_normalized_joint_effort']<=result['minimum_peak_effort_ratio']+1.1e-6
    assert result['equilibrium_residual']<1e-7
    assert result['inequality_residual']<1e-7


def test_contact_on_object_adds_equal_opposite_load_to_robot():
    model,meta=build_model(payload=dict(mass=.2,size=[.3,.4,.15]));q=pin.neutral(model)
    # Algebraic screen only: a load-bearing point at the robot reference point.
    # This intentionally does not claim the point belongs to a real gripper.
    contact=dict(frame='m0/root',point=[0,0,0],target_frame='object',target_point=[0,0,0],normal=[0,0,-1],mu=.6,max_force=30.)
    r=static_balance(model,meta,q,[contact])
    assert r['status']=='static_screen_pass'
    assert abs(r['contacts'][0]['normal_force']+.2*model.gravity.linear[2])<1e-9
    assert abs(sum(r['thrust'])+meta['mass']*model.gravity.linear[2])<1e-9


def test_contact_without_auxiliary_force_cap_still_respects_motor_limits():
    # Algebraic load at the root: remove only the optional 30 N contact cap.
    # The independently known 4 kg load fits total rotor capacity; 8 kg does not.
    for mass in [4.,8.]:
        model,meta=build_model(payload=dict(mass=mass,size=[.3,.4,.15]));q=pin.neutral(model)
        contact=dict(frame='m0/root',point=[0,0,0],target_frame='object',
                     target_point=[0,0,0],normal=[0,0,-1],mu=.6,max_force=None)
        for direct in [False,True]:
            result=static_balance(model,meta,q,[contact],fast_lp=True,direct_burden=direct)
            if mass==4.:
                assert result['status']=='static_screen_pass'
                assert abs(result['contacts'][0]['normal_force']-mass*9.80665)<1e-7
                assert max(result['thrust'])<=20.+1e-8
                assert max(j['normalized'] for j in result['joints'])<=1.+1e-8
            else:
                assert result['status']!='static_screen_pass'
        capped=static_balance(model,meta,q,[dict(contact,max_force=30.,pad_force_limit=30.)])
        assert capped['status']!='static_screen_pass'


def test_collinear_point_supports_do_not_gain_torque_about_their_line():
    model,meta=build_model(payload=dict(mass=.2,size=[.3,.4,.15]));q=pin.neutral(model)
    contacts=[dict(frame='m0/root',point=[0,y,0],target_frame='object',target_point=[0,y,0],
                   normal=[0,0,-1],mu=.6,max_force=30.) for y in [-.1,.1]]
    assert static_balance(model,meta,q,contacts)['status']=='static_screen_pass'
    assert static_balance(model,meta,q,contacts,object_wrench=[0,0,0,.01,0,0])['status']=='static_screen_pass'
    assert static_balance(model,meta,q,contacts,object_wrench=[0,0,0,0,.01,0])['status']=='unresolved_or_inner_model_infeasible'


def test_minimum_pad_load_sums_material_points_and_preserves_force_balance():
    model,meta=build_model(payload=dict(mass=.2,size=[.3,.4,.15]));q=pin.neutral(model)
    # Three points share one pad; their total must equal the independently
    # known payload weight. A per-point 1.5 N minimum would wrongly reject it.
    contacts=[dict(frame='m0/root',point=[0,y,0],target_frame='object',target_point=[0,y,0],
                   normal=[0,0,-1],mu=.6,max_force=.8) for y in [-.1,0.,.1]]
    for balanced in [False,True]:
        result=static_balance(model,meta,q,contacts,minimum_pad_normal_force_n=1.5,balance_thrust=balanced)
        assert result['status']=='static_screen_pass'
        assert abs(result['pad_normal_forces_n']['m0/root']-.2*9.80665)<1e-9
        assert max(c['normal_force'] for c in result['contacts'])<=.8+1e-9
    assert static_balance(model,meta,q,contacts,minimum_pad_normal_force_n=2.)['status']=='unresolved_or_inner_model_infeasible'


def test_minimum_load_is_required_on_each_pad_not_only_the_combined_load():
    model,meta=build_model(payload=dict(mass=.2,size=[.3,.4,.15]));q=pin.neutral(model)
    data=model.createData();pin.framesForwardKinematics(model,data,q)
    names=['m0/root','m0/yaw_connect_dummy_2']
    contacts=[dict(frame=name,point=data.oMf[meta['frames'][name]].actInv(np.zeros(3)),
                   target_frame='object',target_point=[0,0,0],normal=[0,0,-1],mu=.6,max_force=30.) for name in names]
    result=static_balance(model,meta,q,contacts,minimum_pad_normal_force_n=.5)
    assert result['status']=='static_screen_pass'
    assert all(value>=.5-1e-9 for value in result['pad_normal_forces_n'].values())
    assert static_balance(model,meta,q,contacts,minimum_pad_normal_force_n=1.)['status']=='unresolved_or_inner_model_infeasible'


def test_free_support_distribution_preserves_commanded_wrench_without_object_force():
    from amsrr_det.equilibrium import free_support_distribution
    model,meta=build_model(3,[DockEdge(0,'pitch1',1,'yaw2'),DockEdge(1,'pitch1',2,'yaw2')],payload=dict(mass=.6,size=[.3,.3,.3]))
    q=pin.neutral(model);thrust=np.full(len(meta['rotors']),(meta['mass']-.6)*9.80665/len(meta['rotors']))
    # Uneven but known rotor commands need not be an equilibrium. Only the
    # total commanded base action is preserved by this local screen.
    thrust[0]+=.2;thrust[4]-=.2
    result=free_support_distribution(model,meta,q,thrust)
    assert result['status']=='static_screen_pass'
    assert result['contacts']==[] and not result['task_accepted']
    assert result['max_normalized_joint_effort']==max(j['normalized'] for j in result['joints'])
    data=model.createData();pin.framesForwardKinematics(model,data,q)
    def generalized(values):
        out=np.zeros(model.nv)
        for r,f in zip(meta['rotors'],values):
            J=pin.computeFrameJacobian(model,data,q,r['frame'],pin.LOCAL_WORLD_ALIGNED)
            out+=J[:3].T@data.oMf[r['frame']].rotation[:,2]*f
        return out
    before,after=generalized(thrust),generalized(result['thrust'])
    np.testing.assert_allclose(after[:6],before[:6],atol=1e-7)
    acceleration=np.zeros(model.nv);acceleration[:6]=result['free_base_acceleration']
    required=pin.rnea(model,data,q,np.zeros(model.nv),acceleration)
    np.testing.assert_allclose(required[:6],after[:6],atol=1e-7)
    for a,record in zip(meta['actuators'],result['joints']):
        np.testing.assert_allclose(record['effort_nm'],required[a['v']]-after[a['v']],atol=1e-9)
        assert abs(record['effort_nm'])<=a['effort']+1e-8


def test_direct_burden_solves_same_force_contract_and_rejects_overload():
    from amsrr_det.structure_bounds import hover_input_burden
    model,meta=build_model();q=pin.neutral(model)
    for load in [.8,1.,1.2]:
        wanted=[0,0,meta['mass']*9.80665*load,0,0,0]
        old=static_balance(model,meta,q,minimize_burden=True,fast_lp=True,base_load=wanted)
        new=static_balance(model,meta,q,direct_burden=True,base_load=wanted)
        assert old['status']==new['status']=='static_screen_pass'
        assert new['equilibrium_residual']<1e-7
        assert new['inequality_residual']<1e-7
        assert new['max_normalized_joint_effort']<=1.+1e-7
        assert hover_input_burden(new['thrust'])<=hover_input_burden(old['thrust'])+1e-5
        assert 'minimum_peak_effort_ratio' not in new
    overload=static_balance(model,meta,q,base_load=[0,0,10000,0,0,0],direct_burden=True)
    assert overload['status']!='static_screen_pass'


def test_ground_capacity_is_bounded_by_weight_and_preserves_real_motor_limits():
    model,meta=build_model();q=pin.neutral(model)
    # Algebraic point on the root, not a claim about available foot material.
    contacts=[dict(frame='m0/root',point=[0,0,0],normal=[0,0,1],mu=.8,max_force=None)]
    r=static_balance(model,meta,q,contacts,fast_lp=True,maximize_contact_load=True,peak_effort_bound=.9)
    assert r['status']=='static_screen_pass'
    assert 'minimum_peak_effort_ratio' not in r
    assert r['max_normalized_joint_effort']<=.9+1e-8
    assert .9*meta['mass']*9.80665<r['contacts'][0]['normal_force']<=meta['mass']*9.80665+1e-7
    constrained=static_balance(model,meta,q,contacts,fast_lp=True,minimize_burden=True,
        minimum_contact_normal_force_n=.8*r['contacts'][0]['normal_force'],peak_effort_bound=.9)
    assert constrained['status']=='static_screen_pass'
    assert constrained['contacts'][0]['normal_force']>=.8*r['contacts'][0]['normal_force']-1e-7
    assert constrained['max_normalized_joint_effort']<=.9+1e-8
