import numpy as np
import pinocchio as pin
from amsrr_det.model import build_model,DockEdge
from amsrr_det.equilibrium import static_balance


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
