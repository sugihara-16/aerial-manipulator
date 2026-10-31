"""Verify proposed directions are reduced to actual physical rotor inputs."""
import numpy as np
import pinocchio as pin
from amsrr_det.model import build_model
from amsrr_det.rotor_distribution import rotor_actions
from amsrr_det.steerable_trim import steerable_trim
from amsrr_det.equilibrium import static_balance


def test_terminal_rotor_basis_matches_full_articulated_kinematics_and_derivative():
    from amsrr_det.rotor_distribution import rotor_action_basis,rotor_basis_actions
    rng=np.random.default_rng(208)
    for n in [1,3,7]:
        edges=[dict(a=i-1,port_a='pitch1',b=i,port_b='yaw2') for i in range(1,n)]
        model,meta=build_model(n,edges);q=pin.neutral(model)
        q[:3]=rng.normal(size=3);q[3:7]=pin.Quaternion(pin.exp3(rng.normal(size=3))).coeffs()
        for a in meta['actuators']:q[a['q']]=rng.uniform(-1.3,1.3)
        qi=[meta['actuators'][r['gimbal']]['q'] for r in meta['rotors']]
        basis=rotor_action_basis(model,meta,q)
        for _ in range(4):
            angles=rng.uniform(-1.8,1.8,len(qi));q[qi]=angles
            np.testing.assert_allclose(rotor_basis_actions(basis,angles),rotor_actions(model,meta,q),atol=3e-14)
            derivative=-basis[1]*np.sin(angles)+basis[2]*np.cos(angles)
            plus=q.copy();minus=q.copy();plus[qi]+=1e-5;minus[qi]-=1e-5
            np.testing.assert_allclose(derivative,(rotor_actions(model,meta,plus)-rotor_actions(model,meta,minus))/2e-5,atol=3e-10)


def test_convex_ray_generalized_action_matches_real_rotor_for_this_mechanism():
    model,meta=build_model(2,[dict(a=0,port_a='pitch1',b=1,port_b='yaw1')])
    rng=np.random.default_rng(181);q=pin.neutral(model)
    q[2]=2.;q[3:7]=pin.Quaternion(pin.exp3(np.array([.3,-.4,.2]))).coeffs()
    for a in meta['actuators']:q[a['q']]=rng.uniform(-.5,.5)
    qi=[meta['actuators'][r['gimbal']]['q'] for r in meta['rotors']]
    angles=np.linspace(-1.5,1.5,9);rays=[]
    for angle in angles:
        probe=q.copy();probe[qi]=angle;rays.append(rotor_actions(model,meta,probe))
    for _ in range(12):
        coefficients=rng.uniform(0.,1.,(9,len(qi)))
        vertical=np.cos(angles)@coefficients;tangent=np.sin(angles)@coefficients
        probe=q.copy();probe[qi]=np.arctan2(tangent,vertical)
        expected=np.einsum('knr,kr->n',rays,coefficients)
        actual=rotor_actions(model,meta,probe)@np.hypot(vertical,tangent)
        np.testing.assert_allclose(actual,expected,atol=2e-13)
        assert np.all(np.hypot(vertical,tangent)<=coefficients.sum(axis=0)+1e-12)


def test_accepted_direction_uses_exact_static_recheck_and_actual_rotor_count():
    model,meta=build_model();q=pin.neutral(model);q[2]=1.
    for r in meta['rotors']:q[meta['actuators'][r['gimbal']]['q']]=.3
    pose,trim=steerable_trim(model,meta,q,payload_fraction=0.)
    assert trim['status']=='static_screen_pass'
    assert len(trim['thrust'])==len(meta['rotors'])
    checked=static_balance(model,meta,pose,payload_fraction=0.)
    assert checked['status']=='static_screen_pass'
    assert checked['equilibrium_residual']<1e-7
    for a in meta['actuators']:assert a['lower']<=pose[a['q']]<=a['upper']


def test_growth_motor_scale_matches_hover_capacity_and_never_accepts_overload():
    model,meta=build_model();q=pin.neutral(model);nr=len(meta['rotors'])
    # Four parallel vertical rotors at a symmetric pose: normalized force
    # capacity has an independent analytic solution, including overload.
    rays=np.asarray([rotor_actions(model,meta,q)])
    for fraction in [.5,1.,2.]:
        force=nr*20.*fraction
        requested=(rays[0]@np.full(nr,20.*fraction))[:6]
        result=static_balance(model,meta,q,rotor_rays=rays,relaxed_actuator_scale=True,
            base_load=requested,fast_lp=True)
        assert result['status']=='direction_proposal' and not result['task_accepted']
        assert result['scope']=='relaxed_motor_scale_for_growth_not_feasibility'
        np.testing.assert_allclose(result['minimum_peak_effort_ratio'],fraction,atol=1e-7)
    assert static_balance(model,meta,q,base_load=[0,0,160.,0,0,0])['status']!='static_screen_pass'


def test_compact_motor_polygon_preserves_ray_capacity_for_partial_trees():
    from amsrr_det.rotor_distribution import rotor_action_basis,rotor_basis_actions
    from amsrr_det.valve_growth import direction_facets
    from amsrr_det.valve import ValveSpec
    from amsrr_det.valve_plan import bearing_support
    from amsrr_det.grasp_task import material_force_contacts
    from types import SimpleNamespace
    import json
    from pathlib import Path
    saved=json.loads(Path(__file__).with_name('fixtures').joinpath('valve_yaw_symmetry_input.json').read_text())
    rng=np.random.default_rng(220)
    for n in [1,3,7]:
        edges=[dict(a=i-1,port_a='pitch1',b=i,port_b='yaw2') for i in range(1,n)]
        spec=ValveSpec(torque=60.)
        if n==3:edges=saved['edges']
        model,meta=build_model(n,edges,spec.payload);q=pin.neutral(model)
        if n==3:q=np.asarray(saved['q'])
        else:
            q[3:7]=pin.Quaternion(pin.exp3(rng.normal(0,.3,3))).coeffs()
            for a in meta['actuators']:q[a['q']]=rng.uniform(-.8,.8)
        basis=rotor_action_basis(model,meta,q)
        np.testing.assert_allclose(basis[0],0.,atol=1e-14)
        rays=np.asarray([rotor_basis_actions(basis,np.full(4*n,angle)) for angle in np.linspace(-1.98,1.98,5)])
        conditions=[dict()]
        if n==3:
            contacts=material_force_contacts(SimpleNamespace(pin=model,meta=meta),spec.shape,q,saved['surfaces'],force_limit=None)
            conditions.append(dict(contacts=contacts,object_wrench=np.r_[np.zeros(5),66.],object_support=bearing_support(spec)))
        for kwargs in conditions:
            old=static_balance(model,meta,q,rotor_rays=rays,relaxed_actuator_scale=True,fast_lp=True,payload_fraction=0.,**kwargs)
            compact=static_balance(model,meta,q,rotor_polytope=(np.asarray(basis[1:]),direction_facets(-1.98,1.98)),
                relaxed_actuator_scale=True,fast_lp=True,payload_fraction=0.,**kwargs)
            assert old['status']==compact['status']=='direction_proposal'
            assert not compact['task_accepted']
            np.testing.assert_allclose(compact['minimum_peak_effort_ratio'],old['minimum_peak_effort_ratio'],rtol=1e-6,atol=1e-6)
