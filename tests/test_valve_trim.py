"""The release endpoint must support loads between free flight and work."""
import json
from pathlib import Path
import numpy as np
from amsrr_det.valve import ValveSpec,ValvePlant
from amsrr_det.valve_trim import endpoint_load_envelope
from amsrr_det.valve_plan import bearing_support
from amsrr_det.grasp_task import material_force_contacts
from amsrr_det.equilibrium import static_balance


def test_gravity_axis_rotation_preserves_actual_three_module_load_screens():
    import pinocchio as pin
    from amsrr_det.valve_plan import rotate_assembly,gravity_axis_symmetry
    from amsrr_det.rotor_distribution import rotor_actions
    saved=json.loads(Path(__file__).with_name('fixtures').joinpath('valve_yaw_symmetry_input.json').read_text())
    spec=ValveSpec(**saved['request']);p=ValvePlant(saved['modules'],saved['edges'],spec).control
    q=np.asarray(saved['q']);surfaces=list(map(tuple,saved['surfaces']));data=p.pin.createData()
    assert gravity_axis_symmetry(p,spec)
    assert not gravity_axis_symmetry(p,ValveSpec(axis=[0,.6,.8]))
    expected_gravity=pin.computeGeneralizedGravity(p.pin,data,q).copy();expected_rotors=rotor_actions(p.pin,p.meta,q)
    for angle in np.linspace(0,np.pi,9):
        moved=rotate_assembly(p,spec,q,angle,0.)
        np.testing.assert_allclose(pin.computeGeneralizedGravity(p.pin,data,moved),expected_gravity,atol=1e-12)
        np.testing.assert_allclose(rotor_actions(p.pin,p.meta,moved),expected_rotors,atol=1e-12)
        contacts=material_force_contacts(p,spec.shape,moved,surfaces,force_limit=None,material_tangents=True)
        assert all(abs(c['normal'][2])<1e-12 for c in contacts)
        for amount in [0.,.9,1.1]:
            result=static_balance(p.pin,p.meta,moved,contacts if amount else (),payload_fraction=0.,
                object_support=bearing_support(spec) if amount else None,
                object_wrench=np.r_[np.zeros(3),np.asarray(spec.axis)*spec.torque*amount] if amount else None,
                direct_burden=True,burden_power=2.)
            assert result['status']=='static_screen_pass'
            assert result['equilibrium_residual']<1e-7 and result['max_normalized_joint_effort']<=1.+1e-7


def test_material_friction_basis_keeps_multi_pad_load_problem_rotation_equivariant():
    from amsrr_det.valve_plan import rotate_assembly,loaded_trim
    from amsrr_det.equilibrium import contact_basis
    saved=json.loads(Path(__file__).with_name('fixtures').joinpath('valve_multi_axis_symmetry_input.json').read_text())
    spec=ValveSpec(**saved['request']);p=ValvePlant(saved['modules'],saved['edges'],spec).control
    surfaces=list(map(tuple,saved['surfaces']));q=np.asarray(saved['q'])
    q,envelope=endpoint_load_envelope(p,spec,q,surfaces)
    assert envelope['free']==envelope['load']=='static_screen_pass'
    q,_=loaded_trim(p,spec,q,surfaces)
    costs=[]
    for angle in np.linspace(0,np.pi,181):
        moved=rotate_assembly(p,spec,q,angle,0.)
        contacts=material_force_contacts(p,spec.shape,moved,surfaces,force_limit=None,material_tangents=True)
        assert any(abs(c['normal'][2])>.1 for c in contacts)
        for c in contacts:
            R=contact_basis(c);np.testing.assert_allclose(R.T@R,np.eye(3),atol=1e-12)
            # Every vertex of the inner diamond lies on the physical cone;
            # its convex hull cannot enlarge isotropic Coulomb friction.
            for tangent in R[:,1:].T:
                force=R[:,0]+c['mu']*tangent
                np.testing.assert_allclose(np.linalg.norm(force-(force@R[:,0])*R[:,0]),c['mu'],atol=1e-12)
        values=[]
        for amount in [0.,.9,1.1]:
            report=static_balance(p.pin,p.meta,moved,contacts if amount else (),payload_fraction=0.,
                object_support=bearing_support(spec) if amount else None,
                object_wrench=np.r_[np.zeros(5),spec.torque*amount] if amount else None,
                direct_burden=True,burden_power=2.)
            assert report['status']=='static_screen_pass'
            assert report['equilibrium_residual']<1e-7 and report['max_normalized_joint_effort']<=1.+1e-7
            values.append(np.sum(np.asarray(report['thrust'])**1.5))
        costs.append(values)
    np.testing.assert_allclose(costs,np.broadcast_to(costs[0],np.shape(costs)),rtol=1e-5,atol=1e-4)


def test_uncapped_material_contacts_work_with_exact_planning_memo(tmp_path,monkeypatch):
    from amsrr_det import planning_memo
    from amsrr_det.steerable_trim import steerable_trim
    saved=json.loads(Path(__file__).with_name('fixtures').joinpath('valve_release_input.json').read_text())
    spec=ValveSpec(**saved['request']);p=ValvePlant(1,[],spec).control
    q=np.asarray(saved['q']);surfaces=list(map(tuple,saved['surfaces']))
    contacts=material_force_contacts(p,spec.shape,q,surfaces,force_limit=None,material_tangents=True)
    assert all(c['max_force'] is None and 'pad_force_limit' not in c for c in contacts)
    json.dumps(contacts,default=lambda v:v.tolist(),allow_nan=False)
    settings=dict(payload_fraction=0.,object_support=bearing_support(spec),
        object_wrench=np.r_[np.zeros(3),np.asarray(spec.axis)*spec.torque])
    expected=steerable_trim(p.pin,p.meta,q,contacts,**settings)
    monkeypatch.setattr(planning_memo,'active',planning_memo.PlanningMemo(tmp_path))
    for _ in range(2):
        actual=steerable_trim(p.pin,p.meta,q,contacts,**settings)
        np.testing.assert_array_equal(actual[0],expected[0])
        assert actual[1]['status']==expected[1]['status']
    assert len(list(tmp_path.glob('*.pickle')))==1


def test_release_directions_preserve_geometry_and_support_intermediate_loads():
    saved=json.loads(Path(__file__).with_name('fixtures').joinpath('valve_release_input.json').read_text())
    spec=ValveSpec(**saved['request']);system=ValvePlant(1,[],spec);p=system.control
    q=np.asarray(saved['q']);surfaces=list(map(tuple,saved['surfaces']))
    assert static_balance(p.pin,p.meta,q,payload_fraction=0.)['status']!='static_screen_pass'
    final,report=endpoint_load_envelope(p,spec,q,surfaces)
    assert report['free']==report['load']=='static_screen_pass'
    directions=[a['q'] for a in p.meta['actuators'] if a['role']=='vectoring']
    unchanged=np.setdiff1d(np.arange(len(q)),directions)
    np.testing.assert_array_equal(final[unchanged],q[unchanged])
    contacts=material_force_contacts(p,spec.shape,final,surfaces)
    for fraction in [0.,.25,.5,.75,1.,1.1]:
        result=static_balance(p.pin,p.meta,final,contacts,payload_fraction=0.,
            object_wrench=np.r_[np.zeros(3),np.asarray(spec.axis)*spec.torque*fraction],
            object_support=bearing_support(spec))
        assert result['status']=='static_screen_pass'
        assert result['max_normalized_joint_effort']<=1.+1e-7
        assert result['equilibrium_residual']<1e-7


def test_force_workspace_reuses_only_identical_geometry_and_joint_loads():
    saved=json.loads(Path(__file__).with_name('fixtures').joinpath('valve_release_input.json').read_text())
    spec=ValveSpec(**saved['request']);p=ValvePlant(1,[],spec).control
    q,report=endpoint_load_envelope(p,spec,np.asarray(saved['q']),list(map(tuple,saved['surfaces'])))
    contacts=material_force_contacts(p,spec.shape,q,list(map(tuple,saved['surfaces'])))
    workspace={}
    for fraction in [.9,1.1,1.]:
        options=dict(payload_fraction=0.,object_support=bearing_support(spec),direct_burden=True,
            object_wrench=np.r_[np.zeros(3),np.asarray(spec.axis)*spec.torque*fraction])
        expected=static_balance(p.pin,p.meta,q,contacts,**options)
        actual=static_balance(p.pin,p.meta,q,contacts,workspace=workspace,**options)
        assert actual['status']==expected['status']=='static_screen_pass'
        np.testing.assert_allclose(actual['thrust'],expected['thrust'],atol=1e-10)
    assert len(workspace)==1
    contacts[0]['mu']*=.5
    options['base_load']=[0,0,20,0,0,0]
    expected=static_balance(p.pin,p.meta,q,contacts,**options)
    actual=static_balance(p.pin,p.meta,q,contacts,workspace=workspace,**options)
    assert actual['status']==expected['status']
    if 'thrust' in actual:np.testing.assert_allclose(actual['thrust'],expected['thrust'],atol=1e-10)
    assert len(workspace)==2


def test_convex_burden_direction_sensitivity_matches_independent_solves(monkeypatch):
    from amsrr_det.structure_bounds import hover_input_burden
    saved=json.loads(Path(__file__).with_name('fixtures').joinpath('valve_force_gradient_input.json').read_text())
    spec=ValveSpec(**saved['request']);p=ValvePlant(saved['modules'],saved['edges'],spec).control
    q=np.asarray(saved['q']);contacts=material_force_contacts(p,spec.shape,q,list(map(tuple,saved['surfaces'])))
    options=dict(payload_fraction=0.,object_support=bearing_support(spec),direct_burden=True,minimum_pad_normal_force_n=.1,
        object_wrench=np.r_[np.zeros(3),np.asarray(spec.axis)*spec.torque*1.1])
    checked=static_balance(p.pin,p.meta,q,contacts,direct_gradient=True,**options)
    # The production objective tolerance is adequate for force selection.
    # A tiny finite-difference oracle needs tighter objective precision, since
    # subtracting near-equal costs amplifies their numerical error by 1/h.
    from amsrr_det import equilibrium
    solve=equilibrium._minimum_burden
    def strict(*args,**kwargs):
        kwargs['strict']=True
        return solve(*args,**kwargs)
    monkeypatch.setattr(equilibrium,'_minimum_burden',strict)
    for gradient,rotor in zip(checked['burden_direction_gradient'],p.meta['rotors']):
        qi=p.meta['actuators'][rotor['gimbal']]['q'];values=[]
        for sign in [1.,-1.]:
            moved=q.copy();moved[qi]+=sign*1e-4
            row=static_balance(p.pin,p.meta,moved,contacts,**options)
            assert row['status']=='static_screen_pass'
            values.append(hover_input_burden(row['thrust']))
        np.testing.assert_allclose(gradient,(values[0]-values[1])/2e-4,rtol=1e-4,atol=1e-4)


def test_direction_cost_improvement_cannot_discard_release_support():
    from amsrr_det.valve_direction_refinement import refine
    from amsrr_det.free_flight_trim import check_fixed_thrust
    saved=json.loads(Path(__file__).with_name('fixtures').joinpath('valve_release_input.json').read_text())
    spec=ValveSpec(**saved['request']);p=ValvePlant(1,[],spec).control
    surfaces=list(map(tuple,saved['surfaces']))
    q,_=endpoint_load_envelope(p,spec,np.asarray(saved['q']),surfaces)
    result,trim=refine(p,spec,q,surfaces,free_endpoint=True)
    assert trim['status']=='static_screen_pass'
    flight=static_balance(p.pin,p.meta,result,payload_fraction=0.,direct_burden=True,burden_power=2.)
    assert flight['status']=='static_screen_pass'
    # Independently check the returned physical thrust, without requiring
    # convergence of a different power-cone objective at the same pose.
    assert check_fixed_thrust(p.pin,p.meta,result,flight['thrust'])['status']=='static_screen_pass'


def test_shared_direction_refinement_lowers_cost_with_both_loads_feasible():
    from amsrr_det.valve_direction_refinement import refine
    from amsrr_det.valve_trim import static_balance
    from amsrr_det.structure_bounds import hover_input_burden
    saved=json.loads(Path(__file__).with_name('fixtures').joinpath('valve_force_gradient_input.json').read_text())
    spec=ValveSpec(**saved['request']);p=ValvePlant(saved['modules'],saved['edges'],spec).control
    q=np.asarray(saved['q']);surfaces=list(map(tuple,saved['surfaces']))
    contacts=material_force_contacts(p,spec.shape,q,surfaces,force_limit=None,material_tangents=True)
    def costs(pose):
        values=[]
        for factor in [.9,1.1]:
            row=static_balance(p.pin,p.meta,pose,contacts,payload_fraction=0.,
                object_support=bearing_support(spec),direct_burden=True,minimum_pad_normal_force_n=.1,
                object_wrench=np.r_[np.zeros(3),np.asarray(spec.axis)*spec.torque*factor])
            assert row['status']=='static_screen_pass'
            values.append(hover_input_burden(row['thrust']))
        return max(values)
    before=costs(q);moved,report=refine(p,spec,q,surfaces)
    after=costs(moved)
    assert after < before-.01
    np.testing.assert_allclose(after,hover_input_burden(report['thrust']),atol=1e-7)
    qi=[p.meta['actuators'][r['gimbal']]['q'] for r in p.meta['rotors']]
    np.testing.assert_array_equal(moved[np.setdiff1d(np.arange(len(q)),qi)],q[np.setdiff1d(np.arange(len(q)),qi)])


def test_simultaneous_direction_difference_matches_individual_rotors():
    from amsrr_det.rotor_distribution import rotor_actions
    saved=json.loads(Path(__file__).with_name('fixtures').joinpath('valve_force_gradient_input.json').read_text())
    spec=ValveSpec(**saved['request']);p=ValvePlant(saved['modules'],saved['edges'],spec).control
    q=np.asarray(saved['q']);qi=[p.meta['actuators'][r['gimbal']]['q'] for r in p.meta['rotors']]
    plus=q.copy();minus=q.copy();plus[qi]+=1e-6;minus[qi]-=1e-6
    simultaneous=(rotor_actions(p.pin,p.meta,plus)-rotor_actions(p.pin,p.meta,minus))/2e-6
    for i,index in enumerate(qi):
        plus=q.copy();minus=q.copy();plus[index]+=1e-6;minus[index]-=1e-6
        independent=(rotor_actions(p.pin,p.meta,plus)-rotor_actions(p.pin,p.meta,minus))/2e-6
        np.testing.assert_allclose(simultaneous[:,i],independent[:,i],atol=1e-9)
        np.testing.assert_allclose(np.delete(independent,i,axis=1),0.,atol=1e-9)


def test_partial_rigid_support_poses_preserve_contacts_for_nonvertical_axes():
    import pinocchio as pin
    from amsrr_det.valve_plan import rotate_assembly
    from amsrr_det.grasp_geometry import grasp_frame
    saved=json.loads(Path(__file__).with_name('fixtures').joinpath('valve_yaw_symmetry_input.json').read_text())
    original=ValveSpec(**saved['request']);plant=ValvePlant(saved['modules'],saved['edges'],original).control
    surfaces=list(map(tuple,saved['surfaces']));oq=plant.meta['object']['q'];data=plant.pin.createData()
    for axis in [(1.,0.,0.),(.6,0.,.8),(0.,0.,-1.)]:
        spec=ValveSpec(axis=axis,torque=original.torque);q=np.asarray(saved['q']).copy()
        R=spec.rotation@pin.Quaternion(q[oq+3:oq+7]).matrix().T
        q[:3]=np.asarray(spec.center)+R@(q[:3]-original.center)
        q[3:7]=pin.Quaternion(R@pin.Quaternion(q[3:7]).matrix()).coeffs()
        pose=spec.pose();q[oq:oq+7]=np.r_[pose[:3],pose[4:],pose[3]]
        relative=[];gravity=[]
        for angle in [0.,np.pi/2,np.pi]:
            moved=rotate_assembly(plant,spec,q,angle,0.)
            np.testing.assert_array_equal(moved[7:oq],q[7:oq])
            pin.framesForwardKinematics(plant.pin,data,moved)
            object_frame=data.oMf[plant.meta['object']['frame']]
            relative.append([(object_frame.inverse()*data.oMf[plant.meta['frames'][grasp_frame(*s)]]).homogeneous.copy() for s in surfaces])
            gravity.append(pin.computeGeneralizedGravity(plant.pin,data,moved).copy())
        np.testing.assert_allclose(relative,np.broadcast_to(relative[0],np.shape(relative)),atol=1e-12)
        if abs(axis[2])<1.:assert np.linalg.norm(gravity[0]-gravity[-1])>1.


def test_rigid_valve_execution_does_not_reproject_screened_contact_depths():
    from amsrr_det.configuration_path import AxisRotationPath
    from scripts.run_valve_task import ValveReferences
    saved=json.loads(Path(__file__).with_name('fixtures').joinpath('valve_multi_axis_symmetry_input.json').read_text())
    spec=ValveSpec(**saved['request']);system=ValvePlant(saved['modules'],saved['edges'],spec)
    p=system.control;q=p.configuration(np.asarray(saved['q']))
    curve=AxisRotationPath(p,[0.,np.pi],q,spec.axis,spec.center)
    path=np.array([q,curve.pose(np.pi)])
    reference=ValveReferences(system,q,q,np.array([0.,np.pi]),path,path[-1],curve_record=curve.record())
    actual=curve.pose(.2);args,settings=reference.update(5.5,actual,np.zeros(p.mj.nv),.2)
    np.testing.assert_allclose(args[2],actual,atol=1e-13)
    assert settings['shape_contact_depth'] is None
    assert settings['track_object']
