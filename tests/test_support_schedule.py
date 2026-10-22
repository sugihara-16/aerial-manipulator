import json
from pathlib import Path
import numpy as np
import pinocchio as pin
from amsrr_det.model import build_model
from amsrr_det.free_flight_trim import check_fixed_thrust,support_schedule,scheduled_support,refine_support_path


def test_feasible_endpoints_do_not_accept_overloaded_interpolated_arm():
    case=json.loads((Path(__file__).parent/'fixtures/morph_force_interpolation.json').read_text())
    model,meta=build_model(case['modules'],case['edges'],case['payload'])
    first,last=map(np.array,case['poses']);fa,fb=map(np.array,case['forces'])
    a,b=[check_fixed_thrust(model,meta,q,f) for q,f in [(first,fa),(last,fb)]]
    assert a['status']==b['status']=='static_screen_pass'
    midpoint=pin.interpolate(model,first,last,.4)
    assert check_fixed_thrust(model,meta,midpoint,.6*fa+.4*fb)['max_normalized_joint_effort']>1.4
    schedule=support_schedule(model,meta,first,last,a,b)
    np.testing.assert_array_equal(scheduled_support(schedule,0)[1],fa)
    np.testing.assert_array_equal(scheduled_support(schedule,1)[1],fb)
    for s in np.linspace(0,1,101):
        q=pin.interpolate(model,first,last,float(s));bias,f=scheduled_support(schedule,s)
        actual=check_fixed_thrust(model,meta,q,f)
        assert actual['max_normalized_joint_effort']<=1.+1e-7
        assert actual['equilibrium_residual']<.021
        assert max(abs(j['effort_nm']-t)/j['limit_nm'] for j,t in zip(actual['joints'],bias))<.021


def test_unsupported_intermediate_direction_is_repaired_without_moving_docks():
    import pytest
    case=json.loads((Path(__file__).parent/'fixtures/opening_direction_interpolation.json').read_text())
    model,meta=build_model(case['modules'],case['edges'],case['payload'])
    first,last=map(np.array,case['poses'])
    trims=[check_fixed_thrust(model,meta,q,f) for q,f in zip([first,last],case['forces'])]
    assert all(t['status']=='static_screen_pass' for t in trims)
    with pytest.raises(ValueError,match='morph_intermediate_support_not_found'):
        support_schedule(model,meta,first,last,*trims)
    path,distances,schedules,repairs=refine_support_path(model,meta,[first,last],[0.,1.],trims)
    assert repairs and len(path)==len(schedules)+1
    retained=[*range(7),*[a['q'] for a in meta['actuators'] if a['role']=='dock']]
    for q,s in zip(path,distances):
        original=pin.interpolate(model,first,last,float(s))
        np.testing.assert_allclose(q[retained],original[retained],atol=1e-12)
    for a,b,schedule in zip(path[:-1],path[1:],schedules):
        for s in np.linspace(0,1,41):
            q=pin.interpolate(model,a,b,float(s));bias,force=scheduled_support(schedule,s)
            actual=check_fixed_thrust(model,meta,q,force)
            assert actual['max_normalized_joint_effort']<=1.+1e-7
            assert actual['equilibrium_residual']<.021
            assert max(abs(j['effort_nm']-t)/j['limit_nm'] for j,t in zip(actual['joints'],bias))<.021
