import json
from pathlib import Path
import numpy as np
from amsrr_det.model import build_model
from amsrr_det.equilibrium import static_balance,refine_gimbal_trim


def fixture():
    z=json.loads((Path(__file__).parent/'fixtures/heavy_gimbal_trim.json').read_text())
    model,meta=build_model(z['modules'],z['edges'],z['payload'])
    return model,meta,np.array(z['q']),z['contacts']


def test_force_program_envelope_gradient_matches_independent_resolves():
    model,meta,q,contacts=fixture()
    result=static_balance(model,meta,q,contacts,minimum_pad_normal_force_n=.5,gimbal_gradient=True)
    indices=[meta['actuators'][r['gimbal']]['q']for r in meta['rotors']]
    for seed in range(3):
        direction=np.random.default_rng(seed).normal(size=len(indices));direction/=np.linalg.norm(direction)
        values=[]
        for sign in [1.,-1.]:
            trial=q.copy();trial[indices]+=sign*1e-6*direction
            values.append(static_balance(model,meta,trial,contacts,minimum_pad_normal_force_n=.5)['minimum_peak_effort_ratio'])
        np.testing.assert_allclose(np.dot(result['gimbal_peak_gradient'],direction),(values[0]-values[1])/2e-6,atol=2e-6)


def test_task_load_direction_repair_keeps_geometry_and_satisfies_original_limits():
    model,meta,q,contacts=fixture()
    before=static_balance(model,meta,q,contacts,minimum_pad_normal_force_n=.5)
    changed,after=refine_gimbal_trim(model,meta,q,contacts,before)
    assert before['minimum_peak_effort_ratio']>4.
    assert after['minimum_peak_effort_ratio']<1.
    assert after['status']=='static_screen_pass'
    ids=[meta['actuators'][r['gimbal']]['q']for r in meta['rotors']]
    np.testing.assert_array_equal(changed[np.setdiff1d(np.arange(len(q)),ids)],q[np.setdiff1d(np.arange(len(q)),ids)])
    recheck=static_balance(model,meta,changed,contacts,minimum_pad_normal_force_n=.5)
    assert recheck['status']=='static_screen_pass'
