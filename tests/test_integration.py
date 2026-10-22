import numpy as np
import pinocchio as pin
from amsrr_det.model import export_model
from amsrr_det.reference import integrate_reference
from test_physics import execute


def test_implicit_drive_converges_to_independent_radau(tmp_path):
    model,meta=export_model(tmp_path);q=pin.neutral(model)
    command={'position':[.015 if a['name']=='m0/gimbal1' else 0. for a in meta['actuators']]}
    scenario=dict(q0=q.tolist(),duration_s=.1,command=command,integrator='drive_implicit_euler',
                  thrust_lag_s=0.,gimbal_torque_lag_s=0.,dock_torque_lag_s=0.)
    ref=integrate_reference(model,meta,scenario,command,.1)
    errors=[]
    for h in [.002,.001,.0005]:
        r=execute(tmp_path,model,meta,scenario,h)
        assert r['full_horizon']
        error=np.linalg.norm(pin.difference(model,ref['q'],np.array(r['final']['q'])))
        errors.append(error)
    # Includes the first-order freefall position error: must decrease as h.
    assert errors[1]<.6*errors[0]
    assert errors[2]<.6*errors[1]
    assert errors[2]<3e-4


def test_lagged_drive_step_satisfies_discrete_force_balance(tmp_path):
    model,meta=export_model(tmp_path);q=pin.neutral(model);v=np.zeros(model.nv)
    command={'position':[.01]*len(meta['actuators'])}
    h=.001
    sc=dict(q0=q.tolist(),duration_s=h,command=command,integrator='drive_implicit_euler',
            thrust_lag_s=.03,gimbal_torque_lag_s=.01,dock_torque_lag_s=.0022)
    r=execute(tmp_path,model,meta,sc,h)
    assert r['full_horizon']
    vn=np.array(r['final']['v']);qn=np.array(r['final']['q']);tau=np.array(r['final']['tau'])
    required=pin.rnea(model,model.createData(),q,v,(vn-v)/h)
    applied=np.zeros(model.nv)
    for i,a in enumerate(meta['actuators']):
        applied[a['v']]=tau[i]
        desired=np.clip(a['kp']*(.01-qn[a['q']])-a['kd']*vn[a['v']],-a['effort'],a['effort'])
        lag=.01 if a['role']=='vectoring' else .0022
        np.testing.assert_allclose(tau[i],h/(h+lag)*desired,atol=1e-12)
    np.testing.assert_allclose(required,applied,atol=1e-11)
