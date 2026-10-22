import numpy as np
import pinocchio as pin
from amsrr_det.model import export_model
from amsrr_det.reference import integrate_reference
from test_physics import execute


def object_landing(model,meta):
    q=pin.neutral(model);q[2]=1.;q[meta['object']['q']+2]=.075
    v=np.zeros(model.nv);v[meta['object']['v']+2]=-.05
    patch=dict(frame='object',points=[[x,y,-.075] for x,y in [(-.15,-.2),(-.15,.2),(.15,-.2),(.15,.2)]],
               normal=[0.,0.,1.],offset=0.,kn=7500.,cn=75.,kt=7500.,ct=1.,mu=0.,
               lower=[-10.]*3,upper=[10.]*3)
    return dict(q0=q.tolist(),v0=v.tolist(),duration_s=.25,patches=[patch],integrator='drive_implicit_euler',
                thrust_lag_s=0.,gimbal_torque_lag_s=0.,dock_torque_lag_s=0.,max_penetration_m=.002)


def test_soft_contact_step_converges_to_independent_radau(tmp_path):
    model,meta=export_model(tmp_path,payload=dict(mass=1.,size=[.3,.4,.15]))
    scenario=object_landing(model,meta)
    ref=integrate_reference(model,meta,scenario,{},scenario['duration_s'])
    iz,iv=meta['object']['q']+2,meta['object']['v']+2
    ref_penetration=float(max(.075-ref['trajectory'][iz]))
    errors=[]
    for h in [.002,.001,.0005]:
        r=execute(tmp_path,model,meta,scenario,h)
        assert r['full_horizon']
        errors.append(abs(r['metrics']['peak_penetration_m']-ref_penetration))
        assert abs(r['final']['q'][iz]-ref['q'][iz])<1e-6
        assert abs(r['final']['v'][iv]-ref['v'][iv])<2e-5
    assert errors[-1]<2e-5
    assert errors[-1]<.5*errors[0]
