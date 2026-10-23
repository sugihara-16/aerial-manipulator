"""Storage selection must not change the physical input/contact derivative."""
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from build.contact_allocation import ContactAllocation
from build.contact_tangent import local_input_tangent


@pytest.mark.parametrize('nv', [59, 60, 76])
def test_contact_response_matches_dense_and_sparse_storage(nv):
    # An active sphere/plane contact plus independent coordinates straddles
    # MuJoCo's automatic storage threshold, reached by the six-module robot.
    extra=''.join(f'<body pos="{i+1} 0 2"><joint damping="1"/>'
                  '<inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/></body>'
                  for i in range(nv-1))
    xml=f'''<mujoco><option integrator="discrete" solver="Newton"
                tolerance="1e-12" cone="elliptic"/>
      <worldbody><geom name="pad/floor" type="plane" size="1 1 .1"/>
        <body pos="0 0 .099"><joint name="height" type="slide" axis="0 0 1"/>
          <geom name="object" type="sphere" size=".1" mass="1"/></body>{extra}
      </worldbody><actuator><general joint="height" gainprm="1"
        biastype="affine" biasprm="0 0 -1"/></actuator></mujoco>'''
    model=mujoco.MjModel.from_xml_string(xml)
    assert model.nv==nv
    assert bool(mujoco.mj_isSparse(model))==(nv>=60)
    responses=[]
    for layout in [mujoco.mjtJacobian.mjJAC_DENSE, mujoco.mjtJacobian.mjJAC_SPARSE]:
        model.opt.jacobian=layout
        data=mujoco.MjData(model);data.ctrl[:]=5.
        mujoco.mj_forward(model,data)
        assert data.nefc>0 and data.efm_active
        tangent=local_input_tangent(model,data)
        mass=np.empty((nv,nv));mujoco.mj_fullM(model,data,mass)
        local=SimpleNamespace(m=model,nj=1,links=['pad'],qi=np.array([0]),
            vi=np.array([0]),kp=np.array([0.]),kd=np.array([1.]),
            gain=np.array([1.]),effort=np.array([50.]),rotor_min=np.empty(0),
            rotor_max=np.empty(0))
        response=ContactAllocation(local).linearize(data.qpos,data.qvel,mass,
            tangent['moment'].T,np.array([0]),np.ones(1),data.ctrl)
        responses.append((tangent,response))
    for key in ['mass','jacobian','contact_derivative',
                'acceleration_wrt_control','force_wrt_control']:
        np.testing.assert_allclose(responses[0][0][key],responses[1][0][key],
                                   rtol=1e-10,atol=1e-10)
    for dense,sparse in zip(responses[0][1],responses[1][1]):
        np.testing.assert_allclose(dense,sparse,rtol=1e-10,atol=1e-10)
