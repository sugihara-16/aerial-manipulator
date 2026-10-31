"""Check force/angle derivatives against independent inverse dynamics."""
import numpy as np
import pinocchio as pin
import pytest
from amsrr_det.model import build_model
from amsrr_det.free_flight_trim import _free_flight_objective

@pytest.mark.parametrize('n,edges',[(1,[]),(2,[]),(2,[dict(a=0,port_a='pitch1',b=1,port_b='yaw2')])])
def test_free_trim_jacobian_matches_finite_difference(n,edges):
    m,meta=build_model(n,edges)
    q=pin.neutral(m);rng=np.random.default_rng(922)
    for a in meta['actuators']:q[a['q']]=rng.uniform(-.5,.5)
    nr=len(meta['rotors']);x=np.r_[rng.uniform(1.,10.,nr),[q[meta['actuators'][r['gimbal']]['q']] for r in meta['rotors']]]
    fn=_free_flight_objective(m,meta,q,x)
    actual=fn(x)[1].copy();expected=np.empty_like(actual)
    for i in range(len(x)):
        d=np.zeros(len(x));d[i]=1e-6
        expected[:,i]=(fn(x+d)[0]-fn(x-d)[0])/2e-6
    np.testing.assert_allclose(actual,expected,atol=2e-8,rtol=2e-7)
