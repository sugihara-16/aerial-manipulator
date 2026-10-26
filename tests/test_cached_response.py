"""A narrow nonlinear response needs finer secants, without altering predictions."""
from types import MethodType, SimpleNamespace
import mujoco
import numpy as np
from build.cached_response import CachedResponse


def predictor():
    m=mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="pad"><joint name="j"/><geom type="sphere" size=".01"/></body>
      <body name="object" pos="1 0 0"><geom name="object" type="sphere" size=".01"/></body>
      </worldbody><actuator><motor joint="j"/></actuator></mujoco>''')
    p=CachedResponse(SimpleNamespace(mj=m,meta={'actuators':[{'effort':5.}], 'rotors':[]}),
                     ['pad'],steps=1,workers=1)
    def response(self,args):
        q,v,u=args
        if not hasattr(self.thread,'data'):self.thread.data=mujoco.MjData(self.m)
        velocity=np.array([.01*np.tanh(u[0]/1e-4)])
        return dict(q=q.copy(),v=velocity,states=np.array([[0.,q[0],velocity[0]]]),
                    loads=np.zeros((1,1,7)),motors=np.zeros((1,2,1)))
    p._response=MethodType(response,p)
    return p


def test_inaccurate_secant_refines_and_improves_prediction_without_changing_response():
    p=predictor()
    try:
        q=v=u=np.zeros(1);trial=np.array([1e-6])
        before=p.evaluate(q,v,u,True)
        exact=p.evaluate(q,v,trial)
        assert p.counts['step_refinements']==1
        assert p.cached is None
        after=p.evaluate(q,v,u,True)
        checked=p.evaluate(q,v,trial)
        old_error=abs((before['derivatives'][-1,1]@trial)-exact['v'][0])
        new_error=abs((after['derivatives'][-1,1]@trial)-exact['v'][0])
        assert new_error < .1*old_error
        np.testing.assert_array_equal(checked['v'],exact['v'])
        assert p.counts['step_refinements']==1
    finally:p.close()


def test_no_input_change_keeps_derivative_and_step():
    p=predictor()
    try:
        q=v=u=np.zeros(1);step=p.epsilon.copy()
        p.evaluate(q,v,u,True);p.evaluate(q,v,u)
        np.testing.assert_array_equal(p.epsilon,step)
        assert p.cached is not None
        assert p.counts['inaccurate']==0
    finally:p.close()


def test_new_problem_does_not_inherit_refined_contact_secants():
    p=predictor()
    try:
        q=v=u=np.zeros(1);step=p.epsilon.copy()
        p.evaluate(q,v,u,True);p.evaluate(q,v,np.array([1e-6]))
        assert np.all(p.epsilon<step)
        p.evaluate(q,v,u,True)
        p.start_problem()
        np.testing.assert_array_equal(p.epsilon,step)
        assert p.cached is None
        p.evaluate(q,v,u,True)
        cached=p.cached
        p.start_problem()
        assert p.cached is cached
    finally:p.close()


def test_repeated_nonlinear_disagreement_stops_refining_at_roundoff_floor():
    p=predictor()
    try:
        q=v=u=np.zeros(1)
        for _ in range(8):
            p.evaluate(q,v,u,True);p.evaluate(q,v,np.array([.1]))
        np.testing.assert_array_equal(p.epsilon,p.minimum_step)
        assert p.counts['inaccurate']==8
        assert p.counts['step_refinements']==2
    finally:p.close()
