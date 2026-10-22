import numpy as np
import pinocchio as pin
from amsrr_det.model import build_model
from amsrr_det.grasp_control import surface_points
from amsrr_det.rigid_grasp_reference import RigidGraspReferences


def fixture():
    model,meta=build_model(1,payload=dict(size=[.3,.3,.3],mass=.6))
    q=pin.neutral(model);q[2]=1.;oq=meta['object']['q'];q[oq+2]=.4
    samples=[surface_points(model,meta,q,(0,'yaw1'))]
    refs=RigidGraspReferences(model,meta,q,q,[(0,'yaw1')],[[0,0,0]],[[1,0,0]],samples=samples)
    params=dict(base_pose=q[:7],object_pose=q[oq:oq+7],object_goal=q[oq:oq+7],
                base_world_velocity=np.zeros(6),object_world_velocity=np.zeros(6),
                joint_reference=q[refs.qi]+.003)
    return model,meta,q,refs,params


def test_bridge_keeps_entry_velocity_and_reaches_fixed_reference():
    model,meta,q,refs,params=fixture();refs.last_velocity[6:]=.001
    refs.configure_operating_point(params);v=np.zeros(model.nv);goal=params['object_goal']
    initial,initial_rate=refs.update(q,v,goal)
    np.testing.assert_array_equal(initial[refs.qi],q[refs.qi])
    np.testing.assert_allclose(initial_rate[refs.vi],.001,atol=1e-14)
    for _ in range(200):target,rate=refs.update(q,v,goal)
    np.testing.assert_allclose(target[refs.qi],params['joint_reference'],atol=1e-14)
    np.testing.assert_allclose(rate[refs.vi],0.,atol=1e-14)


def test_object_feedback_changes_base_without_changing_joint_target_or_observation():
    model,meta,q,refs,params=fixture();refs.configure_operating_point(params);refs.elapsed=2.
    goal=params['object_goal'].copy();v=np.zeros(model.nv)
    refs.update(q,v,goal)
    observed=q.copy();observed[refs.qi]+=.001
    goal[2]+=.01
    target,rate=refs.update(observed,v,goal)
    assert rate[2]>0.
    np.testing.assert_allclose(target[refs.qi],params['joint_reference'],atol=1e-14)
    np.testing.assert_allclose(rate[refs.vi],0.,atol=1e-14)
    np.testing.assert_array_equal(observed[refs.qi],q[refs.qi]+.001)
