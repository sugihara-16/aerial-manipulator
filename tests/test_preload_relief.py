from copy import copy
import numpy as np
import pinocchio as pin
from amsrr_det.model import build_model
from amsrr_det.grasp_control import GraspReferences
from amsrr_det.grasp_geometry import grasp_frame
from amsrr_det.preload_relief import update_with_preload_relief


def fixture():
    model,meta=build_model(1,payload=dict(size=[.3,.3,.3],mass=.6))
    q=pin.neutral(model);q[2]=q[meta['object']['q']+2]=1.
    data=model.createData();pin.framesForwardKinematics(model,data,q);obj=data.oMf[meta['object']['frame']]
    surfaces=[(0,'yaw1'),(0,'yaw2'),(0,'pitch1')]
    frames=[meta['frames'][grasp_frame(*surface)] for surface in surfaces]
    refs=GraspReferences(model,meta,q,q,surfaces,
        [obj.actInv(data.oMf[f].translation) for f in frames],
        [obj.rotation.T@data.oMf[f].rotation[:,0] for f in frames],
        samples=[(f,np.zeros((1,3))) for f in frames])
    refs.preload[:]=.00015
    return q,np.zeros(model.nv),q[meta['object']['q']:].copy(),refs


def test_without_excess_request_preserves_original_controller_exactly():
    q,v,goal,g=fixture();original=copy(g);bias=np.zeros(4);command=q[g.qi].copy()
    expected=original.update(q,v,goal,drive_guard=(bias,command))
    actual=update_with_preload_relief(g,q,v,goal,drive_guard=(bias,command),previous_joint_bias=bias)
    for a,b in zip(actual,expected):np.testing.assert_array_equal(a,b)
    np.testing.assert_array_equal(g.preload,np.full(3,.00015))
    assert not g.last_diagnostics['preload_relief']['changed']


def test_relief_reduces_final_command_request_and_integrates_reference_only_once():
    q,v,goal,g=fixture();before=g.reference.copy();preload=g.preload.copy()
    bias=np.array([0.,-2.,0.,0.]);command=q[g.qi]+bias/200
    target,velocity=update_with_preload_relief(g,q,v,goal,drive_guard=(bias,command),previous_joint_bias=bias)
    report=g.last_diagnostics['preload_relief']
    assert report['changed'] and report['error_after']<report['error_before']
    assert np.all(g.preload<=preload) and np.all(g.preload>=preload-.001*.01-1e-15)
    np.testing.assert_allclose(target[g.qi]-before[g.qi],.01*velocity[g.vi],atol=1e-15)
    np.testing.assert_array_equal(q,before)
    issued=target[g.qi]+(bias+5*velocity[g.vi])/200
    actual=200*(issued-q[g.qi])-5*v[g.vi]
    np.testing.assert_allclose(actual,report['request_after_nm'],atol=1e-12)
    assert report['command_bounds_rechecked']
    assert max(abs(issued-command))<=.002125+1e-12


def test_conflicting_drive_bounds_do_not_reset_or_relax_the_reference():
    q,v,goal,g=fixture();bias=np.zeros(4);command=q[g.qi].copy()
    g.last_velocity[6]=.2
    command[0]+=1.3/200;g.reference[g.qi[0]]=command[0]-5*.2/200
    before=g.reference.copy();preload=g.preload.copy()
    target,_=update_with_preload_relief(g,q,v,goal,drive_guard=(bias,command),previous_joint_bias=bias)
    assert not g.last_diagnostics['reference_command_feasible']
    np.testing.assert_array_equal(target,before);np.testing.assert_array_equal(g.preload,preload)
