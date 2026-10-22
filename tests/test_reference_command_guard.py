import numpy as np
import pinocchio as pin
from amsrr_det.model import build_model
from amsrr_det.grasp_geometry import grasp_frame
from amsrr_det.grasp_control import GraspReferences


def fixture():
 model,meta=build_model(1,payload=dict(size=[.3,.3,.3],mass=.6));q=pin.neutral(model);q[2]=q[meta['object']['q']+2]=1.
 data=model.createData();pin.framesForwardKinematics(model,data,q);frame=meta['frames'][grasp_frame(0,'yaw1')];obj=data.oMf[meta['object']['frame']]
 grasp=GraspReferences(model,meta,q,q,[(0,'yaw1')],[obj.actInv(data.oMf[frame].translation)],[obj.rotation.T@data.oMf[frame].rotation[:,0]],samples=[(frame,np.zeros((1,3)))])
 return model,meta,q,grasp


def test_untrackable_reference_is_not_reset_when_bounds_conflict():
 model,meta,q,g=fixture();v=np.zeros(model.nv);bias=np.zeros(len(g.docks));command=q[g.qi].copy()
 a=g.docks[0];g.last_velocity[6]=.2
 # Current issued command is exactly at the source effort limit. The stored
 # reference and old velocity reproduce that same command, with zero bias.
 command[0]+=a['effort']/a['kp'];g.reference[g.qi[0]]=command[0]-a['kd']*.2/a['kp']
 before=g.reference.copy();target,_=g.update(q,v,q[meta['object']['q']:],drive_guard=(bias,command))
 assert not g.last_diagnostics['reference_command_feasible']
 np.testing.assert_array_equal(target,before)
 np.testing.assert_array_equal(g.reference,before)


def test_guard_bounds_the_issued_command_including_velocity_compensation():
 model,meta,q,g=fixture();v=np.zeros(model.nv);bias=np.zeros(len(g.docks));command=q[g.qi].copy();a=g.docks[0]
 g.reference[g.qi[0]]+=a['effort']/a['kp']
 command[0]=g.reference[g.qi[0]]
 target,tv=g.update(q,v,q[meta['object']['q']:],drive_guard=(bias,command))
 assert g.last_diagnostics['reference_command_feasible']
 kp=np.array([a['kp'] for a in g.docks]);kd=np.array([a['kd'] for a in g.docks]);limits=np.array([a['effort'] for a in g.docks])
 issued=target[g.qi]+(bias+kd*tv[g.vi])/kp
 actual_raw=kp*(issued-q[g.qi])-kd*v[g.vi]
 np.testing.assert_allclose(actual_raw,g.last_diagnostics['predicted_drive_effort_nm'],atol=1e-12)
 assert np.all(abs(actual_raw)<=limits+1e-9)


def test_selective_braking_preserves_physical_command_bounds_without_reset():
 model,meta,q,g=fixture();v=np.zeros(model.nv);bias=np.zeros(len(g.docks));command=q[g.qi].copy()
 a=g.docks[0];g.last_velocity[6]=.2
 command[0]+=a['effort']/a['kp'];g.reference[g.qi[0]]=command[0]-a['kd']*.2/a['kp']
 before=g.reference.copy()
 target,tv=g.update(q,v,q[meta['object']['q']:],drive_guard=(bias,command),allow_guard_braking=True,previous_joint_bias=bias)
 assert g.last_diagnostics['reference_command_feasible']
 assert g.last_diagnostics['reference_acceleration_relaxed_docks']==[a['name']]
 np.testing.assert_allclose(target[g.qi]-before[g.qi],tv[g.vi]*.01,atol=1e-14)
 kp=np.array([a['kp'] for a in g.docks]);kd=np.array([a['kd'] for a in g.docks])
 issued=target[g.qi]+(bias+kd*tv[g.vi])/kp
 assert np.max(abs(issued-command))<=.002125+1e-12
 assert abs(kp[0]*(issued[0]-q[g.qi[0]]))<=a['effort']+1e-12


def test_braking_does_not_relax_conflicting_position_speed_and_drive_bounds():
 model,meta,q,g=fixture();v=np.zeros(model.nv);bias=np.zeros(len(g.docks));command=q[g.qi].copy()
 g.reference[g.qi[0]]=g.docks[0]['upper']-.03;before=g.reference.copy()
 target,tv=g.update(q,v,q[meta['object']['q']:],drive_guard=(bias,command),allow_guard_braking=True,previous_joint_bias=bias)
 assert not g.last_diagnostics['reference_command_feasible']
 np.testing.assert_array_equal(target,before)
