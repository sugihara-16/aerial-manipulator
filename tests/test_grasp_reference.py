"""A task-local warm start preserves geometry, not the old world pose."""
import mujoco
import numpy as np
from build.generated_grasp_stage import align_reference_to_object


def test_reference_alignment_preserves_object_relative_geometry_and_input():
    model=mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="robot" pos=".2 -.3 .7"><freejoint/><geom size=".1"/>
      <body pos=".1 .2 .3"><joint name="joint"/><geom size=".05"/></body></body>
      <body name="payload" pos=".1 .1 .5"><freejoint name="object"/><geom size=".1"/></body>
    </worldbody></mujoco>''')
    original=model.qpos0.copy();original[7]=.31
    observation=original.copy();oi=int(model.jnt_qposadr[model.joint('object').id])
    observation[oi:oi+3]+=[.3,-.4,.2]
    observation[oi+3:oi+7]=[np.cos(.3),0,np.sin(.3),0]
    untouched=original.copy();aligned=align_reference_to_object(model,original,observation)
    np.testing.assert_array_equal(original,untouched)
    np.testing.assert_allclose(aligned[oi:],observation[oi:],atol=1e-14)
    assert aligned[7]==original[7]
    relative=[]
    for pose in [original,aligned]:
        data=mujoco.MjData(model);data.qpos[:]=pose;mujoco.mj_kinematics(model,data)
        obj=model.body('payload').id;R=data.xmat[obj].reshape(3,3)
        relative.append(((data.xpos-data.xpos[obj])@R,np.einsum('ij,bjk->bik',R.T,data.xmat.reshape(-1,3,3))))
    # The fixed world body is intentionally not carried by the rigid transform.
    for before,after in zip(*relative):np.testing.assert_allclose(before[1:],after[1:],atol=1e-14)
