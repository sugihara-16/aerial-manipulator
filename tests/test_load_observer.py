import mujoco
import numpy as np
from amsrr_det.mujoco_model import make_plant
from amsrr_det.load_observer import JointLoadObserver


def test_motion_and_past_drive_recover_unmeasured_joint_load():
    plant=make_plant(meshes=False,timestep=.0002)
    plant.mj.opt.gravity[:]=0;plant.pin.gravity.linear=np.zeros(3)
    plant.mj.opt.integrator=mujoco.mjtIntegrator.mjINT_DISCRETE
    data=mujoco.MjData(plant.mj)
    selected=next(a for a in plant.meta['actuators'] if a['role']=='dock')
    dof=plant.mj.jnt_dofadr[plant.mj.joint(selected['name']).id]
    # Independent plant-only test load. The observer receives neither this
    # applied force nor any simulator acceleration/force result.
    data.qfrc_applied[dof]=.15
    observer=JointLoadObserver(plant.pin,plant.meta,period=.01,filter_time=0)
    command=np.zeros(len(plant.meta['actuators']));forces=np.zeros(len(plant.meta['rotors']))
    estimates=[]
    for _ in range(30):
        q,v=plant.observation(data);external=observer.update(q,v,command,forces)
        if external is not None:estimates.append(external[selected['v']])
        mujoco.mj_step(plant.mj,data,nstep=50)
    np.testing.assert_allclose(estimates[-10:],.15,atol=.003)
