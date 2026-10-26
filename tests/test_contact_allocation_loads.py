"""Batched private contact loads agree with individual engine contact forces."""
from types import SimpleNamespace
import mujoco
import numpy as np
from build.contact_allocation import ContactAllocation


def test_multiple_contacts_and_separation_match_individual_forces():
    m=mujoco.MjModel.from_xml_string('''<mujoco><option cone="elliptic" gravity="0 0 0"/>
      <worldbody>
        <body name="object"><freejoint name="object"/><geom name="object" type="box" size=".1 .1 .1"/></body>
        <body name="a" pos=".195 0 0"><freejoint/><geom name="a/pad" type="box" size=".1 .08 .08"/></body>
        <body name="b" pos="-.195 0 0"><freejoint/><geom name="b/pad" type="box" size=".1 .08 .08"/></body>
      </worldbody></mujoco>''')
    control=ContactAllocation(SimpleNamespace(m=m,links=['a','b']))
    for displacement in [0., .002, 1.]:
        d=control.d;d.qpos[:]=m.qpos0;d.qpos[0]+=displacement
        _, info=control.check(np.zeros(0),np.zeros(m.nv))
        normals=np.zeros(2);forces=np.zeros((2,3));moments=forces.copy()
        for i,c in enumerate(d.contact):
            if c.efc_address<0 or control.object not in c.geom:continue
            other=int(c.geom[1] if c.geom[0]==control.object else c.geom[0])
            pad=control.pads.get(other)
            if pad is None:continue
            local=np.zeros(6);mujoco.mj_contactForce(m,d,i,local)
            world=c.frame.reshape(3,3).T@local[:3]
            normals[pad]+=local[0];forces[pad]+=world
            moments[pad]+=np.cross(c.pos-d.xpos[m.geom_bodyid[other]],world)
        np.testing.assert_allclose(info['minimum_model_normal_n'],normals.min(),atol=1e-12)
        np.testing.assert_allclose(info['maximum_model_force_n'],np.linalg.norm(forces,axis=1).max(),atol=1e-12)
        np.testing.assert_allclose(info['maximum_model_moment_nm'],np.linalg.norm(moments,axis=1).max(),atol=1e-12)
def test_only_zero_unilateral_response_loses_its_instantaneous_floor():
    import numpy as np
    from build.contact_allocation import instantaneous_normal_floors
    response=np.array([[0.,0.],[1.,0.],[0.,0.],[0.,0.]])
    constant=np.array([0.,0.,.1,.5])
    np.testing.assert_array_equal(instantaneous_normal_floors(response,constant,.5),[0.,.5,.5,.5])



def test_linearization_uses_current_fixed_servo_reference():
    """An input excluded from allocation cannot retain the previous reference."""
    m=mujoco.MjModel.from_xml_string('''<mujoco>
      <option integrator="discrete" gravity="0 0 0"/>
      <worldbody><body name="root">
        <joint name="dock" axis="0 0 1" damping="1"/>
        <geom name="object" type="box" size=".1 .1 .1"/>
        <body name="pad" pos=".25 0 0">
          <joint name="gimbal" axis="0 1 0" damping="1"/>
          <geom name="pad/geom" type="box" size=".1 .08 .08"/>
        </body>
      </body></worldbody>
      <actuator><general joint="dock" gainprm="1" biastype="affine" biasprm="0 -1 -1"/>
        <position joint="gimbal" kp="20" kv="1"/></actuator>
    </mujoco>''')
    local=SimpleNamespace(m=m,links=['pad'],qi=np.array([0,1]),vi=np.array([0,1]),
        kp=np.array([1.,20.]),kd=np.ones(2),effort=np.full(2,10.),
        gain=np.array([1.,20.]),rotor_min=np.zeros(0),rotor_max=np.zeros(0))
    control=ContactAllocation(local)
    q=np.array([.1,.02]);v=np.zeros(2);nominal=np.array([.1,.05])
    control.anchor=np.array([.1,-.05])
    truth=mujoco.MjData(m);truth.qpos[:]=q;truth.qvel[:]=v;truth.ctrl[:]=nominal
    mujoco.mj_forward(m,truth);mass=np.empty((2,2));mujoco.mj_fullM(m,truth,mass)
    _,acceleration,_,_=control.linearize(q,v,mass,np.array([[10.],[0.]]),
        np.array([0]),np.array([10.,.5]),nominal)
    np.testing.assert_array_equal(control.d.ctrl,nominal)
    np.testing.assert_allclose(acceleration,truth.qacc,atol=1e-12,rtol=0.)
