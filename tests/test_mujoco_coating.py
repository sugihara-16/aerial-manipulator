"""Load/penetration is a physical check, independent of task success."""
import numpy as np
import mujoco
import pytest
from amsrr_det.mujoco_model import coat_grippers


@pytest.mark.parametrize('mass',[.4,1.,2.])
@pytest.mark.parametrize('integrator',['implicitfast','discrete'])
def test_coating_static_force_and_release(mass,integrator):
    xml=f'''<mujoco><option timestep="0.0002" gravity="0 0 -9.80665" integrator="{integrator}" cone="elliptic"/>
    <worldbody><geom name="pad/0" type="plane" size="1 1 .1" condim="1"/>
    <body pos="0 0 .05"><freejoint/><geom name="object" type="sphere" size=".05" mass="{mass}" condim="1"/></body></worldbody></mujoco>'''
    m=mujoco.MjModel.from_xml_string(coat_grippers(xml,['pad']));d=mujoco.MjData(m)
    for _ in range(15000):mujoco.mj_step(m,d)
    compression=.05-d.qpos[2]
    assert compression==pytest.approx(mass*9.80665/7500,rel=1e-4)
    assert abs(d.qvel[2])<1e-8
    assert d.ncon==1
    force=np.zeros(6);mujoco.mj_contactForce(m,d,0,force)
    assert force[0]==pytest.approx(mass*9.80665,rel=1e-5)
    # A separating object cannot be pulled back by the coating.
    d.qvel[2]=1.
    for _ in range(200):mujoco.mj_step(m,d)
    assert d.qpos[2]>.05 and d.ncon==0


def test_coating_reuses_inertias_after_adding_world_support_geometry():
    import xml.etree.ElementTree as ET
    xml='''<mujoco><worldbody>
    <body pos="0 0 1"><freejoint/><geom name="pad/0" type="box" size=".05 .05 .05" mass="2"/></body>
    <body pos="0 0 .5"><freejoint/><geom name="object" type="sphere" size=".05" mass="1"/></body>
    </worldbody></mujoco>'''
    reference=mujoco.MjModel.from_xml_string(xml)
    doc=ET.fromstring(xml)
    doc.find('worldbody').insert(0,ET.Element('geom',name='support',type='box',size='.2 .2 .1'))
    scene=ET.tostring(doc,encoding='unicode')
    assert coat_grippers(scene,['pad'],reference_model=reference)==coat_grippers(scene,['pad'])
