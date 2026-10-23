"""The holding-entry guide acts on a physical pad's aggregate compression."""
import mujoco
import numpy as np
import pytest
from build.separation_rate_constraints import preload_domain_rates,preload_in_physical_domain


def scene(compression):
    # Two physical contact elements on one pad; their loads must add.
    xml=f'''<mujoco><worldbody><geom name="object" type="plane" size="1 1 .1"/>
      <body name="pad" pos="0 0 {0.01-compression}">
        <joint type="slide" axis="0 0 1"/>
        <geom name="pad/a" type="sphere" size=".01" pos="-.02 0 0"/>
        <geom name="pad/b" type="sphere" size=".01" pos=".02 0 0"/>
      </body></worldbody></mujoco>'''
    m=mujoco.MjModel.from_xml_string(xml);d=mujoco.MjData(m);mujoco.mj_fwdPosition(m,d)
    assert len(d.contact)==2
    return m,d


def test_safe_aggregate_preload_does_not_require_motion():
    m,d=scene(.0002)
    assert preload_in_physical_domain(m,d,['pad'])
    rows,bounds=preload_domain_rates(m,d,['pad'],.0025)
    assert np.all(rows@np.zeros(m.nv)>=bounds)


def test_two_elements_require_relief_even_if_each_is_below_force_bound():
    m,d=scene(.0005)
    assert 7500*.0005*np.sqrt(1+4.5**2)<30.
    assert not preload_in_physical_domain(m,d,['pad'])
    rows,bounds=preload_domain_rates(m,d,['pad'],.0025)
    assert bounds[0]>0.
    velocity=bounds[0]/rows[0,0]
    before=sum(-7500*c.dist for c in d.contact)
    mujoco.mj_integratePos(m,d.qpos,np.array([velocity]),.0025)
    mujoco.mj_fwdPosition(m,d)
    after=sum(-7500*c.dist for c in d.contact)
    assert after<before
    target=.95*30./np.sqrt(1+4.5**2)
    assert after==pytest.approx(before-(before-target)*.0025/.5,abs=1e-10)
