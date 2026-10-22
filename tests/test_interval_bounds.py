"""Independent geometry checks for the conservative interval certificate."""
import numpy as np
import mujoco
from amsrr_det.interval_bounds import IntervalTree

def test_tree_does_not_omit_close_primitive_pairs():
    m=mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <geom name="floor" type="plane" size="0 0 .1"/>
      <body name="a"><freejoint/><geom type="box" size=".1 .04 .03"/>
        <geom type="sphere" size=".04" pos=".12 0 .08"/></body>
      <body name="b"><freejoint/><geom type="cylinder" size=".06 .08"/>
        <geom type="ellipsoid" size=".07 .03 .05" pos="-.1 0 .05"/></body>
    </worldbody></mujoco>''')
    d=mujoco.MjData(m);tree=IntervalTree();rng=np.random.default_rng(81206)
    entities=np.array([[1,-1,2,-1],[0,0,1,-1],[0,0,2,-1]],np.int32)
    thresholds=np.array([.07,.03,.03]);result=np.empty((100,3),np.int32)
    for _ in range(64):
        for k in [0,7]:
            d.qpos[k:k+3]=rng.uniform(-.2,.2,3)
            quat=rng.normal(size=4);d.qpos[k+3:k+7]=quat/np.linalg.norm(quat)
        mujoco.mj_kinematics(m,d)
        count=tree.call(m._address,d._address,3,entities.ctypes.data,thresholds.ctypes.data,len(result),result.ctypes.data)
        assert count>=0
        candidates={tuple(row)for row in result[:count]}
        for row,(ba,ga,bb,gb)in enumerate(entities):
            aa=np.flatnonzero(m.geom_bodyid==ba)if ga<0 else [ga]
            bs=np.flatnonzero(m.geom_bodyid==bb)if gb<0 else [gb]
            for a in aa:
                for b in bs:
                    cutoff=1.;distance=mujoco.mj_geomDistance(m,d,int(a),int(b),cutoff,None)
                    if distance<thresholds[row]-1e-10:assert (a,b,row)in candidates

def test_support_certificate_uses_geometry_not_claimed_witness_distance():
    m=mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
       <geom type="box" size=".1 .2 .3"/>
       <geom type="box" size=".1 .2 .3" pos=".25 0 0"/>
       <geom type="plane" size="0 0 .1" pos="0 0 -1"/>
    </worldbody></mujoco>''')
    d=mujoco.MjData(m);mujoco.mj_kinematics(m,d);tree=IntervalTree()
    # Deliberately distant fake witnesses only suggest x, never a 100 m gap.
    witness=np.array([0.,0.,0.,100.,0.,0.])
    lower=tree.separation(m._address,d._address,0,1,witness.ctypes.data)
    assert .05-1e-8<lower<.05
    floor=tree.separation(m._address,d._address,2,0,witness.ctypes.data)
    assert .7-1e-8<floor<.7
