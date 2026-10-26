"""Compiled growth scoring must preserve the reference geometry calculation."""
import numpy as np
import pinocchio as pin
import pytest
from amsrr_det import task_structure as growth
from amsrr_det.native_grasp_geometry import library
from amsrr_det.objects import ObjectShape


def test_compiled_growth_overlap_matches_numpy_for_rotated_objects_and_obstacles(monkeypatch):
    if library() is None:pytest.skip('local geometry kernel has not been built')
    rng=np.random.default_rng(913);ports,angles,seams,*_=growth.part_frames();k=len(angles)
    shape=ObjectShape('box',[.67,.91,.35],12.)
    R=pin.exp3(np.array([.2,-.35,.7]));source=np.r_[[.13,-.2,.6],pin.Quaternion(R).coeffs()]
    original=growth.growth_clearance
    for incoming,pad in [(0,1),(1,3),(2,0),(3,2)]:
        base=np.broadcast_to(np.eye(4),(k*k,4,4)).copy()
        for i in range(k*k):base[i,:3,:3]=pin.exp3(rng.normal(size=3));base[i,:3,3]=rng.uniform(-1,1,3)
        for obstacles in [(),[(np.array([.3,-.2,.1]),np.array([.5,.6,.1])),(np.array([-.4,.5,.2]),np.array([.4,.3,.2]))]]:
            monkeypatch.setattr(growth,'growth_clearance',original)
            actual=growth._clearance_cost(base,incoming,pad,shape,source,obstacles)
            monkeypatch.setattr(growth,'growth_clearance',lambda *args:None)
            expected=growth._clearance_cost(base,incoming,pad,shape,source,obstacles)
            np.testing.assert_allclose(actual,expected,rtol=2e-13,atol=2e-13)
