"""A cache contains the same authored pad triangles, not fitted grasps."""
import numpy as np
import pytest
from amsrr_det.model import PORTS,ROOT
from amsrr_det.mujoco_model import CONTACT_CACHE
from amsrr_det.contact_geometry import material_triangles,material_cache_identity


@pytest.mark.parametrize('port',list(PORTS))
@pytest.mark.parametrize('mesh_cache',[None,str(CONTACT_CACHE.resolve())])
def test_saved_triangles_equal_current_authored_surfaces(port,mesh_cache):
    cache=ROOT/'assets/generated/material_surfaces.npz'
    if not cache.exists():pytest.skip('optional material cache not prepared')
    with np.load(cache) as saved:actual=saved[material_cache_identity(port,mesh_cache)]
    expected=material_triangles(port,mesh_cache,use_cache=False)
    np.testing.assert_array_equal(actual,expected)
