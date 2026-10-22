import numpy as np
import pinocchio as pin
from amsrr_det.contact_geometry import close_material
from amsrr_det.objects import ObjectShape


def test_curved_contact_is_found_in_face_interior_not_at_vertices():
    shape=ObjectShape('ellipsoid',[.3,.4,.2],.6)
    triangles=np.array([[[.151,-.04,-.04],[.151,.04,-.04],[.151,.04,.04]],
                        [[.151,-.04,-.04],[.151,.04,.04],[.151,-.04,.04]]])
    material,surface,gap,normals=close_material(shape,pin.SE3.Identity(),triangles)
    np.testing.assert_allclose(gap,.001,atol=1e-8)
    assert len(material)==1
    np.testing.assert_allclose(material[0],[.151,0,0],atol=1e-8)
    np.testing.assert_allclose(surface[0],[.15,0,0],atol=1e-8)


def test_flat_contact_retains_separated_real_points():
    shape=ObjectShape('box',[.3,.4,.2],.6)
    triangles=np.array([[[.151,-.04,-.04],[.151,.04,-.04],[.151,.04,.04]],
                        [[.151,-.04,-.04],[.151,.04,.04],[.151,-.04,.04]]])
    material,surface,gap,normals=close_material(shape,pin.SE3.Identity(),triangles)
    # Coplanar nearest interiors can coincide; include distinct mesh material
    # samples as candidates so broad flat patches do not collapse to one point.
    assert np.max(abs(gap-.001))<1e-8
    assert len(material)>=4
    assert np.linalg.matrix_rank(material-material.mean(axis=0),tol=1e-8)==2


def test_cylinder_face_interior_contact_is_not_biased_toward_axis_center():
    # All vertices are outside, while the long slanted edge crosses the cylinder
    # at y=0. COM-based alternating projection used to miss this penetration.
    shape=ObjectShape('cylinder',[.26,.26,.36],.6)
    triangles=np.array([[[.12985,-.02,-.15],[.12985,.02,.05],[.12985,.04,.05]]])
    _,_,gaps,_=close_material(shape,pin.SE3.Identity(),triangles)
    assert abs(gaps.min()+.00015)<1e-7
