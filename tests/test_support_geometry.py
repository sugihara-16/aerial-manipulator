import xml.etree.ElementTree as ET
import numpy as np
import pytest
from amsrr_det.support_geometry import merge_rectangular_supports
from amsrr_det.support_geometry import support_layers


def boxes(second):
    return f'<mujoco><worldbody><geom name="support" type="box" pos="0 0 0" size="1 1 1"/><geom name="destination_support" type="box" pos="{second}" size="1 1 1"/></worldbody></mujoco>'


def test_union_preserves_occupied_set_and_removes_internal_seam():
    xml,info=merge_rectangular_supports(boxes('1 0 0'))
    geoms=ET.fromstring(xml).findall('worldbody/geom');assert len(geoms)==1
    center=np.fromstring(geoms[0].get('pos'),sep=' ');half=np.fromstring(geoms[0].get('size'),sep=' ')
    points=np.random.default_rng(172).uniform(-3,3,(10000,3))
    before=np.all(abs(points)<=1,axis=1)|np.all(abs(points-[1,0,0])<=1,axis=1)
    np.testing.assert_array_equal(np.all(abs(points-center)<=half,axis=1),before)
    assert info['union_volume_m3']==12.


@pytest.mark.parametrize('position',['3 0 0','1 1 0'])
def test_union_rejects_added_material(position):
    with pytest.raises(ValueError,match='refusing to fill empty space'):merge_rectangular_supports(boxes(position))


def test_layer_partition_preserves_box_and_buries_new_interface():
    half=np.array([.23,.31,.125]);layers=support_layers(half)
    lo=np.array([c-h for c,h in layers]);hi=np.array([c+h for c,h in layers])
    np.testing.assert_allclose(lo.min(axis=0),-half,atol=1e-15)
    np.testing.assert_allclose(hi.max(axis=0),half,atol=1e-15)
    np.testing.assert_allclose(lo[0,2],hi[1,2],atol=1e-15)
    assert abs(sum(np.prod(2*h) for c,h in layers)-np.prod(2*half))<1e-15
    assert half[2]-hi[1,2]>=.004-1e-15
    points=np.random.default_rng(173).uniform(-half*1.1,half*1.1,(20000,3))
    partition=np.any([np.all(abs(points-c)<=h,axis=1) for c,h in layers],axis=0)
    np.testing.assert_array_equal(partition,np.all(abs(points)<=half,axis=1))


@pytest.mark.parametrize('centers',[[[0,0,0],[1,.2,0]],[[0,0,0],[3,0,0]],[[0,0,0],[.2,.3,.4]]])
def test_nonrectangular_union_preserves_boundary_and_volume(centers):
    from amsrr_det.support_geometry import box_union_surface
    centers=np.asarray(centers);half=np.ones_like(centers)
    mesh,info=box_union_surface(centers,half)
    overlap=np.maximum(0,np.minimum(centers[0]+1,centers[1]+1)-np.maximum(centers[0]-1,centers[1]-1))
    assert abs(mesh.volume-(16-np.prod(overlap)))<1e-12
    assert mesh.is_watertight and mesh.is_winding_consistent
    middle=mesh.triangles_center;normal=mesh.face_normals
    inside=lambda pts:np.any(np.all(abs(pts[:,None]-centers)<half,axis=2),axis=1)
    assert inside(middle-normal*1e-6).all()
    assert not inside(middle+normal*1e-6).any()
    # Independent solid-angle winding checks occupied and empty regions.
    rng=np.random.default_rng(174)
    points=rng.uniform(centers.min(axis=0)-1.2,centers.max(axis=0)+1.2,(2000,3))
    r=mesh.triangles[None,:,:,:]-points[:,None,None,:]
    a,b,c=r[:,:,0],r[:,:,1],r[:,:,2]
    na,nb,nc=(np.linalg.norm(x,axis=-1) for x in [a,b,c])
    dot=lambda x,y:np.einsum('...i,...i->...',x,y)
    angle=2*np.arctan2(dot(a,np.cross(b,c)),na*nb*nc+dot(a,b)*nc+dot(b,c)*na+dot(c,a)*nb)
    np.testing.assert_array_equal(abs(angle.sum(axis=1))>2*np.pi,inside(points))
    assert info['scope']=='exact_axis_aligned_box_union_boundary'
