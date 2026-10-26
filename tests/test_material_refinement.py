"""A close pad can still face the wrong normal on an object's curved side."""
import json
from pathlib import Path
from amsrr_det.mujoco_model import make_plant,CONTACT_CACHE
from amsrr_det.grasp_task import refine_material_posture,support_obstacles
from amsrr_det.grasp_geometry import separating_translation


def test_near_zero_gap_does_not_end_refinement_with_wrong_material_normal():
    case=json.loads((Path(__file__).parent/'fixtures/curved_pitch_misalignment.json').read_text())
    payload=case['payload'];shape=ObjectShape(payload['shape'],payload['size'],payload['mass'])
    source=np.array(case['source']);destination=np.array(case['destination']);q=np.array(case['q'])
    plant=make_plant(3,case['edges'],payload=shape.payload(),mesh_cache=CONTACT_CACHE,floor=True)
    _,_,fit=refine_material_posture(plant,shape,source,q,list(map(tuple,case['surfaces'])),
        np.array(case['points']),np.array(case['normals']),
        obstacles=support_obstacles(shape,source,destination,case['support_top']))
    assert max(abs(np.array(fit['material_gap_m'])))<.00005
    assert fit['material_normal_error']<.1


def test_separating_translation_oblique_parallel_and_opposed_planes():
    normals=np.array([[1.,1.,0.],[1.,-1.,0.]])/np.sqrt(2)
    displacement=separating_translation(normals,.045)
    np.testing.assert_allclose(displacement,[-np.sqrt(2)*.045,0.,0.],atol=1e-12)
    repeated=separating_translation(np.repeat(normals,3,axis=0),.045)
    np.testing.assert_allclose(repeated,displacement,atol=1e-12)
    assert separating_translation(np.array([[1.,0.,0.],[-1.,0.,0.]]),.045) is None


def test_adjacent_pads_can_open_by_translation_without_offset_point_matching():
    case=json.loads((Path(__file__).parent/'fixtures/adjacent_pad_opening.json').read_text())
    payload=case['payload'];shape=ObjectShape(payload['shape'],payload['size'],payload['mass'])
    source=np.array(case['source']);destination=np.array(case['destination'])
    plant=make_plant(1,[],payload=shape.payload(),mesh_cache=CONTACT_CACHE,floor=True)
    opened,closed,fit=refine_material_posture(plant,shape,source,np.array(case['q']),
        list(map(tuple,case['surfaces'])),np.array(case['points']),np.array(case['normals']),
        obstacles=support_obstacles(shape,source,destination,case['support_top']))
    assert fit['opening_method']=='separating_translation'
    assert min(fit['open_material_gap_m'])>=.025
    np.testing.assert_array_equal(opened[3:],closed[3:])


import numpy as np
import pinocchio as pin
from amsrr_det.objects import ObjectShape
from amsrr_det.contact_geometry import material_triangles,close_material
from amsrr_det.native_grasp_geometry import library


def test_native_material_projection_matches_python_for_curved_and_polyhedral_objects():
    assert library() is not None
    rng=np.random.default_rng(381)
    for family,size in [('box',[.9,.7,.6]),('ellipsoid',[1.2,.65,.8]),('cylinder',[.8,.8,1.]),
                        ('triangular_prism',[.9,.7,.6]),('tapered_box',[.9,.7,.6])]:
        shape=ObjectShape(family,size,1.)
        for port in ['yaw1','pitch2']:
            triangles=material_triangles(port)
            for _ in range(4):
                direction=rng.normal(size=3);direction/=np.linalg.norm(direction)
                point,_=shape.ray_surface(direction)
                pose=pin.SE3(pin.exp3(rng.normal(size=3)),point)
                reference=close_material(shape,pose,triangles,native=False)
                actual=close_material(shape,pose,triangles,native=True)
                for x,y in zip(actual,reference):np.testing.assert_allclose(x,y,rtol=1e-7,atol=1e-9)
