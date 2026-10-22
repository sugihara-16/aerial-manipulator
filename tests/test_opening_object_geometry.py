"""Collision-repair geometry must retain each object's dimensions and center."""
import numpy as np
import coal
from amsrr_det.objects import ObjectShape
from amsrr_det.retreat import object_avoidance_hull


def test_repair_objects_match_independent_surface_distances():
    for family in ['box','elongated_box','cylinder','ellipsoid','triangular_prism','tapered_box']:
        shape=ObjectShape(family,[.3,.3,.2],.7);hull=object_avoidance_hull(shape)
        for point in [[0,0,.4],[.5,0,0],[0,.5,0]]:
            request=coal.DistanceRequest();result=coal.DistanceResult()
            transform=coal.Transform3s();transform.setTranslation(np.array(point,dtype=float))
            actual=coal.distance(hull,coal.Transform3s(),coal.Sphere(0.),transform,request,result)
            expected=shape.project_surface([point])[1][0]
            assert abs(actual-expected)<1e-6,(family,point,actual,expected)
