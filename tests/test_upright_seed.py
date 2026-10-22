"""Resolve a free rotation without increasing contact alignment error."""
import numpy as np
import pinocchio as pin
from amsrr_det.model import build_model
from amsrr_det.morphology import align_seed
from amsrr_det.grasp_geometry import grasp_frame


def test_collinear_contact_alignment_preserves_fit_and_maximizes_root_up():
    model,meta=build_model(2,[dict(a=1,b=0,port_a='pitch2',port_b='yaw1')])
    q=pin.neutral(model);surfaces=[(0,'yaw2'),(1,'yaw2')]
    axis=np.array([.8,.6,0.]);center=np.array([.1,.2,.5])
    points=np.array([center+.2*axis,center-.2*axis]);normals=np.array([-axis,axis])
    ordinary,_=align_seed(model,meta,q,surfaces,points,normals)
    upright,_=align_seed(model,meta,q,surfaces,points,normals,upright=True)
    def error(configuration):
        data=model.createData();pin.framesForwardKinematics(model,data,configuration)
        frames=[data.oMf[meta['frames'][grasp_frame(*s)]] for s in surfaces]
        return sum(np.linalg.norm(T.translation-p)**2+
                   np.linalg.norm(T.translation+.15*T.rotation[:,0]-p-.15*n)**2
                   for T,p,n in zip(frames,points,normals))
    assert abs(error(ordinary)-error(upright))<1e-12
    old=pin.Quaternion(ordinary[3:7]).matrix()[2,2]
    best=pin.Quaternion(upright[3:7]).matrix()[2,2]
    assert best>=old-1e-12
    for angle in np.linspace(-np.pi,np.pi,41):
        candidate=pin.exp3(axis*angle)@pin.Quaternion(upright[3:7]).matrix()
        assert candidate[2,2]<=best+1e-12
