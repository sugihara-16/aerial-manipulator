"""An interior collision-piece sample is a sufficient, not necessary, rejection."""
import numpy as np
import pinocchio as pin
from amsrr_det.model import build_model
from amsrr_det.objects import ObjectShape
from amsrr_det.object_overlap import object_overlap,part_samples


def test_interior_sample_rejects_but_separated_shape_does_not():
    shape=ObjectShape('box',[2.,2.,2.],.7);model,meta=build_model(1,payload=shape.payload())
    q=pin.neutral(model)
    assert object_overlap(model,meta,q,shape,np.array([0,0,0,0,0,0,1.]),[]) is not None
    assert object_overlap(model,meta,q,shape,np.array([10,0,0,0,0,0,1.]),[]) is None


def test_piece_samples_belong_to_compiled_collision_geometry():
    import mujoco
    from amsrr_det.mujoco_model import make_plant,CONTACT_CACHE
    p=make_plant(1,mesh_cache=CONTACT_CACHE);m=p.mj;d=mujoco.MjData(m)
    q=pin.neutral(p.pin);d.qpos[:]=p.configuration(q);mujoco.mj_kinematics(m,d)
    pd=p.pin.createData();pin.framesForwardKinematics(p.pin,pd,q)
    for link,points in part_samples():
        T=pd.oMf[p.meta['frames']['m0/'+link]]
        world=points@T.rotation.T+T.translation
        # All selected extremal vertices are present in a collision mesh; the
        # per-piece mean lies within its convex hull (tested by its halfspaces).
        from scipy.spatial import ConvexHull
        occupied=np.zeros(len(points),dtype=bool)
        for g in range(m.ngeom):
            if not m.geom(g).name.startswith('m0/'+link+'/'):continue
            mesh=m.geom_dataid[g];start=m.mesh_vertadr[mesh];n=m.mesh_vertnum[mesh]
            vertices=m.mesh_vert[start:start+n]@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g]
            equations=ConvexHull(vertices).equations
            occupied|=np.all(world@equations[:,:3].T+equations[:,3]<2e-7,axis=1)
        assert occupied.all(),link
