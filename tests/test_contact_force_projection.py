import numpy as np
from amsrr_det.equilibrium import contact_normal_obstruction


def test_single_side_and_top_pad_cannot_carry_but_bottom_and_opposed_can():
    def pad(normal,name='a'):return dict(normal=normal,mu=2.25,frame=name)
    load=[0,0,-4.]
    assert contact_normal_obstruction([pad([1,0,0])],load,.5) is not None
    assert contact_normal_obstruction([pad([0,0,1])],load,.5) is not None
    assert contact_normal_obstruction([pad([0,0,-1])],load,.5) is None
    assert contact_normal_obstruction([pad([1,0,0]),pad([-1,0,0],'b')],load,.5) is None


def test_arbitrary_feasible_inner_cone_forces_are_never_excluded():
    rng=np.random.default_rng(41022)
    for _ in range(100):
        contacts=[];total=np.zeros(3)
        count=int(rng.integers(1,7))
        for index in range(count):
            normal=rng.normal(size=3);normal/=np.linalg.norm(normal)
            t1=np.cross(normal,np.eye(3)[np.argmin(abs(normal))]);t1/=np.linalg.norm(t1)
            t2=np.cross(normal,t1);mu=float(rng.uniform(.1,4.5))
            force=float(rng.uniform(.5,3.));tangents=rng.uniform(-1,1,2)
            tangents*=mu*force/max(1.,np.abs(tangents).sum())
            total+=force*normal+tangents[0]*t1+tangents[1]*t2
            contacts.append(dict(normal=normal,mu=mu,frame=str(index)))
        assert contact_normal_obstruction(contacts,total,.5) is None
