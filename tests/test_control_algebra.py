"""Equivalent numerical paths must retain the original mechanics and optima."""
import numpy as np
import mujoco
import quadprog
from amsrr_det import native_qp
from amsrr_det.native_observed_geometry import library
from build.contact_friction_modes import project_force,project_forces


def test_batched_projection_matches_individual_force_law():
    rng=np.random.default_rng(1027)
    for ratio in [1.,100.,10000.]:
        affine=rng.normal(0,10,(100,3));rn=np.exp(rng.normal(size=100))
        modes,actual=project_forces(affine,rn,ratio)
        for i in range(len(affine)):
            mode,expected=project_force(affine[i],rn[i],ratio)
            assert modes[i]==mode
            np.testing.assert_allclose(actual[i],expected,rtol=1e-13,atol=1e-13)


def test_native_qp_preserves_original_optimum_and_dual_residual():
    rng=np.random.default_rng(2827)
    for n in [2,8,30]:
        for _ in range(8):
            C=rng.normal(size=(n,4*n));b=-rng.uniform(.1,2.,4*n);a=rng.normal(size=n)*3
            x,_,_,_,dual,_=native_qp.solve_qp(np.eye(n),a,C,b,factorized=True)
            expected=quadprog.solve_qp(np.eye(n),a,C,b,factorized=True)[0]
            np.testing.assert_allclose(x,expected,rtol=1e-9,atol=1e-9)
            assert min(C.T@x-b)>-1e-9 and min(dual)>=0.
            np.testing.assert_allclose(x-a-C@dual,0.,atol=1e-9)


def test_contact_mechanics_matches_full_update_with_contact_and_motion():
    m=mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <geom name="floor" type="plane" size="2 2 .1"/>
      <body pos="0 0 .12"><freejoint/><geom type="box" size=".1 .1 .1" mass="2"/>
        <body pos=".2 0 0"><joint axis="0 1 0" armature=".003"/>
          <geom type="box" size=".2 .02 .03" mass=".5"/><site name="rotor"/>
        </body></body></worldbody></mujoco>''')
    actual=mujoco.MjData(m);reference=mujoco.MjData(m)
    vi=np.array([6],np.int32);sites=np.array([0],np.int32);M=np.empty((m.nv,m.nv));B=np.empty((m.nv,2))
    rng=np.random.default_rng(25)
    for _ in range(20):
        q=m.qpos0.copy();mujoco.mj_integratePos(m,q,rng.normal(0,.1,m.nv),1.)
        v=rng.normal(0,.3,m.nv)
        actual.qpos[:]=reference.qpos[:]=q;actual.qvel[:]=reference.qvel[:]=v
        mujoco.mj_fwdPosition(m,reference);mujoco.mj_fwdVelocity(m,reference)
        library().contact_mechanics(m._address,actual._address,1,vi.ctypes.data,1,sites.ctypes.data,M.ctypes.data,B.ctypes.data)
        expected=np.empty_like(M);mujoco.mj_fullM(m,reference,expected)
        np.testing.assert_allclose(M,expected,atol=1e-13,rtol=1e-13)
        for field in ['qfrc_bias','xpos','xmat','site_xmat','cdof']:
            np.testing.assert_allclose(getattr(actual,field),getattr(reference,field),atol=1e-13,rtol=1e-13)
        assert actual.ncon==reference.ncon
        for a,b in zip(actual.contact,reference.contact):
            np.testing.assert_array_equal(a.geom,b.geom)
            np.testing.assert_allclose(a.frame,b.frame,atol=1e-13)
            assert a.dist==b.dist
        J=np.empty((3,m.nv));mujoco.mj_jacSite(m,reference,J,None,0)
        np.testing.assert_allclose(B[:,1],J.T@reference.site_xmat[0].reshape(3,3)[:,2],atol=1e-13)


def test_native_branches_preserve_contact_equations_and_original_bounds():
    from build.contact_friction_modes import MODES,H,VERTICES,C
    rng=np.random.default_rng(321);nv,nu,count=3,2,4;n=nv+nu+2*count;ne=nv+2*count;ni=5
    A=rng.normal(size=(ne+ni,n));b=rng.normal(size=len(A));pressure=rng.normal(size=(len(A),count))
    Dn=rng.normal(size=(count,nv));p0=rng.normal(size=count);rn=rng.uniform(.2,2,count)
    alpha=rng.uniform(.01,1,2*count);ratio=100.;encoded=np.array([-1,0,4,15])
    selected=['apex' if i<0 else MODES[i] for i in encoded]
    matrix,bound,eq,ine= native_qp.branch(A,b,nv,nu,ne,ni,pressure,Dn,p0,rn,alpha,ratio,H,VERTICES,C,encoded)
    y=rng.normal(size=matrix.shape[1]);x=y[:n];correction=y[n:n+count];multiplier=n+count
    ordinary=A@x+pressure@correction-b;equal=list(ordinary[:nv]);extra=[];upper=[]
    for i,mode in enumerate(selected):
        r=nv+2*i;col=nv+nu+2*i;factor=rn[i]/ratio/alpha[2*i]
        if mode=='apex':
            extra.extend([-Dn[i]@x[:nv]+correction[i]+p0[i],x[col],x[col+1]])
            upper.extend(rn[i]*(p0[i]-Dn[i]@x[:nv])+factor*(VERTICES@(b[r:r+2]-A[r:r+2,:nv]@x[:nv])))
        else:
            tangent=factor*ordinary[r:r+2];normal=rn[i]*correction[i]
            for face in mode:
                tangent=tangent+H[face]*y[multiplier];normal-=C*y[multiplier]
                extra.append(C*(Dn[i]@x[:nv]-p0[i]-correction[i])+H[face]@x[col:col+2])
                upper.append(-y[multiplier]);multiplier+=1
            equal.extend(tangent);extra.append(normal)
    expected=np.r_[equal,extra,ordinary[ne:],upper]
    assert eq==len(equal)+len(extra) and ine==ni+len(upper)
    np.testing.assert_allclose(matrix@y-bound,expected,rtol=1e-13,atol=1e-13)
