"""Check the contact-law projection against an independent convex solve."""
import numpy as np
import scipy.sparse as sp
import clarabel
import pytest
from build.contact_friction_modes import project_force, G


def test_weighted_projection_matches_convex_force_problem():
    rng=np.random.default_rng(73)
    examples=[np.array([1.,0.,0.]),np.array([-1.,0.,0.]),np.array([.01,2.,-8.])]
    examples.extend(rng.normal(size=(24,3))*[.3,10.,10.])
    settings=clarabel.DefaultSettings();settings.verbose=False;settings.max_threads=1
    settings.tol_gap_abs=settings.tol_gap_rel=settings.tol_feas=1e-11
    for ratio in [1.,10000.]:
        metric=np.array([138.,138./ratio,138./ratio])
        matrix=np.vstack([G,[-1.,0.,0.]])
        for affine in examples:
            mode,actual=project_force(affine,metric[0],ratio)
            answer=clarabel.DefaultSolver(sp.diags(metric,format='csc'),-metric*affine,
                sp.csc_matrix(matrix),np.zeros(9),[clarabel.NonnegativeConeT(9)],settings).solve()
            assert str(answer.status)in ['Solved','AlmostSolved']
            assert max(abs(actual-np.array(answer.x)))<2e-5
            assert max(matrix@actual)<1e-7


@pytest.mark.parametrize('fixture,separate_all',[
    ('release_zero_force_branch.npz',False),('release_full_separation_branch.npz',True)])
def test_release_can_leave_contact_without_attraction(fixture,separate_all):
    """Real releases previously remained in a feasible closed contact mode."""
    from pathlib import Path
    from build.contact_friction_modes import solve
    from build.held_input_solver import violation
    f=np.load(Path(__file__).parent/'fixtures'/fixture)
    nv,nu,nt=(int(f[k])for k in ['nv','nu','nt'])
    dims=f['cone_dims']
    cones=[clarabel.ZeroConeT(int(dims[0])),clarabel.NonnegativeConeT(int(dims[1])),
           *[clarabel.SecondOrderConeT(int(n))for n in dims[2:]]]
    x,correction,note=solve(f['P'],f['c'],f['A'],f['b'],cones,nv,nu,nt,
        f['pressure_matrix'],f['Dn'],f['p0'],f['rn'],float(f['ratio']),
        f['observed_velocity'],boundary_hint=f['hint'],separate_all=separate_all)
    cost=lambda z:.5*z@f['P']@z+f['c']@z
    assert cost(x)<cost(f['hint'])-.1
    apex=[i for i,mode in enumerate(note['friction_modes'])if mode=='apex']
    assert apex
    if separate_all:
        assert len(apex)==nt//2
        # Waiting for every force to reach numerical zero misses this branch.
        assert min(f['p0']-f['Dn']@f['hint'][:nv])>1e-5
    normal=f['p0']-f['Dn']@x[:nv]
    # The affine spring would pull; unilateral projection instead gives zero.
    assert min(normal[apex])<-.01
    np.testing.assert_allclose((normal+correction)[apex],0.,atol=1e-7)
    assert note['projection_error']<=1e-5
    # Tangential constitutive equations are replaced by the checked branch;
    # original dynamics, input limits and all force/moment cones remain.
    kept=np.r_[np.arange(nv),np.arange(nv+nt,len(f['b']))]
    revised=[clarabel.ZeroConeT(int(dims[0])-nt),*cones[1:]]
    assert violation(f['A'][kept],(f['b']-f['pressure_matrix']@correction)[kept],revised,x)<=1e-5
