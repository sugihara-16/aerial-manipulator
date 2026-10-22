"""Check the contact-law projection against an independent convex solve."""
import numpy as np
import scipy.sparse as sp
import clarabel
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
