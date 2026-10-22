"""Exact zero rows cannot change a matrix factorization or fitted optimum."""
import numpy as np
import pytest
from scipy.optimize import least_squares as reference
from amsrr_det.geometry_least_squares import compact_svd,least_squares


@pytest.mark.parametrize('shape,active', [((120,8),20),((40,8),8),((40,8),4),((8,12),8)])
def test_compact_decomposition_reconstructs_original_matrix(shape,active):
    rng=np.random.default_rng(20261022);matrix=np.zeros(shape)
    matrix[rng.choice(shape[0],active,replace=False)]=rng.normal(size=(active,shape[1]))
    u,s,v=compact_svd(matrix)
    np.testing.assert_allclose((u*s)@v,matrix,atol=2e-13)
    np.testing.assert_allclose(s,np.linalg.svd(matrix,compute_uv=False),atol=2e-13)


def test_bounded_fit_matches_public_solver_when_inactive_rows_change():
    def residual(x):return np.r_[x-[2.,-.6],np.maximum(0.,np.linspace(-1,1,100)[:,None]+x).ravel(),1e-4*x]
    def derivative(x):return np.vstack([np.eye(2),np.array([np.diag(row>0.)for row in np.linspace(-1,1,100)[:,None]+x]).reshape(-1,2),1e-4*np.eye(2)])
    settings=dict(jac=derivative,bounds=([-1.,-1.],[1.,1.]),ftol=1e-11,xtol=1e-11,gtol=1e-11)
    expected=reference(residual,[.8,-.8],**settings);actual=least_squares(residual,[.8,-.8],**settings)
    np.testing.assert_allclose(actual.x,expected.x,atol=1e-9)
    assert abs(actual.cost-expected.cost)<1e-11


def test_failed_compact_decomposition_uses_the_original_matrix(monkeypatch):
    import amsrr_det.geometry_least_squares as module
    old=module.svd;matrix=np.zeros((120,8));matrix[:8]=np.eye(8)
    def reject_smaller(a,**kwargs):
        if len(a)<120:raise np.linalg.LinAlgError('fixture: compact layout failed')
        return old(a,**kwargs)
    monkeypatch.setattr(module,'svd',reject_smaller)
    u,s,v=module.compact_svd(matrix)
    np.testing.assert_allclose((u*s)@v,matrix,atol=1e-14)
