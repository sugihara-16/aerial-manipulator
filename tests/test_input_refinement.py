import numpy as np
import pytest
from build.minimum_input_lp import minimum_input


@pytest.mark.parametrize('method',['conic','slsqp','piecewise','hybrid'])
def test_convex_input_refinement_reaches_analytic_optimum_with_required_reserve(method):
    problem=dict(A_ub=np.array([[0.,0.,1.]]),b_ub=np.array([.5]),
                 A_eq=np.array([[1.,1.,0.]]),b_eq=np.array([1.]),
                 bounds=[(0.,1.),(0.,1.),(0.,.5)])
    result,report=minimum_input(problem,np.array([.9,.1,.5]),[0,1],np.array([20.,20.]),.2,method=method)
    assert report['accepted']
    optimum=2*10**1.5
    if method=='piecewise':
        assert report['model_optimum_lower_bound']<=optimum+1e-8
        assert report['burden']-optimum<=report['interpolation_error_bound']+1e-8
    else:assert abs(report['burden']-optimum)<1e-6
    assert .2-1e-10<=result[-1]<=.5+1e-10
    if method!='piecewise':assert np.allclose(result[:2],[.5,.5],atol=1e-5)


def test_refinement_preserves_a_fixed_variable_and_an_active_force_limit():
    problem=dict(A_ub=np.array([[0.,0.,0.,1.]]),b_ub=np.array([.5]),
                 A_eq=np.array([[0.,1.,1.,0.]]),b_eq=np.array([1.]),
                 bounds=[(0.,0.),(.8,1.),(0.,1.),(0.,.5)])
    result,report=minimum_input(problem,np.array([0.,.9,.1,.5]),[1,2],np.array([20.,20.]),.25)
    assert report['accepted'] and result[0]==0
    assert np.allclose(result[1:3],[.8,.2],atol=1e-7)
    assert abs(report['burden']-(16**1.5+4**1.5))<1e-5
