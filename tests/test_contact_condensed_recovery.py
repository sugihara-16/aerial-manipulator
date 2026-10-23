"""Same physical contact problem, with independently checked residuals."""
import json
from pathlib import Path
import clarabel
import numpy as np
import pytest
from build.contact_condensed_solver import scaled_recovery


def test_captured_contact_equations_recover_without_relaxing_any_bound():
    path=Path(__file__).parent/'fixtures/contact_equality_scaling.npz'
    data=np.load(path);P,c,A,b=[data[k] for k in ['P','c','A','b']]
    types=json.loads(path.with_suffix('.json').read_text())
    cones=[getattr(clarabel,x['type'])(x['dim']) for x in types]
    result,note=scaled_recovery(P,c,A,b,cones);x=np.asarray(result.x)
    slack=b-A@x;offset=0
    for cone in cones:
        block=slack[offset:offset+cone.dim];offset+=cone.dim
        if isinstance(cone,clarabel.ZeroConeT):assert np.max(abs(block))<1e-7
        elif isinstance(cone,clarabel.NonnegativeConeT):assert min(block)>-1e-7
        elif isinstance(cone,clarabel.SecondOrderConeT):assert block[0]-np.linalg.norm(block[1:])>-1e-7
        else:raise AssertionError(type(cone))
    assert note['eliminated_equalities']>0
    assert abs(.5*x@P@x+c@x-.464758323)<1e-6


def test_inconsistent_equalities_are_not_accepted_after_elimination():
    with pytest.raises(ValueError,match='contact solve'):
        scaled_recovery(np.eye(1),np.zeros(1),np.ones((2,1)),np.array([1.,2.]),[clarabel.ZeroConeT(2)])


def test_well_scaled_solution_and_objective_are_unchanged():
    result,note=scaled_recovery(np.eye(2),np.array([-1.,-2.]),
        np.array([[1.,1.],[-1.,0.],[0.,-1.]]),np.array([1.,0.,0.]),
        [clarabel.ZeroConeT(1),clarabel.NonnegativeConeT(2)])
    np.testing.assert_allclose(result.x,[0.,1.],atol=1e-5)
    assert note['original_residual']<=1e-7
