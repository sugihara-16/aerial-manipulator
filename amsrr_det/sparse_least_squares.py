"""Sparse trust-region solves without nested Python matrix-vector wrappers.

SciPy's bounded trust-region algorithm is retained. Its two operator factories
are specialized to explicit sparse Jacobians in a private function namespace;
no process-wide SciPy symbols are changed. Both constructed matrices are
algebraically identical to the original operators, including regularization.
"""
from functools import lru_cache
from types import FunctionType
import numpy as np
from scipy.sparse import issparse,csr_matrix,diags,vstack
from scipy.optimize import least_squares as scipy_least_squares
from scipy.optimize._lsq import trf as trust_region
from scipy.sparse.linalg import lsmr, splu
from scipy.sparse.csgraph import reverse_cuthill_mckee
from scipy.linalg import cholesky_banded, cho_solve_banded, cho_factor, cho_solve


def _copy(function,namespace):
    result=FunctionType(function.__code__,namespace,function.__name__,function.__defaults__,function.__closure__)
    result.__kwdefaults__=function.__kwdefaults__
    return result


def _right_scaled(matrix,scale):
    return matrix.multiply(scale).tocsr() if issparse(matrix) else np.asarray(matrix)*scale


def _regularized(matrix,diagonal):
    return vstack([csr_matrix(matrix),diags(diagonal)],format='csr')


def _normal_solve(matrix,rhs):
    # The small shared-posture block is dense enough for Cholesky to be
    # cheaper than sparse LU. The caller still checks the original residual.
    if matrix.shape[0]<=640:
        try:
            return cho_solve(cho_factor(matrix.toarray(),check_finite=False),rhs,check_finite=False)
        except np.linalg.LinAlgError:
            pass
    # A cycle couples only nearby time blocks. Reordering closes the wrap
    # without a dense fill-in; band Cholesky solves the same normal equation.
    if matrix.shape[0]>640:
        order=reverse_cuthill_mckee(matrix,symmetric_mode=True)
        a=matrix[order,:][:,order].tocoo();keep=a.row>=a.col
        row,col=a.row[keep],a.col[keep];width=int((row-col).max(initial=0))
        if width<matrix.shape[0]//3:
            band=np.zeros((width+1,len(order)),order='F');band[row-col,col]=a.data[keep]
            try:
                factor=cholesky_banded(band,lower=True,check_finite=False)
                x=np.empty_like(rhs);x[order]=cho_solve_banded((factor,True),rhs[order],check_finite=False)
                return x
            except np.linalg.LinAlgError:
                pass
    return splu(matrix).solve(rhs)


def _arrow_solve(matrix,rhs,blocks):
    """Eliminate independent force blocks around a shared posture block.

    The trust-region scaling and diagonal regularization preserve this
    pattern. The caller verifies the original normal-equation residual.
    """
    shared=blocks[0]
    reduced=matrix[:shared,:shared].toarray();target=rhs[:shared].copy()
    solved=[]
    for start,end in zip(blocks[:-1],blocks[1:]):
        coupling=matrix[start:end,:shared].toarray()
        factor=cho_factor(matrix[start:end,start:end].toarray(),check_finite=False)
        inverse=cho_solve(factor,np.column_stack([rhs[start:end],coupling]),check_finite=False)
        target-=coupling.T@inverse[:,0]
        reduced-=coupling.T@inverse[:,1:]
        solved.append((start,end,inverse))
    result=np.empty_like(rhs)
    result[:shared]=cho_solve(cho_factor((reduced+reduced.T)*.5,check_finite=False),target,check_finite=False)
    for start,end,inverse in solved:
        result[start:end]=inverse[:,0]-inverse[:,1:]@result[:shared]
    return result


def _linear_step(matrix,rhs,**options):
    # The regularized sparse normal matrix has local cycle blocks or a
    # small shared-angle block. Factor it once instead of thousands of
    # Python-driven iterations. Reject inaccurate/singular factorizations.
    if issparse(matrix) and not options.get('damp',0.):
        transpose=matrix.T.tocsc();normal=(transpose@matrix).tocsc()
        target=transpose@rhs
        try:
            x=_normal_solve(normal,target)
            error=np.linalg.norm(transpose@(matrix@x-rhs))
            if np.isfinite(x).all() and error<=1e-8*(1+np.linalg.norm(target)):
                return (x,)
        except RuntimeError:
            pass
    return lsmr(matrix,rhs,**options)


@lru_cache(maxsize=32)
def _solver(direct,blocks=None):
    linear=_linear_step
    if blocks is not None:
        def normal(matrix,rhs):
            if matrix.shape[0]<=640:return _normal_solve(matrix,rhs)
            try:return _arrow_solve(matrix,rhs,blocks)
            except np.linalg.LinAlgError:return _normal_solve(matrix,rhs)
        linear=_copy(_linear_step,dict(_linear_step.__globals__,_normal_solve=normal))
    namespace=dict(vars(trust_region),lsmr=linear if direct else lsmr,right_multiplied_operator=_right_scaled,
                   regularized_lsq_operator=_regularized)
    namespace['trf_bounds']=_copy(trust_region.trf_bounds,namespace)
    trf=_copy(trust_region.trf,namespace)
    namespace=dict(scipy_least_squares.__globals__,trf=trf)
    return _copy(scipy_least_squares,namespace)


def least_squares(*args,direct=False,blocks=None,**kwargs):
    if kwargs.get('tr_solver')!='lsmr' or kwargs.get('method','trf')!='trf':
        raise ValueError('explicit sparse bounded trust-region solve required')
    return _solver(direct,None if blocks is None else tuple(blocks))(*args,**kwargs)
