"""Omit exactly zero SVD rows without mutating SciPy globals.

The pinned SciPy optimizer and its stopping/step rules are unchanged. Clone
only their function bindings so other solver users keep the standard path.
"""
from types import FunctionType
import numpy as np
from scipy.optimize import least_squares as original


def clone(function,**bindings):
    copied=FunctionType(function.__code__,dict(function.__globals__,**bindings),
                        function.__name__,function.__defaults__,function.__closure__)
    copied.__kwdefaults__=function.__kwdefaults__
    return copied


from scipy.linalg import svd

def compact_svd(matrix,full_matrices=False,**kwargs):
    if full_matrices or matrix.shape[0]<=matrix.shape[1]:return svd(matrix,full_matrices=full_matrices,**kwargs)
    used=np.any(matrix!=0.,axis=1);n=int(used.sum())
    if n<matrix.shape[1] or n==matrix.shape[0]:return svd(matrix,full_matrices=False,**kwargs)
    try:
        u,s,v=svd(matrix[used],full_matrices=False,**kwargs)
    except np.linalg.LinAlgError:
        # LAPACK can fail on the smaller layout even when the original
        # matrix converges. Keep the original decomposition as recovery.
        return svd(matrix,full_matrices=False,**kwargs)
    expanded=np.zeros((len(matrix),len(s)));expanded[used]=u
    return expanded,s,v


try:
    trf=original.__globals__['trf'];bounded=trf.__globals__['trf_bounds']
    least_squares=clone(original,trf=clone(trf,trf_bounds=clone(bounded,svd=compact_svd)))
except (AttributeError,KeyError,TypeError):
    # An incompatible future SciPy keeps its public, unmodified solver.
    least_squares=original
