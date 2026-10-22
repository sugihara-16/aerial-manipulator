"""Exact scalar slices of a linear actuator image, within numerical tolerance.

These bounds concern one fixed matrix and input box. They do not certify future
geometry, contact stability, or the existence of a better robot structure.
"""
import numpy as np
from .linear_program import linprog


def scalar_bound(matrix,base,direction,lower,upper,*,maximize=True,nonnegative=False):
    """Extremize a in matrix*x = base + direction*a, with bounded x."""
    matrix,base,direction=map(np.asarray,(matrix,base,direction))
    equality=np.c_[matrix,-direction]
    scale=np.maximum(np.linalg.norm(equality,axis=1),1e-12)
    objective=np.r_[np.zeros(matrix.shape[1]),-1. if maximize else 1.]
    answer=linprog(objective,A_eq=equality/scale[:,None],b_eq=base/scale,
        bounds=list(zip(lower,upper))+[(0. if nonnegative else None,None)],
        options=dict(primal_feasibility_tolerance=1e-9,dual_feasibility_tolerance=1e-9))
    if not answer.success:return None
    x=answer.x[:-1];value=float(answer.x[-1])
    residual=max(float(abs(matrix@x-base-direction*value).max()),
                 float((np.asarray(lower)-x).max()),float((x-np.asarray(upper)).max()))
    if not np.isfinite(residual) or residual>1e-7:return None
    return value


def scalar_interval(matrix,base,direction,lower,upper):
    low=scalar_bound(matrix,base,direction,lower,upper,maximize=False)
    if low is None:return None
    high=scalar_bound(matrix,base,direction,lower,upper)
    return None if high is None else (low,high)


def l1_image_radius(matrix,centers,feedback,lower,upper):
    """Largest rho with center+feedback*z in the image for ||z||_1<=rho.

    Checking both signed endpoints of every feedback column is exact for this
    cross-polytope, by convexity; it is not a box of simultaneous axis extremes.
    Every center has its own input allocation, but the same actuator matrix.
    """
    radius=np.inf
    for center in centers:
        for direction in np.asarray(feedback).T:
            if np.linalg.norm(direction)<1e-14:continue
            for sign in (-1.,1.):
                value=scalar_bound(matrix,center,sign*direction,lower,upper,nonnegative=True)
                if value is None:return None
                radius=min(radius,value)
    return float(radius)
