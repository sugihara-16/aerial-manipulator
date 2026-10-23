"""Exact scalar slices of a linear actuator image, within numerical tolerance.

These bounds concern one fixed matrix and input box. They do not certify future
geometry, contact stability, or the existence of a better robot structure.
"""
import numpy as np
from scipy.sparse import csc_matrix
from scipy.optimize._highspy import _core as highs


def scalar_bound(matrix,base,direction,lower,upper,*,maximize=True,nonnegative=False):
    matrix,base,direction=map(np.asarray,(matrix,base,direction))
    equality=np.c_[matrix,-direction]
    scale=np.maximum(np.linalg.norm(equality,axis=1),1e-12)
    sparse=csc_matrix(equality/scale[:,None]);n=matrix.shape[1]
    lp=highs.HighsLp();lp.num_col_=n+1;lp.num_row_=len(base)
    lp.col_cost_=np.r_[np.zeros(n),-1. if maximize else 1.]
    lp.col_lower_=np.r_[lower,0. if nonnegative else -np.inf];lp.col_upper_=np.r_[upper,np.inf]
    lp.row_lower_=base/scale;lp.row_upper_=base/scale
    lp.a_matrix_.num_col_=n+1;lp.a_matrix_.num_row_=len(base)
    lp.a_matrix_.format_=highs.MatrixFormat.kColwise
    lp.a_matrix_.start_=sparse.indptr;lp.a_matrix_.index_=sparse.indices;lp.a_matrix_.value_=sparse.data
    solver=highs._Highs();options=highs.HighsOptions()
    options.output_flag=False;options.threads=1
    options.primal_feasibility_tolerance=1e-9;options.dual_feasibility_tolerance=1e-9
    solver.passOptions(options);solver.passModel(lp);solver.run()
    if solver.getModelStatus()!=highs.HighsModelStatus.kOptimal:return None
    sol=np.asarray(solver.getSolution().col_value);x=sol[:-1];value=float(sol[-1])
    residual=max(float(abs(matrix@x-base-direction*value).max()),float((np.asarray(lower)-x).max()),float((x-np.asarray(upper)).max()))
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


class ScalarBounds:
    """Repeated nonnegative radial bounds for one fixed actuator image.

    Reuse solver setup and its simplex basis; independently recheck every
    returned point against the original equality and input bounds.
    """
    def __init__(self,matrix,lower,upper):
        self.matrix=np.asarray(matrix);self.lower=np.asarray(lower);self.upper=np.asarray(upper);self.n=self.matrix.shape[1]
        self.scale=np.maximum(np.linalg.norm(self.matrix,axis=1),1e-12)
        sparse=csc_matrix(np.c_[self.matrix/self.scale[:,None],np.zeros(len(self.scale))])
        lp=highs.HighsLp();lp.num_col_=self.n+1;lp.num_row_=len(self.scale)
        lp.col_cost_=np.r_[np.zeros(self.n),-1.];lp.col_lower_=np.r_[lower,0.];lp.col_upper_=np.r_[upper,np.inf]
        lp.row_lower_=np.zeros(len(self.scale));lp.row_upper_=np.zeros(len(self.scale))
        lp.a_matrix_.num_col_=self.n+1;lp.a_matrix_.num_row_=len(self.scale);lp.a_matrix_.format_=highs.MatrixFormat.kColwise
        lp.a_matrix_.start_=sparse.indptr;lp.a_matrix_.index_=sparse.indices;lp.a_matrix_.value_=sparse.data
        self.solver=highs._Highs();options=highs.HighsOptions();options.output_flag=False;options.threads=1
        options.primal_feasibility_tolerance=options.dual_feasibility_tolerance=1e-9
        self.solver.passOptions(options);self.solver.passModel(lp)
    def bound(self,base,direction):
        base,direction=np.asarray(base),np.asarray(direction)
        for row,(b,w,scale)in enumerate(zip(base,direction,self.scale)):
            self.solver.changeCoeff(row,self.n,float(-w/scale));self.solver.changeRowBounds(row,float(b/scale),float(b/scale))
        self.solver.run()
        if self.solver.getModelStatus()!=highs.HighsModelStatus.kOptimal:return None
        sol=np.asarray(self.solver.getSolution().col_value);x=sol[:-1];value=float(sol[-1])
        residual=max(float(abs(self.matrix@x-base-direction*value).max()),float((self.lower-x).max()),float((x-self.upper).max()))
        return value if np.isfinite(residual)and residual<=1e-7 else None
