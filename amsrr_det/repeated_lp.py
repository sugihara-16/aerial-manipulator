"""Reusable dense LP workspace; each answer is checked in original units."""
import numpy as np
from scipy.sparse import csc_matrix
from scipy.optimize._highspy import _core as highs


class RepeatedLinearProgram:
    def __init__(self):
        self.solver=highs._Highs();options=highs.HighsOptions()
        options.output_flag=False;options.threads=1
        options.primal_feasibility_tolerance=options.dual_feasibility_tolerance=1e-9
        self.solver.passOptions(options);self.basis=None;self.shape=None

    def load(self,c,A,row_lower,row_upper,lower,upper):
        A=np.asarray(A);nr,nc=A.shape
        if self.shape!=A.shape:
            self.shape=A.shape;self.basis=None
        self.A,self.rl,self.ru,self.lo,self.hi=map(np.asarray,(A,row_lower,row_upper,lower,upper))
        lp=highs.HighsLp();lp.num_col_=nc;lp.num_row_=nr
        lp.col_cost_=c;lp.col_lower_=lower;lp.col_upper_=upper
        scale=np.maximum(np.linalg.norm(A,axis=1),1e-12)
        lp.row_lower_=self.rl/scale;lp.row_upper_=self.ru/scale
        matrix=lp.a_matrix_;matrix.num_col_=nc;matrix.num_row_=nr
        matrix.format_=highs.MatrixFormat.kColwise
        sparse=csc_matrix(A/scale[:,None])
        matrix.start_=sparse.indptr;matrix.index_=sparse.indices;matrix.value_=sparse.data
        self.solver.passModel(lp)
        if self.basis is not None:self.solver.setBasis(self.basis)

    def solve(self):
        self.solver.run()
        if self.solver.getModelStatus()!=highs.HighsModelStatus.kOptimal:
            self.basis=None;return None
        x=np.asarray(self.solver.getSolution().col_value);values=self.A@x
        residual=max(np.max(self.rl-values,initial=0.),np.max(values-self.ru,initial=0.),
                     np.max(self.lo-x,initial=0.),np.max(x-self.hi,initial=0.))
        if not np.isfinite(x).all() or not np.isfinite(residual) or residual>1e-7:
            self.basis=None;return None
        self.basis=self.solver.getBasis()
        return x

    def change_cost(self,column,value):
        self.solver.changeColCost(column,value)
