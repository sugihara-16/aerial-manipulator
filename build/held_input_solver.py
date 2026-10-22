"""Optional fast solve with all original conic constraints checked afterwards.

The linear relaxation is used only when its solved optimum also satisfies
every omitted second-order cone. Otherwise solve the original conic problem.
This changes numerical solution, not physical limits or the objective.
"""
import time

import clarabel
import numpy as np
import proxsuite
import scipy.sparse as sp
from scipy.linalg import svd


def violation(matrix, bound, cones, x):
    slack = bound-matrix@x
    offset = 0
    residual = -np.inf
    for cone in cones:
        values = slack[offset:offset+cone.dim]; offset += cone.dim
        if type(cone).__name__ == 'NonnegativeConeT':
            residual = max(residual, float(-values.min()))
        elif type(cone).__name__ == 'ZeroConeT':
            residual = max(residual, float(abs(values).max(initial=0.)))
        else:
            residual = max(residual, float(np.linalg.norm(values[1:])-values[0]))
    return residual


def solve(P, c, matrix, bound, cones):
    started = time.perf_counter()
    n = len(c)
    # Cholesky change of coordinates preserves the original quadratic cost.
    transform = np.linalg.solve(np.linalg.cholesky(P).T, np.eye(n))
    scaled_matrix, scaled_c = matrix@transform, transform.T@c
    unconstrained=-transform@scaled_c
    residual=violation(matrix,bound,cones,unconstrained)
    if residual<=1e-9:
        return unconstrained,dict(wall_s=time.perf_counter()-started,
            relaxed_status='unconstrained_optimum',reduced_variables=n,
            used_relaxation=True,maximum_violation=residual,
            objective=float(.5*unconstrained@P@unconstrained+c@unconstrained))
    indices, equalities, offset = [], [], 0
    for cone in cones:
        if type(cone).__name__ == 'NonnegativeConeT':
            indices.extend(range(offset, offset+cone.dim))
        elif type(cone).__name__ == 'ZeroConeT':
            equalities.extend(range(offset, offset+cone.dim))
        offset += cone.dim
    # Solve the equality equations once in whitened coordinates. Only their
    # null space remains free, so redundant contact reactions cannot make the
    # iterative QP solve spend time satisfying dependent equations.
    origin=np.zeros(n);basis=np.eye(n)
    if equalities:
        equality=scaled_matrix[equalities];value=bound[equalities]
        row_scale=np.maximum(np.linalg.norm(equality,axis=1),1e-30)
        U,s,Vt=svd(equality/row_scale[:,None],full_matrices=True)
        rank=int(np.count_nonzero(s>s[0]*1e-12))
        origin=Vt[:rank].T@((U[:,:rank].T@(value/row_scale))/s[:rank])
        basis=Vt[rank:].T
    reduced=basis.shape[1]
    solver = proxsuite.proxqp.dense.QP(reduced, 0, len(indices))
    solver.settings.eps_abs = solver.settings.eps_rel = 1e-8
    solver.settings.eps_primal_inf = solver.settings.eps_dual_inf = 1e-9
    solver.settings.max_iter = 200
    solver.settings.max_iter_in = 100
    solver.init(np.eye(reduced), basis.T@(scaled_c+origin), None, None, scaled_matrix[indices]@basis,
                np.full(len(indices), -np.inf), bound[indices]-scaled_matrix[indices]@origin)
    solver.solve()
    x = transform@(origin+basis@solver.results.x)
    residual = violation(matrix, bound, cones, x)
    relaxed_status = str(solver.results.info.status)
    used_relaxation = relaxed_status.endswith('PROXQP_SOLVED') and residual <= 1e-7
    if not used_relaxation:
        settings = clarabel.DefaultSettings(); settings.verbose = False
        settings.tol_gap_abs = settings.tol_gap_rel = settings.tol_feas = 1e-9 if equalities else 1e-7
        result = clarabel.DefaultSolver(sp.csc_matrix(np.triu(P)), c,
            sp.csc_matrix(matrix), bound, cones, settings).solve()
        if str(result.status) not in ['Solved', 'AlmostSolved']:
            raise ValueError('held-input conic solve: '+str(result.status))
        x = np.array(result.x)
        residual = violation(matrix, bound, cones, x)
    return x, dict(wall_s=time.perf_counter()-started, relaxed_status=relaxed_status,reduced_variables=reduced,
                   used_relaxation=used_relaxation, maximum_violation=residual,
                   objective=float(.5*x@P@x+c@x))
