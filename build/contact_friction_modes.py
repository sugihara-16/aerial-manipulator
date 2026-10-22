"""Finite friction boundary branches of the regularized contact law.

Each accepted command satisfies the weighted normal/tangent force projection,
the original dynamics and every physical bound. A relaxed solve proposes a
branch only; it never supplies an executable command. Branch search is local
and may fail to find a feasible command. No simulation is used in this solve.
"""
import time
import numpy as np
import clarabel
from build.contact_condensed_solver import scaled_recovery

H = np.c_[np.cos(np.arange(8)*np.pi/4), np.sin(np.arange(8)*np.pi/4)]
C = 4.5*np.cos(np.pi/8)
G = np.c_[np.full(8, -C), H]
VERTICES = 4.5*np.c_[np.cos((np.arange(8)+.5)*np.pi/4), np.sin((np.arange(8)+.5)*np.pi/4)]
MODES = [(), *[(j,) for j in range(8)], *[(j, (j+1)%8) for j in range(8)]]


def project_force(affine, rn, ratio):
    """Independent exact projection onto the 8-sided cone, in the R metric."""
    metric = np.array([rn, rn/ratio, rn/ratio])
    candidates = [('apex', np.zeros(3))]
    for mode in MODES:
        force = affine.copy()
        if mode:
            active = G[list(mode)]
            multiplier = np.linalg.solve((active/metric)@active.T, active@affine)
            if min(multiplier) < -1e-9:
                continue
            force -= active.T@multiplier/metric
        if force[0] >= -1e-8 and max(G@force) <= 1e-8:
            candidates.append((mode, force))
    return min(candidates, key=lambda item: float(np.sum(metric*(item[1]-affine)**2)))


def solve(P, c, A, b, cones, nv, nu, nt, pressure_matrix, Dn, p0, rn,
          ratio, observed_velocity):
    """Return the original variables and the coupled normal-force correction."""
    began = time.perf_counter()
    n, count = len(c), nt//2
    ne, ni = cones[0].dim, cones[1].dim
    tangent_rows = np.arange(nv, nv+nt)
    tangent_columns = np.arange(nv+nu, nv+nu+nt)
    alpha = A[tangent_rows, tangent_columns]

    def affine_at(x, index):
        rows = nv+np.arange(2*index, 2*index+2)
        return np.r_[p0[index]-Dn[index]@x[:nv],
                     (b[rows]-A[rows, :nv]@x[:nv])/alpha[2*index]]

    def proposed_modes(x):
        return tuple(project_force(affine_at(x, i), rn[i], ratio)[0] for i in range(count))

    # Minimize constitutive residual subject to all other original bounds.
    # This supplies only a branch proposal; exact equations are restored below.
    retained = np.r_[np.arange(nv), np.arange(nv+nt, len(b))]
    relaxed_cones = [clarabel.ZeroConeT(ne-nt), *cones[1:]]
    residual_map = A[tangent_rows]/alpha[:, None]
    residual_target = b[tangent_rows]/alpha
    queue = []
    try:
        hint, _ = scaled_recovery(residual_map.T@residual_map,
            -residual_map.T@residual_target, A[retained], b[retained], relaxed_cones)
        queue.append(proposed_modes(np.asarray(hint.x)))
    except ValueError:
        pass
    queue.append(proposed_modes(np.r_[observed_velocity, np.zeros(n-nv)]))
    # Adjacent simpler branches are useful when a relaxed solution places
    # several contacts on a boundary but only one needs to slide.
    for proposal in queue.copy():
        for index, mode in enumerate(proposal):
            if mode:
                single = [()]*count; single[index] = mode
                queue.append(tuple(single))
    tried = []
    for selected in dict.fromkeys(queue):
        multipliers = sum(len(mode) for mode in selected if mode != 'apex')
        total = n+count+multipliers
        matrix = np.c_[A, pressure_matrix, np.zeros((len(b), multipliers))]
        bound = b.copy()
        equality, equal_b, upper, upper_b, remove = [], [], [], [], []
        offset = n+count
        for index, mode in enumerate(selected):
            rows = nv+np.arange(2*index, 2*index+2)
            col = nv+nu+2*index
            factor = rn[index]/ratio/alpha[2*index]
            if mode == 'apex':
                remove.extend(rows)
                row = np.zeros(total); row[:nv] = -Dn[index]; row[n+index] = 1.
                equality.append(row); equal_b.append(-p0[index])
                for column in [col, col+1]:
                    row = np.zeros(total); row[column] = 1.
                    equality.append(row); equal_b.append(0.)
                for vertex in VERTICES:
                    row = np.zeros(total)
                    row[:nv] = -rn[index]*Dn[index]-factor*(vertex@A[rows, :nv])
                    upper.append(row); upper_b.append(-rn[index]*p0[index]-factor*float(vertex@b[rows]))
                continue
            matrix[rows] *= factor; bound[rows] *= factor
            normal = np.zeros(total); normal[n+index] = rn[index]
            for face in mode:
                matrix[rows, offset] = H[face]; normal[offset] = -C
                row = np.zeros(total); row[:nv] = C*Dn[index]
                row[col:col+2] = H[face]; row[n+index] = -C
                equality.append(row); equal_b.append(C*p0[index])
                row = np.zeros(total); row[offset] = -1.
                upper.append(row); upper_b.append(0.); offset += 1
            equality.append(normal); equal_b.append(0.)
        retained_equal = np.setdiff1d(np.arange(ne), remove)
        full_A = np.vstack([matrix[retained_equal], np.asarray(equality).reshape(-1, total),
            matrix[ne:ne+ni], np.asarray(upper).reshape(-1, total), matrix[ne+ni:]])
        full_b = np.r_[bound[retained_equal], equal_b, bound[ne:ne+ni], upper_b, bound[ne+ni:]]
        full_cones = [clarabel.ZeroConeT(len(retained_equal)+len(equality)),
                      clarabel.NonnegativeConeT(ni+len(upper)), *cones[2:]]
        try:
            answer, note = scaled_recovery(np.pad(P, ((0,total-n),(0,total-n))),
                np.r_[c, np.zeros(total-n)], full_A, full_b, full_cones)
        except ValueError as error:
            tried.append(dict(modes=selected, reason=str(error)))
            continue
        x = np.asarray(answer.x)
        error = 0.
        for index in range(count):
            affine = affine_at(x, index)
            force = np.r_[affine[0]+x[n+index], x[nv+nu+2*index:nv+nu+2*index+2]]
            projection = project_force(affine, rn[index], ratio)[1]
            error = max(error, float(np.max(abs(force-projection))))
        if error > 1e-5:
            tried.append(dict(modes=selected, projection_error=error))
            continue
        note.update(friction_modes=selected, projection_error=error, iterations=int(answer.iterations),
                    trials=len(tried)+1, wall_s=time.perf_counter()-began)
        return x[:n], x[n:n+count], note
    raise ValueError('contact friction branches unresolved: '+str(tried))
