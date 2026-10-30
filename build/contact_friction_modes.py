"""Finite friction boundary branches of the regularized contact law.

Each accepted command satisfies the weighted normal/tangent force projection,
the original dynamics and every physical bound. Observed motion and a relaxed
solve propose branches only; neither supplies an executable command. Search is local
and may fail to find a feasible command. No simulation is used in this solve.
"""
import time
from build.deadline_context import available
from functools import lru_cache
from types import SimpleNamespace
import numpy as np
import clarabel
from build.contact_condensed_solver import scaled_recovery

H = np.c_[np.cos(np.arange(8)*np.pi/4), np.sin(np.arange(8)*np.pi/4)]
C = 4.5*np.cos(np.pi/8)
G = np.c_[np.full(8, -C), H]
VERTICES = 4.5*np.c_[np.cos((np.arange(8)+.5)*np.pi/4), np.sin((np.arange(8)+.5)*np.pi/4)]
MODES = [(), *[(j,) for j in range(8)], *[(j, (j+1)%8) for j in range(8)]]


@lru_cache(maxsize=16)
def _projection_operators(ratio):
    # The common normal compliance cancels from the force projection. Only
    # the fixed friction ratio changes these tiny linear systems.
    inverse=np.array([1.,ratio,ratio]);operators=[];multipliers=[]
    for mode in MODES:
        projection=np.eye(3);dual=np.zeros((2,3))
        if mode:
            active=G[list(mode)]
            coefficient=np.linalg.solve((active*inverse)@active.T,active)
            projection-=inverse[:,None]*active.T@coefficient
            dual[:len(mode)]=coefficient
        operators.append(projection);multipliers.append(dual)
    return np.array(operators),np.array(multipliers)


def project_force(affine, rn, ratio):
    """Same weighted cone projection; batch the fixed face equations."""
    operators,dual=_projection_operators(float(ratio))
    force=operators@affine;multiplier=rn*(dual@affine)
    valid=(multiplier.min(axis=1)>=-1e-9)&(force[:,0]>=-1e-8)&((force@G.T).max(axis=1)<=1e-8)
    forces=np.vstack([np.zeros(3),force]);valid=np.r_[True,valid]
    cost=((forces-affine)**2*np.array([rn,rn/ratio,rn/ratio])).sum(axis=1)
    cost[~valid]=np.inf;index=int(np.argmin(cost))
    return ('apex' if index==0 else MODES[index-1]),forces[index]

def project_forces(affine,rn,ratio):
    """Evaluate all independent contact projections in one array operation."""
    from amsrr_det.native_qp import project
    operators,dual=_projection_operators(float(ratio))
    indices,forces=project(affine,rn,ratio,operators,dual,G)
    return tuple('apex' if i<0 else MODES[i] for i in indices),forces


def solve(P, c, A, b, cones, nv, nu, nt, pressure_matrix, Dn, p0, rn,
          ratio, observed_velocity, boundary_hint=None, separate_all=False, progress_matrix=None, progress_lower=None, preferred_modes=None, workspace=None):
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
        return project_forces(affine_all(x),rn,ratio)[0]

    tangent_map=A[tangent_rows,:nv];tangent_bound=b[tangent_rows]
    def affine_all(x):
        return np.column_stack([p0-Dn@x[:nv],((tangent_bound-tangent_map@x[:nv])/alpha).reshape(count,2)])

    # Minimize constitutive residual subject to all other original bounds.
    # This supplies only a branch proposal; exact equations are restored below.
    retained = np.r_[np.arange(nv), np.arange(nv+nt, len(b))]
    relaxed_cones = [clarabel.ZeroConeT(ne-nt), *cones[1:]]
    residual_map = A[tangent_rows]/alpha[:, None]
    residual_target = b[tangent_rows]/alpha
    def candidates():
        if preferred_modes is not None and len(preferred_modes)==count:
            yield tuple(preferred_modes), True, 'cached'
        if boundary_hint is None:
            # Current measured motion proposes a physical contact branch.
            # Try only its fast exact solve before constructing a relaxed
            # proposal. Failure retains the original full recovery below.
            yield proposed_modes(np.r_[observed_velocity, np.zeros(n-nv)]), True, 'observed'
        if boundary_hint is not None and separate_all:
            # Try the physical zero-force branch before solving a relaxed
            # proposal. It still must improve the same objective and pass
            # every original force-law and actuator check below.
            yield ('apex',)*count, False, 'detachment'
        queue = []
        if boundary_hint is None and available(.002):
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
        if boundary_hint is not None:
            # A feasible interior solution may pin a zero-force contact at zero
            # gap. Compare its unilateral, nonadhesive branch as well. The hint
            # only proposes a branch; the original force projection is rechecked.
            proposal=list(proposed_modes(boundary_hint))
            for i in range(count):
                if affine_at(boundary_hint,i)[0] <= 1e-5:
                    proposal[i]='apex'
            queue.insert(0,tuple(proposal))
            if separate_all:
                queue.insert(0,('apex',)*count)
        if boundary_hint is not None and separate_all:
            # The relaxed task optimum proposes a mode only. Every force-law
            # equation and original physical bound is restored before acceptance.
            try:
                if progress_matrix is not None:
                    seed_A=np.vstack([A[retained],-progress_matrix])
                    seed_b=np.r_[b[retained],-progress_lower]
                    seed_cones=[*relaxed_cones,clarabel.NonnegativeConeT(len(progress_lower))]
                    from build.contact_active_solver import solve as active_solve
                    hint,_=active_solve(P,c,seed_A,seed_b,seed_cones,mechanical_partition=(nv,nu))
                    queue.insert(0,proposed_modes(np.asarray(hint)))
            except (ValueError,ArithmeticError,np.linalg.LinAlgError):
                pass
        for proposal in queue:
            yield proposal, False, 'proposal'

    tried = []
    mechanical_cache=dict(base=n+count)
    seen=set()
    for selected,fast_only,source in candidates():
        if not available(.001):break
        if selected in seen:continue
        seen.add(selected)
        from amsrr_det.native_qp import branch
        encoded=[-1 if mode=='apex' else MODES.index(mode) for mode in selected]
        full_A,full_b,eq_count,ineq_count=branch(A,b,nv,nu,ne,ni,pressure_matrix,Dn,p0,rn,alpha,ratio,H,VERTICES,C,encoded)
        total=full_A.shape[1]
        full_cones=[clarabel.ZeroConeT(eq_count),clarabel.NonnegativeConeT(ineq_count),*cones[2:]]
        try:
            full_P=np.pad(P,((0,total-n),(0,total-n)));full_c=np.r_[c,np.zeros(total-n)]
            # Boundary multipliers have zero cost, but their equations can
            # remove all zero-curvature directions. Use the checked fast QP
            # when positive curvature remains; otherwise retain recovery.
            from build.contact_active_solver import solve as active_solve
            try:
                values,note=active_solve(full_P,full_c,full_A,full_b,full_cones,
                    mechanical_partition=(nv,nu),workspace=workspace,mechanical_cache=mechanical_cache)
                answer=SimpleNamespace(x=values,iterations=note['iterations'])
                note['active_set']=True
            except (ValueError,ArithmeticError):
                if fast_only:
                    # Cheap cached/observed proposals must not trigger expensive
                    # recovery. Ordinary proposals may retry them afterwards.
                    seen.remove(selected)
                    tried.append(dict(modes=selected,reason=source+' fast solve unresolved'))
                    continue
                if boundary_hint is not None:
                    # An already feasible input exists. Failure to find an
                    # optional improvement does not justify a slow recovery.
                    # Required feasibility searches keep their full recovery.
                    tried.append(dict(modes=selected,reason='optional branch fast solve unresolved'))
                    continue
                answer,note=scaled_recovery(full_P,full_c,full_A,full_b,full_cones)
        except ValueError as error:
            tried.append(dict(modes=selected, reason=str(error)))
            continue
        x = np.asarray(answer.x)
        affine=affine_all(x)
        force=np.column_stack([affine[:,0]+x[n:n+count],x[nv+nu:nv+nu+nt].reshape(count,2)])
        projection=project_forces(affine,rn,ratio)[1]
        error=float(np.max(abs(force-projection),initial=0.))
        if error > 1e-5:
            if fast_only:seen.remove(selected)
            tried.append(dict(modes=selected, projection_error=error))
            continue
        if boundary_hint is not None:
            value=.5*x[:n]@P@x[:n]+c@x[:n]
            previous=.5*boundary_hint@P@boundary_hint+c@boundary_hint
            if value>=previous-1e-10:
                tried.append(dict(modes=selected,reason='feasible but not better than current task input'))
                continue
        note.update(friction_modes=selected, projection_error=error, iterations=int(answer.iterations),
                    trials=len(tried)+1, preferred_mode_accepted=source=='cached',
                    observed_mode_accepted=source=='observed', wall_s=time.perf_counter()-began)
        return x[:n], x[n:n+count], note
    raise ValueError('contact friction branches unresolved: '+str(tried))
