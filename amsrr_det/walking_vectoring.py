"""Periodic rotor directions under actual angle, rate and load constraints.

Each convex step proposes gimbal-angle increments and support forces. Exact
moving-body dynamics, CAD and disturbance screens decide acceptance. Critical
load phases refine the next step; no forward task simulation selects a plan.
"""
import numpy as np
import scipy.sparse as sp
import clarabel
import pinocchio as pin
import mujoco
from scipy.interpolate import CubicSpline
from .rotor_distribution import rotor_action_basis, rotor_basis_actions
from .equilibrium import point_jacobian, contact_basis, static_balance
from .walking_load import motion_load, ground_contacts, support_switch_times
from .walking_cycle import CycleReference
from .walking_gait import validate_cycle

def inspect_proposal(plant, controller, spec, gait, screen_motion, complete_check=True):
    geometry = validate_cycle(plant, gait, spec=spec)
    if not geometry['passed']:
        return (None, np.empty(0), dict(reason='geometry', geometry=geometry))
    ref = CycleReference(plant, spec, gait)
    # Reject a failing proposal on a coarse subset first. Acceptance still
    # requires every one of the 96 phases and the unchanged margin screen.
    scores = np.full(96,np.nan)
    times=np.linspace(0,ref.period,97)[:-1]
    for index in np.r_[np.arange(0,96,3),[i for i in range(96) if i%3]]:
        t=times[index]
        (q, v, a) = ref.rates(t)
        (qp, load) = motion_load(plant, controller.drive, q, v, a)
        contacts = ground_contacts(controller, ref.feet(t)[3])
        r = static_balance(plant.pin, plant.meta, qp, contacts, generalized_load=load, fast_lp=True)
        scores[index]=r.get('minimum_peak_effort_ratio', 1000000.0)
        if index==93 and np.nanmax(scores)>.9:break
    scores = np.asarray(scores)
    if np.nanmax(scores) <= 0.9:
        if not complete_check:
            return None, np.empty(0), dict(reason='support_objective_not_converged',max_peak=float(np.nanmax(scores)))
        try:
            return (screen_motion(plant, controller, spec, gait, scores), np.empty(0), dict(geometry=geometry))
        except ValueError as e:
            return (None, np.empty(0), dict(reason=str(e), max_peak=float(np.nanmax(scores))))
    ids=np.flatnonzero(np.isfinite(scores))
    candidates = [int(i) for j,i in enumerate(ids) if scores[i]>.9 and scores[i]>=scores[ids[(j-1)%len(ids)]] and scores[i]>=scores[ids[(j+1)%len(ids)]]]
    selected = sorted(candidates, key=lambda i: scores[i], reverse=True)[:2]
    return (None, np.array(selected) / 96, dict(reason='periodic_actuator_load', max_peak=float(np.nanmax(scores))))


def _direction_iterations(plant, controller, spec, gait, screen_motion, iterations, restore=False, improve=False):
    meta = plant.meta
    m = plant.pin
    data = m.createData()
    nr = len(meta['rotors'])
    W = meta['mass'] * 9.80665
    radius=max(.2,float(np.max(np.linalg.norm(gait['feet'][:,:2]-gait['origin'][:2],axis=1))))
    free=6 if restore else 0
    offsets = np.asarray(gait.get('phases', np.arange(len(gait['feet'])) / len(gait['feet'])))
    switch_phase = np.r_[-offsets, gait.get('duty', 0.62) - offsets] % 1.0
    phases = np.sort(np.r_[np.arange(16) / 16, switch_phase])
    phases = phases[np.r_[True, np.diff(phases) > 1e-08]]
    times = phases * gait['period']
    nt = len(times)
    interval = np.diff(np.r_[times, gait['period']])
    weights = (interval + np.roll(interval, 1)) / (2 * gait['period'])
    av = np.array([a['v'] for a in meta['actuators']])
    lim = np.array([a['effort'] for a in meta['actuators']])
    nj = len(av)
    act = [meta['actuators'][r['gimbal']] for r in meta['rotors']]
    lower = np.array([a['lower'] + 0.03 for a in act])
    upper = np.array([a['upper'] - 0.03 for a in act])
    speed = np.array([a['speed'] for a in act])
    maximum = np.array([r['maximum'] for r in meta['rotors']])
    mq = controller.drive.qi[np.array([r['gimbal'] for r in meta['rotors']])]
    new = dict(gait, configurations=gait['configurations'].copy())
    prev = np.full((nt, nr), W * 0.5 / nr)
    if improve:
        ref=CycleReference(plant,spec,gait)
        for k,t in enumerate(times):
            q,v,a=ref.rates(t);qp,load=motion_load(plant,controller.drive,q,v,a)
            report=static_balance(m,meta,qp,ground_contacts(controller,ref.feet(t)[3]),
                                  generalized_load=load,fast_lp=True,minimize_burden=True)
            if report.get('status')=='static_screen_pass':prev[k]=np.maximum(report['thrust'],0.)
    history = []
    geometry_cache = {}
    extra_phases = []
    for iteration in range(iterations):
        if extra_phases:
            old_times = times.copy()
            old_prev = prev.copy()
            phases = np.sort(np.r_[times / gait['period'], extra_phases])
            phases = phases[np.r_[True, np.diff(phases) > 1e-08]]
            times = phases * gait['period']
            nt = len(times)
            interval = np.diff(np.r_[times, gait['period']])
            weights = (interval + np.roll(interval, 1)) / (2 * gait['period'])
            prev = np.clip(CubicSpline(np.r_[old_times, gait['period']], np.vstack([old_prev, old_prev[0]]), bc_type='periodic')(times), 0.0, maximum)
            geometry_cache = {}
            extra_phases = []
        ref = CycleReference(plant, spec, new)
        eq = []
        ineq = []
        beq = []
        bineq = []
        maps = []
        grounds = []
        widths = []
        diags = []
        currents = []
        for (k, t) in enumerate(times):
            (q, v, a) = ref.rates(t)
            (qp, load) = motion_load(plant, controller.drive, q, v, a)
            theta = q[mq]
            currents.append(theta)
            if k not in geometry_cache:
                stance = ref.feet(t - 1e-07)[3] & ref.feet(t + 1e-07)[3]
                contacts = ground_contacts(controller, stance)
                basis = rotor_action_basis(m, meta, qp)
                pin.framesForwardKinematics(m, data, qp)
                C = np.column_stack([point_jacobian(m, data, qp, meta['frames'][c['frame']], c['point']).T @ contact_basis(c) for c in contacts])
                geometry_cache[k] = (basis, C, len(contacts))
            (basis, C, nc) = geometry_cache[k]
            B = rotor_basis_actions(basis, theta)
            D = (-basis[1] * np.sin(theta) + basis[2] * np.cos(theta)) * prev[k]
            F = np.c_[D, B, C, np.eye(m.nv,free), np.zeros((m.nv, nj))]
            size = F.shape[1]
            widths.append(size)
            slack = 2 * nr + 3 * nc
            eq.append(sp.csr_matrix(F[:6]))
            beq.extend(load[:6])
            A = [F[av].copy(), -F[av].copy()]
            A[0][:, slack+free:] = -np.eye(nj)
            A[1][:, slack+free:] = -np.eye(nj)
            b = [load[av] + 0.8 * lim, -load[av] + 0.8 * lim]
            dr = controller.drive
            d = dr.d
            R = d.xmat[controller.floor_bids].reshape(-1, 3, 3)
            corners = controller.floor_centers - np.sign(R[:, 2]) * controller.floor_halves
            lowest = d.xpos[controller.floor_bids] + np.einsum('bij,bj->bi', R, corners)
            jacobian = np.empty((3, plant.mj.nv))
            gvi = dr.vi[np.array([z['gimbal'] for z in meta['rotors']])]
            for gi in np.flatnonzero(lowest[:, 2] < 0.08):
                mujoco.mj_jac(plant.mj, d, jacobian, None, lowest[gi], int(controller.floor_bids[gi]))
                gradient = jacobian[2, gvi]
                if np.max(abs(gradient)) < 1e-09:
                    continue
                row = np.zeros((1, size))
                row[0, :nr] = -gradient
                A.append(row)
                b.append(np.array([lowest[gi, 2] - 0.001]))
            lo = np.r_[np.maximum(lower - theta, -0.6), np.zeros(nr), np.tile([0.0, -np.inf, -np.inf], nc), np.full(free,-np.inf), np.zeros(nj)]
            hi = np.r_[np.minimum(upper - theta, 0.6), maximum, np.full(3 * nc + free + nj, np.inf)]
            ident = np.eye(size)
            il = np.isfinite(lo)
            ih = np.isfinite(hi)
            A.extend([-ident[il], ident[ih]])
            b.extend([-lo[il], hi[ih]])
            for i in range(nc):
                row = np.zeros((4, size))
                j = 2 * nr + 3 * i
                row[:, j:j + 3] = [[-0.8, 1, 1], [-0.8, 1, -1], [-0.8, -1, 1], [-0.8, -1, -1]]
                A.append(row)
                b.append(np.zeros(4))
            ineq.append(sp.csr_matrix(np.vstack(A)))
            bineq.extend(np.concatenate(b))
            maps.append(sp.hstack([sp.eye(nr), sp.csr_matrix((nr, size - nr))]))
            sr = np.zeros(size)
            sr[2 * nr:slack:3] = 1.0
            grounds.append(sr)
            diags.extend(np.r_[np.full(nr, 0.02), np.full(nr + 3 * nc, 1e-05), (1e6/np.r_[np.full(3,W),np.full(3,W*radius)]**2)[:free], 10000.0 / lim ** 2])
        E = sp.block_diag(eq, format='csc')
        A = sp.block_diag(ineq, format='csc')
        S = sp.block_diag(maps, format='csc')
        size = sum(widths)
        current = np.array(currents).ravel()
        D1 = sp.lil_matrix((nt, nt))
        D2 = sp.lil_matrix((nt, nt))
        for k in range(nt):
            D1[k, k] = -1 / interval[k]
            D1[k, (k + 1) % nt] = 1 / interval[k]
            factor = 2 / (interval[k] + interval[k - 1])
            D2[k, k] = -factor * (1 / interval[k] + 1 / interval[k - 1])
            D2[k, (k + 1) % nt] = factor / interval[k]
            D2[k, (k - 1) % nt] = factor / interval[k - 1]
        velocity = sp.kron(D1, sp.eye(nr))
        accel = sp.kron(D2, sp.eye(nr))
        R = velocity @ S
        H = accel @ S
        P = sp.diags(diags, format='csc') + 0.05 * (R.T @ R) + 0.001 * (H.T @ H)
        cost = 0.05 * (R.T @ (velocity @ current)) + 0.001 * (H.T @ (accel @ current))
        # T**1.5 is concave in T**2. Its tangent gives a quadratic upper
        # model touching the previous thrust (ordinary T tangents do not).
        # First recover feasible directions, then improve the same burden.
        energy_weight=200. if improve else 20. if iteration>=2 else 0.
        rotor_diagonal=np.zeros(size);offset=np.r_[0,np.cumsum(widths)]
        hover=W**1.5/np.sqrt(nr)
        for k,start in enumerate(offset[:-1]):
            rotor_diagonal[start+nr:start+2*nr]=energy_weight*weights[k]*1.5/np.sqrt(np.maximum(prev[k],.02*W/nr))/hover
        P+=sp.diags(rotor_diagonal,format='csc')
        ground = sp.csc_matrix(-np.concatenate([w * g for (w, g) in zip(weights, grounds)])[None])
        con = sp.vstack([E, A, R, -R, ground], format='csc')
        rhs = np.r_[beq, bineq, np.tile(0.75 * speed, nt) - velocity @ current, np.tile(0.75 * speed, nt) + velocity @ current, -0.6 * W]
        col = np.zeros(con.shape[0])
        col[-1] = -1.0
        con = sp.hstack([con, col[:, None]], format='csc')
        P = sp.block_diag([P, sp.csc_matrix([[10000.0 / W ** 2]])], format='csc')
        cost = np.r_[cost, 0.0]
        settings = clarabel.DefaultSettings()
        settings.verbose = False
        settings.max_threads = 1
        settings.direct_solve_method = 'qdldl'
        settings.max_iter = 70
        settings.tol_gap_abs = settings.tol_gap_rel = settings.tol_feas = 1e-07
        solved = clarabel.DefaultSolver(sp.triu(P, format='csc'), cost, con, rhs, [clarabel.ZeroConeT(E.shape[0]), clarabel.NonnegativeConeT(len(rhs) - E.shape[0])], settings).solve()
        if str(solved.status) not in ['Solved', 'AlmostSolved']:
            raise ValueError(str(solved.status))
        x = np.array(solved.x[:-1])
        angles = np.array(currents) + (S @ x).reshape(nt, nr)
        theta = np.vstack([angles, angles[0]])
        spline = CubicSpline(np.r_[times, gait['period']], theta, bc_type='periodic')
        new['configurations'][:, mq] = spline(np.linspace(0, gait['period'], len(new['configurations'])))
        offsets = np.r_[0, np.cumsum(widths)]
        prev = np.array([x[offsets[k] + nr:offsets[k] + 2 * nr] for k in range(nt)])
        history.append(dict(iteration=iteration, ground_deficit=solved.x[-1] / W, max_torque_slack=max((float(max(x[offsets[k + 1] - nj:offsets[k + 1]] / lim)) for k in range(nt))), change=float(abs(S @ x).max()), solve_time=solved.solve_time))
        if iteration%3==2 or iteration==iterations-1:
            (report, cuts, diagnostic) = inspect_proposal(plant, controller, spec, new, screen_motion, complete_check=True)
            history[-1]['inspection'] = diagnostic
            history[-1]['phase_count'] = nt
            if report is not None:
                return (new, dict(history=history, accepted_report=report,geometry=diagnostic.get('geometry')))
            extra_phases = [phase for phase in cuts if np.min(abs(times / gait['period'] - phase)) > 1e-08]
    return (new, dict(history=history))


def periodic_directions(plant,controller,spec,gait,screen_motion,iterations=6):
    """Fixed candidate-local refinement; all independent trials are compared.

    Temporary root-wrench residuals only repair an infeasible local model.
    The original CAD, moving-load and disturbance screens decide acceptance.
    An already known rejection is not redundantly recomputed by the caller.
    """
    restored=False
    try:
        result=_direction_iterations(plant,controller,spec,gait,screen_motion,iterations)
    except ValueError as error:
        if str(error) not in ['PrimalInfeasible','AlmostPrimalInfeasible']:raise
        result=_direction_iterations(plant,controller,spec,gait,screen_motion,iterations,restore=True)
        restored=True
    if 'accepted_report' not in result[1]:
        inspection=result[1]['history'][-1].get('inspection',{})
        if 'reason' in inspection:
            raise ValueError('directed_periodic_geometry' if inspection['reason']=='geometry' else inspection['reason'])
    if restored:result[1]['local_feasibility_restoration']=True
    return result
