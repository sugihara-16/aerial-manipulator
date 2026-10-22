"""Experimental reference governor for excessive position-drive requests.

Only relaxes the existing geometric preload. No contact force is estimated.
Trial references all start from one frozen controller state and are committed
once, after rechecking the actual next command and its existing bounds.
"""
from copy import copy
import numpy as np
from scipy.optimize import lsq_linear


def update_with_preload_relief(grasp, q, v, goal, **options):
    guard = options.get('drive_guard')
    if guard is None:
        raise ValueError('preload relief requires the existing drive guard')
    if options.get('releasing') or options.get('preload') is not None:
        raise ValueError('preload relief is only for loaded grasp phases')
    grasp.geometry(q)  # Freeze this observation's material points for all trials.
    initial = grasp.preload.copy()
    step = options.get('normal_speed', .001) * grasp.dt
    kp = np.array([a['kp'] for a in grasp.docks])
    kd = np.array([a['kd'] for a in grasp.docks])
    limit = np.array([a['effort'] for a in grasp.docks])
    lower = np.array([a['lower'] for a in grasp.docks])
    upper = np.array([a['upper'] for a in grasp.docks])
    bias, previous_command = map(np.asarray, guard)
    previous_raw = kp*(previous_command-q[grasp.qi])-kd*v[grasp.vi]
    low, high = np.minimum(-limit, previous_raw), np.maximum(limit, previous_raw)
    # This is the increment already allowed by the existing speed and
    # acceleration bounds, including a known change in the bias.
    previous_bias = options.get('previous_joint_bias')
    previous_bias = bias if previous_bias is None else np.asarray(previous_bias)
    command_step = grasp.dt*grasp.scale[6:] + kd/kp*(.5*grasp.dt) + abs(bias-previous_bias)/kp
    trial_options = {k:value for k,value in options.items() if k != 'preload'}

    def trial(preload):
        candidate = copy(grasp)
        target, velocity = candidate.update(q, v, goal, preload=preload, **trial_options)
        command = np.clip(target[grasp.qi]+(bias+kd*velocity[grasp.vi])/kp, lower, upper)
        raw = kp*(command-q[grasp.qi])-kd*v[grasp.vi]
        valid = (candidate.last_diagnostics.get('reference_command_feasible', True)
                 and np.all(raw >= low-1e-9) and np.all(raw <= high+1e-9)
                 and np.all(abs(command-previous_command) <= command_step+1e-9))
        return candidate, target, velocity, raw, bool(valid)

    baseline = trial(initial)
    chosen, preload = baseline, initial
    rho = baseline[3]
    clipped = np.clip(rho, -limit, limit)
    before = float(np.sum(((rho-clipped)/limit)**2))
    attempts = 0
    if baseline[4] and before > 1e-14:
        # A one-sided difference respects the allowed opening direction. The
        # finite perturbation is 1% of this cycle's allowed preload increment.
        epsilon = .01*step
        sensitivity = []
        for i in range(len(initial)):
            shifted = initial.copy();shifted[i] -= epsilon
            sensitivity.append((rho-trial(shifted)[3]) / epsilon)
        matrix = np.column_stack(sensitivity)*step/limit[:,None]
        regularizer = 1e-8
        solution = lsq_linear(np.vstack([matrix,regularizer*np.eye(len(initial))]),
                             -np.r_[(rho-clipped)/limit,np.zeros(len(initial))],
                             bounds=(-np.ones(len(initial)),np.zeros(len(initial))),
                             method='bvls',tol=1e-12,max_iter=30)
        if solution.success:
            # Recheck the nonlinear reference map, not only its local slope.
            for fraction in [1., .5]:
                attempts += 1
                proposed = initial + fraction*step*solution.x
                candidate = trial(proposed)
                after = float(np.sum(((candidate[3]-clipped)/limit)**2))
                if candidate[4] and after < before-1e-14:
                    chosen, preload = candidate, proposed
                    break
    controller, target, velocity, raw, valid = chosen
    grasp.reference = controller.reference
    grasp.last_velocity = controller.last_velocity
    grasp.last_diagnostics = controller.last_diagnostics
    grasp.preload = preload.copy()
    grasp.last_diagnostics['preload_relief'] = dict(
        previous=initial.tolist(),selected=preload.tolist(),changed=bool(np.any(preload<initial)),
        request_before_nm=rho.tolist(),request_after_nm=raw.tolist(),
        error_before=before,error_after=float(np.sum(((raw-clipped)/limit)**2)),
        command_bounds_rechecked=valid,nonlinear_trials=attempts)
    return target, velocity
