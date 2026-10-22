"""Experimental constrained inverse of a complete 10 ms contact response.

Uses observations and issued commands only. A local derivative proposes an
input update; the same nonlinear response checks it before acceptance.
This does not certify whole-task reachability or continuous-time safety.
"""
import time

import clarabel
import mujoco
import numpy as np
import scipy.sparse as sp

from build.contact_step_control import pad_gap_jacobians
from build.short_contact_response import ShortContactResponse
from amsrr_det.clearance import MotionBoundedClearance


class ResponseContactControl:
    def __init__(self, plant, links, period=.01, iterations=3, clearance_model=None,
                 objective='tracking',prediction_steps=40,fast_solve=False,adaptive_damping=False,distance_closing=False):
        self.plant, self.m, self.links, self.h = plant, plant.mj, links, period
        m = self.m
        self.predictor = ShortContactResponse(plant, links, period=period,steps=prediction_steps)
        self.fast_solve=fast_solve
        self.damping=1e-3 if adaptive_damping else 0.
        self.adaptive_damping=adaptive_damping
        self.distance_closing=distance_closing
        self.query = mujoco.MjData(m)
        self.groups = self.predictor.runner.groups
        self.groups = [tuple(np.flatnonzero(self.groups == j)) for j in range(len(links))]
        self.nj = len(plant.meta['actuators'])
        ids = [m.joint(a['name']).id for a in plant.meta['actuators']]
        self.qi, self.vi = m.jnt_qposadr[ids], m.jnt_dofadr[ids]
        self.qlo, self.qhi = m.jnt_range[ids].T
        self.gimbal = np.array([a['role'] != 'dock' for a in plant.meta['actuators']])
        self.effort = m.actuator_forcerange[:self.nj, 1]
        self.kp = -m.actuator_biasprm[:self.nj, 1]
        self.kd = -m.actuator_biasprm[:self.nj, 2]
        self.gain = m.actuator_gainprm[:, 0]
        self.rotor_limit = np.array([r['maximum'] for r in plant.meta['rotors']])
        self.input_scale = np.r_[self.effort, self.rotor_limit] / self.gain
        self.speed = np.full(m.nv, 3.)
        for j in np.flatnonzero(m.jnt_type == mujoco.mjtJoint.mjJNT_FREE):
            k = m.jnt_dofadr[j]
            self.speed[k:k+6] = [.02]*3 + [.1]*3
        self.iterations = iterations
        if objective not in ['tracking','release_load_motion']:
            raise ValueError('unknown response objective')
        self.objective=objective
        self.minimum_normal_load=0.
        self.previous = None
        self.clearance = None if clearance_model is None else MotionBoundedClearance(clearance_model, links)

    def update(self, q, v, reference, velocity=None, input_reference=None, normal_targets=None):
        started = time.perf_counter()
        m, n, h = self.m, self.m.nu, self.h
        d = self.query
        d.qpos[:] = q; d.qvel[:] = v
        mujoco.mj_fwdPosition(m, d)
        if self.objective=='release_load_motion':
            mass=np.empty((m.nv,m.nv));mujoco.mj_fullM(m,d,mass)
            mass_root=np.linalg.cholesky(mass).T
        gaps, J = pad_gap_jacobians(m, d, self.links, self.groups)
        error = np.empty(m.nv)
        mujoco.mj_differentiatePos(m, error, 1., q, reference)
        desired = error/.5 + (np.zeros(m.nv) if velocity is None else velocity)
        closing_limit=np.maximum(.0002,np.maximum(gaps,0.)/.5) if self.distance_closing else .0002
        normal_velocity = np.clip((-np.asarray(normal_targets)-gaps)/.5, -closing_limit, .0002)
        anchor = np.asarray(input_reference)
        current = anchor.copy() if self.previous is None else self.previous.copy()
        # The previous command can lie outside the NEW observation's input
        # bounds. Start the numerical search inside those bounds; otherwise a
        # unit trust region may have an empty intersection with the input box.
        # This changes no observed state and sends no command before validation.
        offset = self.kp*q[self.qi] + self.kd*v[self.vi]
        low = np.r_[(-self.effort+offset)/self.gain[:self.nj], np.zeros(n-self.nj)]
        high = np.r_[(self.effort+offset)/self.gain[:self.nj], self.rotor_limit]
        low[:self.nj][self.gimbal] = np.maximum(low[:self.nj][self.gimbal], self.qlo[self.gimbal])
        high[:self.nj][self.gimbal] = np.minimum(high[:self.nj][self.gimbal], self.qhi[self.gimbal])
        limited = m.actuator_ctrllimited.astype(bool)
        low[limited] = np.maximum(low[limited], m.actuator_ctrlrange[limited, 0])
        high[limited] = np.minimum(high[limited], m.actuator_ctrlrange[limited, 1])
        if np.any(low > high):raise ValueError('observed state has an empty initial input domain')
        current = np.clip(current, low, high)
        logs = []
        radius = 1.

        def value(response, command):
            # Actual predicted displacement, not desired fictitious contact forces.
            dq = np.empty(m.nv)
            mujoco.mj_differentiatePos(m, dq, 1., q, response['q'])
            if self.objective=='release_load_motion':
                # A common optimization cost, NOT the engine's stored energy
                # or a Lyapunov proof. The nominal contact stiffness fixes the
                # force weighting; no case-dependent fitted weight is used.
                residual=np.r_[mass_root@response['v'],
                    response['loads'][-1,:,0]/np.sqrt(7500.),
                    1e-6*(command-anchor)/self.input_scale]
            else:
                residual = np.r_[(response['v']-desired)/self.speed,
                                  (J@dq/h-normal_velocity)/.0002,
                                  .001*(command-anchor)/self.input_scale]
            states = response['states']
            qs, vs = states[:, 1:1+m.nq], states[:, 1+m.nq:]
            raw = command[:self.nj]*self.gain[:self.nj] - self.kp*qs[:, self.qi] - self.kd*vs[:, self.vi]
            initial_raw = command[:self.nj]*self.gain[:self.nj]-self.kp*q[self.qi]-self.kd*v[self.vi]
            ratios = [np.linalg.norm(response['loads'][:, :, 1:4], axis=2)/30.,
                      np.linalg.norm(response['loads'][:, :, 4:7], axis=2)/5.,
                      abs(initial_raw)/self.effort,
                      abs(response['motors'][:, 0, :self.nj])/self.effort,
                      abs(vs[:, self.vi])/3., response['penetration']/.002]
            violation = max(float(np.max(a, initial=0.))-1. for a in ratios)
            if self.minimum_normal_load>0:
                violation=max(violation,float(np.max(self.minimum_normal_load-response['loads'][-1,:,0],initial=0.))/self.minimum_normal_load)
            violation = max(violation, float(np.max(self.qlo-qs[:, self.qi], initial=0.)),
                            float(np.max(qs[:, self.qi]-self.qhi, initial=0.)), 0.)
            violation = max(violation, float(np.max(-command[self.nj:]/self.rotor_limit, initial=0.)),
                            float(np.max(command[self.nj:]/self.rotor_limit-1., initial=0.)),
                            float(np.max(self.qlo[self.gimbal]-command[:self.nj][self.gimbal], initial=0.)),
                            float(np.max(command[:self.nj][self.gimbal]-self.qhi[self.gimbal], initial=0.)))
            if self.clearance is not None:
                bad,_,_=self.clearance.check_many(np.vstack([q,qs]),True)
                if bad:violation=max(violation,1.)
            return .5*float(residual@residual), violation, residual

        accepted = None
        for iteration in range(max(self.iterations,8)):
            phase_start = time.perf_counter()
            response = self.predictor.evaluate(q, v, current, True)
            prediction_s = time.perf_counter()-phase_start
            cost, violation, residual = value(response, current)
            if violation <= 1e-6:
                accepted = (current.copy(), cost, response)
            D = response['derivatives']*self.input_scale
            if self.objective=='release_load_motion':
                A=np.vstack([mass_root@D[-1,m.nv:],
                    response['load_derivatives'][-1,:,0]*self.input_scale/np.sqrt(7500.),
                    1e-6*np.eye(n)])
            else:
                A = np.vstack([D[-1, m.nv:]/self.speed[:, None],
                               J@D[-1, :m.nv]/(h*.0002), .001*np.eye(n)])
            P = A.T@A; c = A.T@residual
            # Levenberg-Marquardt regularizes the LOCAL input increment, not
            # the task cost. Adapt it from actual/predicted cost reduction.
            solve_P=P+self.damping*max(float(np.diag(P).max()),1e-12)*np.eye(n)
            matrices, bounds, cones = [], [], []

            def bounded(matrix, low_bound, high_bound):
                # Exact extrema over the actual input/trust box, including
                # asymmetric actuator headroom and a shrunken trust radius.
                smallest=np.minimum(matrix*lower,matrix*upper).sum(axis=1)
                largest=np.maximum(matrix*lower,matrix*upper).sum(axis=1)
                upper_needed,lower_needed=largest>high_bound,smallest<low_bound
                for needed, a, b in [(upper_needed, matrix, high_bound), (lower_needed, -matrix, -low_bound)]:
                    if np.any(needed):
                        matrices.append(a[needed]); bounds.append(b[needed])
                        cones.append(clarabel.NonnegativeConeT(int(np.count_nonzero(needed))))

            lower, upper = -radius*np.ones(n), radius*np.ones(n)
            lower = np.maximum(lower, (low-current)/self.input_scale)
            upper = np.minimum(upper, (high-current)/self.input_scale)
            from build.direct_response_step import direct_step
            phase_start=time.perf_counter()
            direct=direct_step(self,solve_P,c,response,D,current,lower,upper) if self.fast_solve else None
            if direct is not None:
                change_scaled=direct;status='Solved';matrix=None;bound=None
                solver_note=dict(direct_optimum=True,checked_substeps=len(response['states']))
            else:
                matrices.extend([np.eye(n), -np.eye(n)])
                bounds.extend([upper, -lower])
                cones.extend([clarabel.NonnegativeConeT(n), clarabel.NonnegativeConeT(n)])
                # Batch the same per-substep bounds; neither collocation times nor
                # physical constraints are removed by this vectorization.
                states=response['states'];qs=states[:,1:1+m.nq];vs=states[:,1+m.nq:]
                dq,dv=D[:,:m.nv],D[:,m.nv:]
                bounded(dq[:,self.vi].reshape(-1,n),
                        (self.qlo-qs[:,self.qi]).ravel(),(self.qhi-qs[:,self.qi]).ravel())
                bounded(dv[:,self.vi].reshape(-1,n),
                        (-3.-vs[:,self.vi]).ravel(),(3.-vs[:,self.vi]).ravel())
                # Future servo saturation is part of the unchanged physical
                # response. Bound the initial issued input and verify actual
                # clipped effort at every substep; do not additionally demand
                # that the hypothetical pre-limiter effort stay unsaturated.
                if self.minimum_normal_load>0:
                    normal_change=response['load_derivatives'][-1,:,0]*self.input_scale
                    bounded(normal_change,self.minimum_normal_load-response['loads'][-1,:,0],
                            np.full(len(self.links),np.inf))
                for sl,limit in [(slice(1,4),30.),(slice(4,7),5.)]:
                    loads=response['loads'][:,:,sl]
                    derivatives=response['load_derivatives'][:,:,sl]*self.input_scale
                    possible=(np.linalg.norm(loads,axis=2)
                              +np.linalg.norm(derivatives,axis=2)@np.maximum(abs(lower),abs(upper)))
                    for k,pad in np.argwhere(possible>limit):
                        matrices.append(np.vstack([np.zeros(n),-derivatives[k,pad]]))
                        bounds.append(np.r_[limit,loads[k,pad]])
                        cones.append(clarabel.SecondOrderConeT(4))
                settings = clarabel.DefaultSettings(); settings.verbose = False
                settings.tol_gap_abs = settings.tol_gap_rel = settings.tol_feas = 1e-7
                phase_start = time.perf_counter()
                cost_scale=1/max(np.diag(P).max(),1e-12) if self.objective=='release_load_motion' else 1.
                matrix,bound=np.vstack(matrices),np.concatenate(bounds)
                solver_note={}
                if self.fast_solve:
                    from build.held_input_solver import solve
                    try:
                        change_scaled,solver_note=solve(solve_P*cost_scale,c*cost_scale,matrix,bound,cones)
                        status='Solved'
                    except (ValueError,np.linalg.LinAlgError):status='fallback'
                else:status='fallback'
                if status=='fallback':
                    solution=clarabel.DefaultSolver(sp.csc_matrix(np.triu(solve_P*cost_scale)),c*cost_scale,
                        sp.csc_matrix(matrix),bound,cones,settings).solve()
                    status=str(solution.status);change_scaled=np.array(solution.x)
            solver_s = time.perf_counter()-phase_start
            if status not in ['Solved', 'AlmostSolved']:
                logs.append(dict(solver=status, cost=cost, violation=violation))
                break
            change = self.input_scale*change_scaled
            self.last_problem = None if matrix is None else dict(P=P, c=c, matrix=matrix, bound=bound, cones=cones)
            improved = False
            for alpha in [1., .5, .25, .125]:
                proposal = current+alpha*change
                trial = self.predictor.evaluate(q, v, proposal)
                next_cost, next_violation, _ = value(trial, proposal)
                # A feasible current input cannot be replaced by an infeasible one.
                valid = (next_violation <= 1e-6 and next_cost <= cost+1e-10)
                repairing = violation > 1e-6 and next_violation < violation*.9
                if valid or repairing:
                    current = proposal; improved = True
                    if next_violation <= 1e-6:
                        accepted = (proposal.copy(), next_cost, trial)
                    break
            logs.append(dict(cost=cost, violation=violation, improved=improved,
                             next_cost=next_cost, next_violation=next_violation,
                             prediction_s=prediction_s, solver_s=solver_s, linear_rows=sum(len(x) for x in bounds),radius=radius,
                             solver_note=solver_note))
            if self.adaptive_damping:
                step_scaled=alpha*change_scaled
                predicted=-(c@step_scaled+.5*step_scaled@P@step_scaled)
                agreement=((violation-next_violation)/max(violation,1e-15) if violation>1e-6
                           else (cost-next_cost)/max(float(predicted),1e-15))
                if not improved:self.damping=min(self.damping*10.,1e8)
                elif violation>1e-6 or predicted>1e-12:
                    if agreement>.75:self.damping=max(self.damping/3.,1e-12)
                    elif agreement<.25:self.damping=min(self.damping*4.,1e8)
                logs[-1].update(damping=self.damping,agreement=float(agreement))
            self.last_command = current.copy()
            if iteration+1>=self.iterations:
                if accepted is not None:break
            if not improved:
                if not self.adaptive_damping:radius *= .25
                continue
            if np.max(abs(alpha*change/self.input_scale)) < 1e-4:
                break
        self.diagnostics = dict(objective=self.objective,iterations=logs, wall_s=time.perf_counter()-started)
        if accepted is None:
            raise ValueError('complete contact inverse found no feasible held input: '+str(logs))
        command, cost, response = accepted
        self.previous = command.copy()
        self.response = response
        self.predicted_q = response['q'].copy()
        self.predicted_v = response['v'].copy()
        self.predicted_loads = response['loads'][-1,:,0].copy()
        self.diagnostics.update(cost=cost, predicted_velocity=response['v'].tolist(),
                                peak_force=float(np.linalg.norm(response['loads'][:, :, 1:4], axis=2).max()))
        return command
