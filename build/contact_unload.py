"""Open the grasp under finite contact and support constraints, then withdraw.

The placed object's pose remains the reference; the robot's old closed pose
must not oppose opening. Handoff uses observed geometric separation, not time
alone. Loads and final object motion remain independently audited.
"""
import numpy as np
import mujoco
from build.background_withdrawal_hold import HeldBackgroundWithdrawal
from build.contact_step_control import ContactStepControl, pad_gap_jacobians
from build.unloaded_hover import command as gravity_command
from build.sticking_design import LocalGrasp
from build.check_local_pickup import smooth


class ContactUnload:
    def __init__(self, plan, plant, q, *, task_duration=110., **kwargs):
        self.plan, self.plant, self.q = plan, plant, q.copy()
        self.options = kwargs
        self.links = [f'm{i}/' + ('yaw_dock_mech' if p.startswith('yaw') else 'pitch_dock_mech') + p[-1]
                      for i, p in plan.surfaces]
        self.local = LocalGrasp(plant, q, self.links)
        self.control = ContactStepControl(
            plant, self.links, period=.0025, regularized_friction=True,
            finite_support=True, distance_closing=True, condensed_solve=True,
            inertial_updates=True)
        from build.separation_rate_constraints import SeparationRateConstraints
        self.control.separation = SeparationRateConstraints(plan.plant.mj, self.links, all_geometry=True)
        limits = np.r_[[.02] * 3, [.1] * 3]
        self.control.object_velocity_bounds = (-limits, limits)
        # The 1 mm opening target uses the same 0.5 s restoration time.
        # The slow 0.2 mm/s cap is for acquiring contact, not releasing it.
        self.control.opening_speed = .001 / .5
        self.control.release_drift_horizon = float(task_duration)
        self.oi = int(self.control.m.jnt_qposadr[self.control.m.joint('object').id])
        self.elapsed = 0.
        self.release = self.first = None
        self.ready = self.complete = False
        self.info = {}
        from build.held_input_guard import HeldInputGuard
        # Anticipate release transients beyond one command hold. Commands still
        # update every 2.5 ms; the 20 ms preview uses the physical time step.
        self.guard = HeldInputGuard(plant, self.links, plan.plant.mj, period=.02, workers=16)
        self.response_fallbacks = 0
        self.withdrawal_guard = None

    @property
    def control_period(self):
        return .0025 if self.release is None else .01

    def update(self, q, v, previous):
        if self.release is not None:
            command = self.release.update(q, v, previous)
            if self.withdrawal_guard is None:
                from build.held_input_guard import HeldInputGuard
                self.withdrawal_guard = HeldInputGuard(self.plant, self.links, self.plan.plant.mj, period=.01)
            command = self.withdrawal_guard.update(q, v, command)
            self.ready, self.complete = self.release.ready, self.release.complete
            self.info = dict(self.release.info, withdrawal_hold_repairs=self.withdrawal_guard.repairs)
            return command
        if self.first is None:
            self.first = previous.copy()
            self.control.d.qpos[:] = q
            mujoco.mj_fwdPosition(self.control.m, self.control.d)
            gaps = pad_gap_jacobians(self.control.m, self.control.d, self.links, self.control.groups)[0]
            # Release must not ask the pads to preserve their initial compression.
            # Synchronize a non-compressing target with the existing load transfer.
            self.open_start = np.maximum(gaps, 0.)
        empty = gravity_command(self.local, q, previous)[0]
        ratio = smooth(self.elapsed / 2.)
        reference = q.copy()
        reference[self.oi:self.oi + 7] = self.q[self.oi:self.oi + 7]
        self.control.previous_input = previous.copy()
        try:
            command = self.control.update(
                q, v, reference, input_reference=(1 - ratio) * self.first + ratio * empty,
                normal_targets=-((1 - ratio) * self.open_start + ratio * .001))
        except ValueError as error:
            if not str(error).startswith(('contact friction branches unresolved','contact solve')):
                raise
            # The simplified friction equations can lose their local branch
            # during separation. Solve the same physical held response directly;
            # never issue the unresolved proposal or relax its physical limits.
            inverse = self.guard.control
            inverse.objective, inverse.previous = 'tracking', previous.copy()
            try:
                command = inverse.update(q, v, reference, input_reference=previous,
                                         normal_targets=np.full(len(self.links), -.001))
            finally:
                inverse.objective = 'input'
            self.response_fallbacks += 1
        command = self.guard.update(q, v, command)
        self.elapsed += .0025
        self.info = dict(unload_ratio=ratio, finite_supported_release=getattr(self.control,'diagnostics',{}),
                         held_input_repairs=self.guard.repairs,
                         held_response_fallbacks=self.response_fallbacks)
        gaps = pad_gap_jacobians(self.control.m, self.control.d, self.links, self.control.groups)[0]
        if self.elapsed >= 2. and min(gaps) >= .0002:
            self.release = HeldBackgroundWithdrawal(self.plan, self.plant, q, **self.options)
            self.release.elapsed = .3  # Supported opening has already replaced initial unloading.
        return command

    def close(self):
        self.guard.close()
        if self.withdrawal_guard is not None:self.withdrawal_guard.close()
        if self.release is not None and hasattr(self.release, 'close'):
            self.release.close()
