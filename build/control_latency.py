"""Event-time execution of a sampled controller with measured computation delay.

The plant holds the previously issued input (including servo gains) during the
measured wall duration. Only then may the new command act. Physics is evaluated
in event order, so slow numerical integration does not itself become actuator
latency. This is not a wall-clock-paced hardware-in-the-loop implementation.
"""
import math
import numpy as np


class ControlCycle:
    def __init__(self, model, data, period, *, measured=True):
        self.m, self.d = model, data
        self.observed_at = float(data.time)
        self.period = float(period)
        self.measured = measured
        self.previous = data.ctrl.copy()
        self.bias = model.actuator_biasprm.copy()
        self.gain = model.actuator_gainprm.copy()
        self.q = data.qpos.copy()
        self.v = data.qvel.copy()
        self.elapsed = 0.
        self.application_time = None
        self.failure = None

    def wait_for_result(self, elapsed, advance):
        """advance(steps) integrates/audits old input; returns a failure or None."""
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError('invalid measured control duration')
        self.elapsed = float(elapsed)
        self.next_bias = self.m.actuator_biasprm.copy()
        self.next_gain = self.m.actuator_gainprm.copy()
        self.m.actuator_biasprm[:] = self.bias
        self.m.actuator_gainprm[:] = self.gain
        self.d.ctrl[:] = self.previous
        delay = elapsed if self.measured else 0.
        steps = math.ceil(max(0., delay / self.m.opt.timestep - 1e-10))
        self.failure = advance(steps)
        self.completed_at = float(self.d.time)
        self.position_change = float(np.linalg.norm(self.d.qpos - self.q))
        self.velocity_change = float(np.linalg.norm(self.d.qvel - self.v))
        # A single controller job is outstanding. Missed sampling ticks are
        # skipped, never replayed against old observations in a catch-up burst.
        periods = max(1, math.ceil(max(0., (self.completed_at - self.observed_at) / self.period - 1e-8)))
        self.next_observation = self.observed_at + periods * self.period
        self.skipped_ticks = periods - 1
        return self.failure

    def apply(self, command):
        if self.failure:
            raise ValueError('cannot issue after a physical violation')
        self.m.actuator_biasprm[:] = self.next_bias
        self.m.actuator_gainprm[:] = self.next_gain
        self.d.ctrl[:] = command
        self.application_time = float(self.d.time)

    def remaining_steps(self):
        return max(0, round((self.next_observation - float(self.d.time)) / self.m.opt.timestep))

    def record(self, phase):
        return dict(phase=phase, observation_s=self.observed_at,
                    calculation_s=self.elapsed, completion_s=self.completed_at,
                    application_s=self.application_time, next_observation_s=self.next_observation,
                    skipped_ticks=self.skipped_ticks, wait_failure=self.failure,
                    configuration_change_norm=self.position_change,
                    velocity_change_norm=self.velocity_change)
