"""Use the contact-free drive predictor only where its geometry is valid.

The geometry forecast uses known surfaces and observed object motion. It is a
sampled nominal-model check, not a bound on unknown motion or real-time delays.
"""
import numpy as np
import pinocchio as pin
from .coupled_flight import CoupledFlightController
from .clearance import MotionBoundedClearance, PART_CLEARANCE


class FreeFlightSwitch:
    proximity_m = .002

    def __init__(self, plan, contact_controller):
        self.plant = plan.plant
        self.contact = contact_controller
        self.free = CoupledFlightController(self.plant.pin, self.plant.meta,
            plan.initial_q, track_vectoring=True, planned_input_reference=True)
        self.active = self.free
        self.acquired = False
        self.released = False
        self.transitions = []
        # This private geometry query receives predicted configurations only.
        # No physical contact force or hidden simulator state enters control.
        self.clearance = MotionBoundedClearance(self.plant.mj, [])
        m = self.plant.mj
        onset = max(float(m.geom_margin.max(initial=0.)),
                    float(m.pair_margin.max(initial=0.)))
        if onset >= PART_CLEARANCE - 1e-6:
            raise ValueError('free_flight_contact_range_not_covered')
        self.last_forecast = None

    def select(self, phase, grasp, q, time_s):
        far = False
        if phase in ['ACQUIRE', 'RELEASE'] and grasp is not None:
            far = bool(np.all(grasp.geometry(q)[-1] < -self.proximity_m))
        if phase == 'ACQUIRE' and grasp is not None and not far:
            self.acquired = True
        if phase == 'RELEASE' and far:
            self.released = True
        use_free = (phase == 'APPROACH'
                    or (phase == 'ACQUIRE' and not self.acquired)
                    or (phase == 'RELEASE' and self.released))
        chosen = self.free if use_free else self.contact
        if chosen is not self.active:
            previous = self.active
            for name in ['command', 'force', 'last_joint_bias', 'last_target_velocity']:
                setattr(chosen, name, getattr(previous, name).copy())
            chosen.control_count = previous.control_count
            self.transitions.append(dict(time_s=time_s, phase=phase,
                controller='free' if use_free else 'contact'))
            self.active = chosen
        return chosen

    def check(self, q, v):
        if self.active is not self.free:
            return True
        samples = self.free.prediction_samples
        if not samples:
            self.last_forecast = dict(valid=False, reason='missing_free_flight_prediction')
            return False
        model, meta = self.plant.pin, self.plant.meta
        obj = meta['object']
        object_velocity = np.zeros(model.nv)
        if obj is not None:
            object_velocity[obj['v']:obj['v']+6] = v[obj['v']:obj['v']+6]
        lowest_part = lowest_environment = np.inf
        for index, sample in enumerate(samples):
            predicted = sample['q'].copy()
            elapsed = self.free.dt * (index + 1) / len(samples)
            if obj is not None:
                extrapolated = pin.integrate(model, q, elapsed*object_velocity)
                predicted[obj['q']:obj['q']+7] = extrapolated[obj['q']:obj['q']+7]
            bad, part, environment = self.clearance.check(
                self.plant.configuration(predicted), grasp_contact_allowed=False)
            lowest_part = min(lowest_part, part)
            lowest_environment = min(lowest_environment, environment)
            if bad:
                self.last_forecast = dict(valid=False, prediction_time_s=elapsed,
                    pairs=bad, minimum_part_gap_m=part,
                    minimum_environment_gap_m=environment)
                return False
        self.last_forecast = dict(valid=True, samples=len(samples),
            minimum_part_gap_m=lowest_part, minimum_environment_gap_m=lowest_environment,
            object_motion='observed_constant_body_velocity', scope='sampled_nominal_prediction')
        return True
