"""Observation-only held-input response and locally linearized sensitivities.

Contact geometry and force-law branches are recomputed at each substep.
Sensitivities omit derivatives of contact location/normal/regularization;
they are proposals, and finite input changes must be evaluated again.
"""
import mujoco
import numpy as np

from build.contact_tangent import local_input_tangent, frozen_response
from build.native_contact_rollout import ContactRollout


class ShortContactResponse:
    def __init__(self, plant, links, steps=40, period=.01):
        self.plant = plant
        self.m = plant.mj
        self.steps = steps
        self.timestep = period / steps
        self.m.opt.timestep = self.timestep
        self.d = mujoco.MjData(self.m)
        self.runner = ContactRollout(self.m, links)

    def evaluate(self, q, v, command, linearize=False):
        m, d = self.m, self.d
        m.opt.timestep = self.timestep
        mujoco.mj_resetData(m, d)
        d.qpos[:] = q
        d.qvel[:] = v
        d.ctrl[:] = command
        if not linearize:
            states, loads, motors, penetration = self.runner.advance(d, self.steps, include_penetration=True)
            return dict(q=d.qpos.copy(), v=d.qvel.copy(), states=states,
                        loads=loads, motors=motors, penetration=penetration)
        derivative = np.zeros((2*m.nv, m.nu))
        states, loads, motors, penetration, derivatives, load_derivatives = [], [], [], [], [], []
        max_balance = 0.
        for _ in range(self.steps):
            qpin, vpin = self.plant.observation(d)
            state, load, motor, pen = self.runner.advance(d, 1, include_penetration=True)
            tangent = local_input_tangent(m, d)
            response = frozen_response(self.plant, qpin, vpin, d, tangent)
            force_inputs = response['Lx'] @ derivative + response['Lu']
            aggregated = np.zeros((self.runner.pads, 7, m.nu))
            for contact in d.contact:
                ga, gb = map(int, contact.geom)
                if self.runner.object not in (ga, gb) or contact.efc_address < 0:
                    continue
                pad = self.runner.groups[gb if ga == self.runner.object else ga]
                if pad < 0:
                    continue
                row = contact.efc_address
                world = contact.frame.reshape(3, 3).T @ force_inputs[row:row+3]
                world *= 1. if ga == self.runner.object else -1.
                offset = contact.pos - d.xpos[self.runner.bodies[pad]]
                aggregated[pad, 0] += force_inputs[row]
                aggregated[pad, 1:4] += world
                aggregated[pad, 4:7] += np.cross(offset, world.T).T
            load_derivatives.append(aggregated)
            derivative = response['A'] @ derivative + response['B']
            derivatives.append(derivative.copy())
            states.append(state[0]); loads.append(load[0]); motors.append(motor[0]); penetration.append(pen[0])
            max_balance = max(max_balance, tangent['dynamic_balance_max'])
        return dict(q=d.qpos.copy(), v=d.qvel.copy(), states=np.array(states),
                    loads=np.array(loads), motors=np.array(motors), penetration=np.array(penetration),
                    derivatives=np.array(derivatives), load_derivatives=np.array(load_derivatives),
                    dynamic_balance_max=max_balance)
