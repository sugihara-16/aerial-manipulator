"""Offline static diagnostics at saved approach states, not plan selection."""
import json
import sys
from pathlib import Path
import numpy as np
import mujoco
import pinocchio as pin
from amsrr_det.valve import ValveSpec
from amsrr_det.valve_search import unpack
from amsrr_det.valve_trim import static_balance
from scripts.run_valve_task import ValveReferences

root = Path(sys.argv[1])
rows = []
for name in ['high_down_t48_a0', 'high_down_t48_a1', 'high_down_t48_a2']:
    folder = root / name
    selection = json.loads((folder / 'selection.json').read_text())
    spec = ValveSpec(**selection['request'])
    plan = unpack(spec, selection['plan'])
    system = plan.system
    plant = system.control
    refs = ValveReferences(system, plant.configuration(plan.closed),
                           plant.configuration(plan.opened))
    states = np.load(folder / 'physical_states.npz')
    trace = json.loads((folder / 'physical_trace.json').read_text())
    data = mujoco.MjData(system.m)
    pq, mq, *_ = plant.coordinate_maps

    def screen(q):
        pose = pin.neutral(plant.pin)
        pose[pq] = q[mq]
        answer = static_balance(plant.pin, plant.meta, pose,
                                payload_fraction=0., direct_burden=True)
        return {'status': answer['status'],
                'thrust': answer.get('thrust'),
                'max_effort_ratio': max((abs(j['effort_nm']) / a['effort']
                    for j, a in zip(answer.get('joints', []), plant.meta['actuators'])),
                    default=None)}

    samples = []
    for time in [0., 1., 2., 3., 3.5, 4.]:
        index = min(range(len(trace)), key=lambda i: abs(trace[i]['time_s'] - time))
        data.qpos[:] = states['q'][index]
        data.qvel[:] = states['v'][index]
        actual, velocity = system.observation(data)
        args, kwargs = refs.update(time, actual, velocity, float(data.qpos[system.oi]))
        planned = args[2]
        synced = refs.opened.copy()
        p = min(time / 4., 1.)
        fraction = p**3 * (10 - 15*p + 6*p*p)
        mujoco.mj_integratePos(plant.mj, synced, refs.delta, fraction)
        samples.append({'time_s': time, 'actual': screen(actual),
                        'opened_direction_reference': screen(planned),
                        'synchronized_direction_reference': screen(synced)})
    rows.append({'id': name, 'samples': samples})
    print(name, [(s['time_s'], *(s[k]['status'] for k in ['actual',
          'opened_direction_reference', 'synchronized_direction_reference']))
          for s in samples], flush=True)

(root / 'acquisition_support_review.json').write_text(json.dumps({
    'rows': rows,
    'scope': 'Static self-weight support at fixed poses/directions with no object contact. '
             'Not dynamic feasibility, not proof of the initial cause of an execution failure.'
}, indent=2))
