"""Freeze thirty size-extension tasks without inspecting previous outcomes."""
import copy
import json
from pathlib import Path
import numpy as np
from amsrr_det.objects import ObjectShape


def generate(source):
    rng = np.random.default_rng(20261024)
    panel = copy.deepcopy(source)
    families = panel['configuration']['families']
    cases = []
    for index, target in enumerate([.65, .75, .85, 1., 1.2]):
        for family in families:
            # Select across yaw bins, using input data only, never result files.
            candidates = [c for c in source['cases'] if c['strata']['family'] == family
                          and c['strata']['yaw_bin'] == index % 4]
            old = candidates[int(rng.integers(len(candidates)))]
            case = copy.deepcopy(old)
            size = np.array(old['payload']['size'])
            scale = target / max(size)
            shape = ObjectShape(family, size * scale, old['payload']['mass'])
            case['payload'] = shape.payload()
            case['id'] = f'large_{family}_{round(target*100):03d}cm'
            for key in ['source', 'destination']:
                pose = np.array(old[key]);pose[2] = 0.
                pose[2] = old['support_top'] - shape.lowest_z(pose)
                case[key] = pose.tolist()
            case['size_extension'] = dict(original_case=old['id'],linear_scale=float(scale),
                target_max_dimension_m=target,original_size_m=size.tolist(),mass_unchanged=True,
                orientation_preserved_by_uniform_scaling=True)
            case['strata']['size_bin'] = f'{round(target*100):03d}cm'
            case['resting_pose_generation'] = dict(method='uniform scaling of stable input pose',
                                                   source=old.get('resting_pose_generation'))
            cases.append(case)
    panel.update(cases=cases,role='size_extension',seed=20261024,
        scope='30 new enlarged inputs; unchanged masses and task success criteria; no previous outcomes used')
    panel['configuration']['holdout_cases'] = 30
    panel['configuration']['object_dimensions_m'] = [min(min(c['payload']['size']) for c in cases),1.2]
    panel['configuration']['size_extension_max_dimensions_m'] = [.65,.75,.85,1.,1.2]
    return panel


if __name__ == '__main__':
    source=Path('artifacts/runs/task_pose120_v7_20261023/panel.json')
    output=Path('artifacts/runs/large30_inputs_20261024.json')
    panel=generate(json.loads(source.read_text()))
    with output.open('x') as stream:json.dump(panel,stream,indent=2)
    print(output)
