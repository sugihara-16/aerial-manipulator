"""Reproduce the approved span/load panel; never choose inputs from outcomes."""
import argparse,copy,json
from pathlib import Path
import numpy as np
from amsrr_det.objects import ObjectShape


def generate(source):
    rng=np.random.default_rng(2026102404);panel=copy.deepcopy(source);cases=[]
    for i,mass in enumerate([3.,3.5,4.,4.5,5.]):
        for family in source['configuration']['families']:
            choices=[c for c in source['cases'] if c['strata']['family']==family and c['strata']['yaw_bin']==i%4]
            original=choices[int(rng.integers(len(choices)))];case=copy.deepcopy(original)
            target=float(rng.uniform(1.,1.3));size=np.asarray(case['payload']['size'])
            shape=ObjectShape(family,size*target/max(size),mass)
            case['payload']=shape.payload();case['id']=f'span_load_{family}_{round(mass*10):03d}'
            for key in ['source','destination']:
                pose=np.array(case[key]);pose[2]=0.;pose[2]=case['support_top']-shape.lowest_z(pose);case[key]=pose.tolist()
            case['strata'].update(mass_bin=f'{mass:g}kg',size_bin='over_115cm' if target>1.15 else 'up_to_115cm')
            case['span_load_extension']=dict(original_case=original['id'],target_max_dimension_m=target,mass_kg=mass)
            cases.append(case)
    panel.update(role='span_load_extension',seed=2026102404,cases=cases,
        scope='30 cases fixed with user approval; 3-5 kg, maximum dimensions 1.0-1.3 m; unchanged physical limits and success criteria; search 1-8 modules',
        rationale='Larger opposing-surface separation requires connection reach; moderate additional payload motivates distributing limited pad forces. Neither mechanism forces a module count.')
    panel['configuration'].update(holdout_cases=30,object_mass_kg=[3.,5.],object_dimensions_m=[min(min(c['payload']['size'])for c in cases),1.3])
    return panel


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,default=Path('artifacts/runs/task_pose120_v7_20261023/panel.json'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    with args.output.open('x') as stream:
        json.dump(generate(json.loads(args.source.read_text())),stream,indent=2)
