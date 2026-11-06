#!/usr/bin/env python3
"""Select every available generated structure, irrespective of task outcome."""
import hashlib
import json
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
SOURCES = dict(
    grasp='artifacts/runs/control_tail150_v8_20261027',
    valve='artifacts/valve_side_faces_20261101/full36_v15',
    walking='artifacts/walking_landing_20261105/final36_v13')


def panel():
    cases=[];excluded=[]
    for task,folder in SOURCES.items():
        for case in sorted((ROOT/folder).iterdir()):
            if not case.is_dir() or not (case/'selection.json').exists(): continue
            source=case/('generated_plan/plan.json' if task=='grasp' else 'selection.json')
            value=json.loads(source.read_text()) if source.exists() else None
            plan=value if task=='grasp' else value.get('plan')
            if not plan:
                excluded.append(dict(task=task,id=case.name,reason='no_generated_structure'))
                continue
            model=plan['model'] if task=='grasp' else plan
            identity=task+'/'+case.name
            cases.append(dict(id=identity,task=task,modules=model['modules'],edges=model['edges'],
                source=str(source.relative_to(ROOT)),sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                initial_seed=int.from_bytes(hashlib.sha256(('assembly-v1/'+identity).encode()).digest()[:4],'little')))
    return dict(scope='Assembly of all final task-generated structures, including downstream task failures. One module is a takeoff/hold control, reported separately.',
                sources=SOURCES,cases=cases,excluded=excluded,
                counts=dict(Counter(x['task'] for x in cases)),
                module_counts=dict(Counter(x['modules'] for x in cases)))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path);args=parser.parse_args()
    result=panel();args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ('cases','excluded')},indent=2))
