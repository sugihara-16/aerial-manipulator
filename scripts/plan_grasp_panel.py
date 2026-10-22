"""Fresh generation on task inputs; no saved graph or posture is selected."""
import argparse,json,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from amsrr_det.objects import ObjectShape
from amsrr_det.structure_search import plan_grasp
from amsrr_det.evidence import save_plan,save_sources


def execute(item):
    case,root=item;out=Path(root)/case['id'];out.mkdir(parents=True,exist_ok=True);save_sources(out)
    payload=case.get('payload',case.get('shape'))
    shape=ObjectShape(payload.get('shape',payload.get('family')),payload['size'],payload['mass'])
    request=dict(payload=shape.payload(),source=case['source'],destination=case['destination'],support_top=case['support_top'])
    (out/'request.json').write_text(json.dumps(request,indent=2));start=time.perf_counter()
    try:
        plan=plan_grasp(shape,np.asarray(case['source']),np.asarray(case['destination']),case['support_top'])
        save_plan(plan,out,'fresh_input_geometric_static_candidate_only')
        result=dict(candidate_found=True,modules=plan.plant.meta['modules'],surfaces=plan.surfaces,reported_planning_s=plan.wall_s)
    except ValueError as exc:result=dict(candidate_found=False,reason=str(exc),evidence=getattr(exc,'evidence',None))
    result.update(case=case['id'],task_success=False,scope='planning_only_no_physical_execution',wall_s=time.perf_counter()-start)
    (out/'planning_result.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='evidence'}),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--workers',type=int,default=2);args=parser.parse_args()
    cases=json.loads(args.input.read_text());cases=cases['cases'] if isinstance(cases,dict) else cases
    args.output.mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:results=list(pool.map(execute,[(case,args.output) for case in cases]))
    (args.output/'planning_summary.json').write_text(json.dumps(results,indent=2))
