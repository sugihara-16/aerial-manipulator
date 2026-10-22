"""Small varied development panel. Never a held-out acceptance result."""
import argparse,json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np,pinocchio as pin
from amsrr_det.objects import ObjectShape
from amsrr_det.evaluation_inputs import resting_pose
from scripts.run_grasp_task import run


def execute_case(item):
    request,output=item
    shape=ObjectShape(**request['shape']);source=np.asarray(request['source']);destination=np.asarray(request['destination'])
    result=run(shape,source,destination,request['support_top'],output)
    brief={k:result[k] for k in ['task_success','reason','planning_time_s','wall_s','position_error_m'] if k in result}
    brief['case']=request['id'];print(json.dumps(brief),flush=True);return brief


def cases():
    families=['box','elongated_box','cylinder','ellipsoid','triangular_prism','tapered_box']
    for family in families:
        old=json.loads(Path('artifacts/runs/development_panel_v1',family,'request.json').read_text())
        for alternate in [False,True]:
            size=np.array(old['payload']['size'])*(0.90 if alternate else 1.)
            shape=ObjectShape(family,size,old['payload']['mass']*(1.1 if alternate else 1.))
            source=np.asarray(old['source']);top=old['support_top']
            if alternate:
                yaw=np.deg2rad(70+families.index(family)*53)
                roll=0 if family=='cylinder' else np.pi/2
                source[3:]=pin.Quaternion(pin.rpy.rpyToMatrix(roll,0,yaw)).coeffs()
                source[:2]*=-1;top+=.03
            rested,rest_provenance=resting_pose(shape,source)
            source[3:]=rested[3:];source[2]=top+rested[2]
            destination=source.copy();destination[0]+=.35 if alternate else old['destination'][0]-old['source'][0]
            yield dict(id=family+('_alternate' if alternate else '_base'),shape=dict(family=family,size=size.tolist(),mass=shape.mass),
                       source=source.tolist(),destination=destination.tolist(),support_top=top,
                       resting_pose_generation=rest_provenance)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--workers',type=int,default=2);args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True);requests=list(cases())
    (args.output/'cases.json').write_text(json.dumps(requests,indent=2))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results=list(pool.map(execute_case,[(r,args.output/r['id']) for r in requests]))
    (args.output/'summary.json').write_text(json.dumps(results,indent=2))
