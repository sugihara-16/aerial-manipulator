"""Recheck saved physical poses after correcting detection bounds.

Samples can disprove clearance, but cannot certify unsaved trajectory intervals.
"""
from pathlib import Path
import argparse, json
import numpy as np
import mujoco
from build.load_saved_plan import load_saved_plan
from amsrr_det.clearance import proximity_model, ContactClassifier


def check(root):
    plan,_=load_saved_plan(root);p=plan.plant
    m=proximity_model(p.mj);data=mujoco.MjData(m)
    names=[m.geom(i).name for i in range(m.ngeom)]
    links=[f'm{module}/'+('yaw_dock_mech' if port.startswith('yaw') else 'pitch_dock_mech')+port[-1]
           for module,port in plan.surfaces]
    classify=ContactClassifier(m,names,links)
    records=json.loads((root/'trace.json').read_text())
    checked=0;first=None;worst=None
    for row in records:
        q=np.asarray(row['configuration']);data.qpos[:]=p.configuration(q)
        mujoco.mj_kinematics(m,data);mujoco.mj_collision(m,data)
        intended,limits=classify(data)
        if row['phase']=='APPROACH':
            a,b=np.asarray(data.contact.geom1),np.asarray(data.contact.geom2)
            intended=((a==classify.object)&classify.supports[b])|((b==classify.object)&classify.supports[a])
        dist=np.asarray(data.contact.dist);bad=np.flatnonzero(~intended&(dist<limits-1e-6))
        checked+=1
        if len(bad):
            i=bad[np.argmin(dist[bad]-limits[bad])];c=data.contact[i]
            violation=dict(t=row['t'],phase=row['phase'],parts=[names[c.geom1],names[c.geom2]],distance_m=float(c.dist),required_m=float(limits[i]))
            if first is None:first=violation
            if worst is None or violation['distance_m']-violation['required_m']<worst['distance_m']-worst['required_m']:worst=violation
    return dict(source=str(root),scope='saved_pose_samples_only_no_interval_or_force_certification',samples=checked,
                first_violation=first,worst_violation=worst)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--results',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    reports=[]
    for parent in args.results:
        for path in sorted(parent.glob('*/result.json')):
            if not (path.parent/'trace.json').exists():continue
            result=check(path.parent);reports.append(result)
            args.output.write_text(json.dumps(reports,indent=2))
            print(json.dumps(result),flush=True)
