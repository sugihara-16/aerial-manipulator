"""Best-effort, sampled geometry for the GUI; never used by the planner.

Each worker replaces its own snapshot. The reader selects the newest sample
for each module count. No renderer, IPC reply, or other worker is awaited.
"""
import json
import os
from pathlib import Path
import time

_last = {}


class GrowthPreview:
    def __init__(self, modules):
        folder = os.environ.get('AMSRR_PLANNING_PREVIEW')
        self.folder = Path(folder) if folder else None
        self.modules = int(modules)
        self.published = False
        self.complete = False
        self.candidate = f'{os.getpid()}-{time.monotonic_ns()}' if folder else None

    def update(self, poses, edges, surfaces, targets):
        if self.folder is None:
            return
        now = time.monotonic()
        key = (str(self.folder), self.modules)
        final = len(poses) == self.modules
        if now - _last.get(key, -1.) < .033 and not (final and self.published and not self.complete):
            return
        _last[key] = now
        try:
            row = dict(modules=self.modules, candidate=self.candidate,
                       timestamp_ns=time.time_ns(), built=len(poses),
                       stage='topology_generated' if len(poses)==self.modules else 'growing',
                       poses=[p.tolist() for p in poses], edges=edges,
                       surfaces=surfaces, targets=targets.tolist())
            path = self.folder / f'{os.getpid()}_{self.modules}.json'
            temp = path.with_suffix('.tmp')
            temp.write_text(json.dumps(row, separators=(',', ':'), allow_nan=False))
            temp.replace(path)
            self.published = True
            self.complete = final
        except (OSError, ValueError):
            # Display failure must not reject a structure or stop execution.
            self.folder = None


def read_previews(folder):
    latest = {}
    for path in Path(folder).glob('*.json'):
        try:
            row = json.loads(path.read_text())
            n = row['modules']
            if row['timestamp_ns'] > latest.get(n, {}).get('timestamp_ns', -1):
                latest[n] = row
        except (OSError, ValueError, KeyError):
            continue
    try:
        checks=json.loads((Path(folder)/'checks.json').read_text())
    except (OSError,ValueError):checks={}
    selected=checks.get('selected')
    for key,check in checks.get('rows',{}).items():
        n=int(key)
        row=latest.setdefault(n,dict(modules=n,built=0,poses=[],edges=[],surfaces=[],targets=[],stage='no_sample',timestamp_ns=0))
        row['checks']={k:v for k,v in check.items() if k!='best'}
        row['search_finished']=bool(checks.get('finished'))
        candidate=selected if selected and selected['modules']==n else check.get('best')
        if candidate:
            row['validated']=_checked_geometry(json.dumps(candidate,sort_keys=True))
            row['selected']=bool(selected and selected['modules']==n)
    for row in latest.values():
        for view in [row]+([row['validated']] if 'validated' in row else []):
            count=view.get('built',len(view.get('poses',[])))
            if count:view['flat']=flat_geometry(count,json.dumps(view['edges'],sort_keys=True))
    return [latest[n] for n in sorted(latest)]


class SearchPreview:
    """Parent-owned checks and best checked candidates, separate from growth samples."""
    def __init__(self):
        folder=os.environ.get('AMSRR_PLANNING_PREVIEW')
        self.folder=Path(folder) if folder else None
        self.rows={};self.updated=0.;self.finished=False

    def outcome(self,n,status,cost=None,candidate=None):
        if self.folder is None:return
        row=self.rows.setdefault(int(n),dict(checked=0,accepted=0,rejected=0,bounded=0,cancelled=0))
        category={'accepted':'accepted','bounded_nonimprovement':'bounded','cancelled':'cancelled'}.get(status,'rejected')
        row[category]+=1
        if category!='cancelled':row['checked']+=1
        if category=='accepted' and candidate is not None and (row.get('best_cost') is None or cost<row['best_cost']):
            try:row['best']=candidate();row['best_cost']=float(cost)
            except (ValueError,TypeError,KeyError,AttributeError):
                self.folder=None;return
            self.flush(force=True)
        else:self.flush()

    def finish(self,selected=None):
        if self.folder is None:return
        self.finished=True
        self.flush(force=True,selected=selected)

    def flush(self,force=False,selected=None):
        if self.folder is None or (not force and time.monotonic()-self.updated<.05):return
        self.updated=time.monotonic()
        try:
            path=self.folder/'checks.json';temp=path.with_suffix('.tmp')
            temp.write_text(json.dumps(dict(rows=self.rows,finished=self.finished,selected=selected),
                separators=(',',':'),allow_nan=False,default=lambda x:x.tolist()))
            temp.replace(path)
        except (OSError,ValueError,TypeError):self.folder=None


def compact_candidate(n,edges,q,surfaces):
    # Pinocchio coordinates, not MuJoCo's differently ordered joint vector.
    return dict(modules=n,edges=edges,q=q,surfaces=surfaces)


def grasp_candidate(plan):
    return compact_candidate(plan.plant.meta['modules'],plan.plant.meta['edges'],plan.closed_q,plan.surfaces)


def valve_candidate(plan):
    return compact_candidate(plan['modules'],plan['edges'],plan['closed'],plan['surfaces'])


def walking_candidate(plan,plant):
    import numpy as np
    q=np.asarray(plan['q']).copy();pq,mq,*_=plant.coordinate_maps
    q[pq]=np.asarray(plan['gait']['configurations'][0])[mq]
    return compact_candidate(plan['modules'],plan['edges'],q,plan['surfaces'])


def select_candidate(candidate):
    """Publish the exact final selection after optional posture refinement."""
    folder=os.environ.get('AMSRR_PLANNING_PREVIEW')
    if not folder:return
    try:
        path=Path(folder)/'checks.json'
        row=json.loads(path.read_text());row['selected']=candidate();row['finished']=True
        temp=path.with_suffix('.tmp');temp.write_text(json.dumps(row,default=lambda x:x.tolist(),allow_nan=False));temp.replace(path)
    except (OSError,ValueError,TypeError,KeyError,AttributeError):pass


from functools import lru_cache


@lru_cache(maxsize=64)
def _checked_geometry(raw):
    # Runs in the GUI server, never inside structure search or control.
    import numpy as np
    import pinocchio as pin
    from .model import build_model
    row=json.loads(raw);model,meta=build_model(row['modules'],row['edges'])
    data=model.createData();pin.framesForwardKinematics(model,data,np.array(row['q'][:model.nq]))
    return dict(modules=row['modules'],built=row['modules'],edges=row['edges'],
        poses=[data.oMf[meta['frames'][f'm{i}/root']].homogeneous.tolist() for i in range(row['modules'])],
        surfaces=row['surfaces'],targets=[],stage='validated')


@lru_cache(maxsize=256)
def flat_geometry(modules, edges_json):
    """Display-only top view: neutral joints and world-aligned root."""
    import pinocchio as pin
    from .model import build_model,PORTS
    edges=json.loads(edges_json);model,meta=build_model(modules,edges)
    data=model.createData();pin.framesForwardKinematics(model,data,pin.neutral(model))
    return dict(poses=[data.oMf[meta['frames'][f'm{i}/root']].homogeneous.tolist() for i in range(modules)],
        ports=[{port:data.oMf[meta['frames'][f'm{i}/{link}']].translation.tolist() for port,link in PORTS.items()} for i in range(modules)],
        rotors=[[data.oMf[meta['frames'][f'm{i}/thrust_{j}']].translation.tolist() for j in range(1,5)] for i in range(modules)])
