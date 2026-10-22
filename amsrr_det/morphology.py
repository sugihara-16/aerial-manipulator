"""Generate connected trees and contact assignments from port rules.

Only part geometry is reusable. No finished structure or task-indexed answer is
stored. A beam limits search, so failure means 'not found', never impossibility.
"""
from itertools import combinations
import numpy as np
import pinocchio as pin
from .model import build_model,PORTS
from .grasp_geometry import free_surfaces,grasp_frame,fit_contacts


def grow_tree(n,edges):
    occupied={(e['a'],e['port_a']) for e in edges}|{(e['b'],e['port_b']) for e in edges}
    for module in range(n):
        for port in PORTS:
            if (module,port) in occupied:continue
            for incoming in PORTS:
                if port[:3]!=incoming[:3]:
                    yield edges+[dict(a=module,port_a=port,b=n,port_b=incoming)]


def align_seed(model,meta,q,surfaces,points,normals, *, upright=False):
    data=model.createData();pin.framesForwardKinematics(model,data,q)
    frames=[data.oMf[meta['frames'][grasp_frame(*s)]] for s in surfaces]
    local=np.vstack([[T.translation,T.translation+.15*T.rotation[:,0]] for T in frames])
    goal=np.vstack([[p,p+.15*n] for p,n in zip(points,normals)])
    u,singular,vh=np.linalg.svd((local-local.mean(axis=0)).T@(goal-goal.mean(axis=0)))
    R=vh.T@np.diag([1.,1.,np.linalg.det(vh.T@u.T)])@u.T
    if upright and singular[0]>1e-14 and singular[1]<1e-12*singular[0]:
        # Collinear targets leave rotation around their line undetermined.
        # Resolve that exact Kabsch freedom by maximizing root up alignment;
        # the least-squares contact fit is unchanged. This does not impose a
        # level posture in the subsequent optimization.
        axis=vh[0];up=R[:,2]
        a=up[2]-axis[2]*(axis@up);b=np.cross(axis,up)[2]
        R=pin.exp3(axis*np.arctan2(b,a))@R
    translation=goal.mean(axis=0)-R@local.mean(axis=0)
    seed=q.copy();seed[:3]=translation;seed[3:7]=pin.Quaternion(R).coeffs()
    error=np.linalg.norm(local@R.T+translation-goal)
    # Prefer roots near level only as an ordering heuristic; tilted candidates
    # remain eligible and all accepted ones face the same physical checks.
    return seed,float(error+.02*(1-R[2,2]))


def posture_candidates(shape,source, *, max_modules=3,beam=12,seed=194):
    """Yield fitted candidates and diagnostics, then grow promising topologies."""
    source=np.asarray(source);R=pin.Quaternion(source[3:]).matrix()
    tasks=[]
    for angle in [0.,np.pi/2,np.pi/4,-np.pi/4]:
        axis=np.array([np.cos(angle),np.sin(angle),0.]);points,normals=shape.opposed_sites(source,axis)
        tasks.append((points,normals))
    graphs=[[]];rng=np.random.default_rng(seed)
    for n in range(1,max_modules+1):
        pool=[]
        for edges in graphs:
            m,meta=build_model(n,edges,payload=shape.payload());q=pin.neutral(m);q[meta['object']['q']:]=source
            surfaces=free_surfaces(meta)
            # Neutral and one small random fold are initial guesses only.
            for attempt in range(2):
                start=q.copy()
                if attempt:
                    for a in meta['actuators']:
                        if a['role']=='dock':start[a['q']]=rng.uniform(-1.2,1.2)
                for assigned in combinations(surfaces,2):
                    for points,normals in tasks:
                        world=points@R.T+source[:3];wn=normals@R.T
                        aligned,score=align_seed(m,meta,start,assigned,world+.001*wn,wn)
                        pool.append((score,edges,m,meta,aligned,list(assigned),points,normals))
        pool.sort(key=lambda row:row[0]);parents=[];seen=set()
        # Avoid spending the entire beam on different guesses for one topology.
        counts={}
        for score,edges,m,meta,q,surfaces,points,normals in pool:
            key=tuple((e['a'],e['port_a'],e['b'],e['port_b']) for e in edges)
            if counts.get(key,0)>=max(4,beam//len(graphs)):continue
            counts[key]=counts.get(key,0)+1
            if key not in seen:parents.append(edges);seen.add(key)
            closed,fit=fit_contacts(m,meta,q,surfaces,points@R.T+source[:3]+.001*(normals@R.T),normals@R.T,
                floor_clearance=.03,base_rotation=True,max_evaluations=100)
            yield dict(modules=n,edges=edges,surfaces=surfaces,points=points,normals=normals,q=closed,fit=fit,initial_score=score)
            if sum(counts.values())>=beam:break
        graphs=[child for edges in parents[:3] for child in grow_tree(n,edges)]
