"""Assembly procedures for an immutable task-generated connection tree.

Finite path samples are collision screens, not continuous safety proofs.
The subset dynamic program compares all component-disjoint merge rounds in
its specified planar/open-edge path family; it does not optimize topology.
"""
import time
from functools import lru_cache
import numpy as np
import pinocchio as pin
import mujoco
from scipy.optimize import least_squares,linear_sum_assignment
from .assembly import make_assembly_plant,connected_view,forest_posture,DockFilters
from .model import PORTS
from .grasp_geometry import part_bounds
from .assembly_air import AirParameters,interference


def groups(n,edges,mask):
    parents=list(range(n))
    def root(i):
        while parents[i]!=i:i=parents[i]
        return i
    for i,e in enumerate(edges):
        if mask&(1<<i):parents[root(e['b'])]=root(e['a'])
    return [root(i) for i in range(n)]


def initial_xy(n,seed):
    rng=np.random.default_rng(seed)
    if n==1:return rng.uniform(-1.,1.,(1,2)),rng.uniform(-np.pi,np.pi,1)
    for _ in range(10000):
        xy=rng.uniform(-1.8,1.8,(n,2));dist=np.linalg.norm(xy[:,None]-xy[None,:],axis=2)
        np.fill_diagonal(dist,np.inf);nearest=dist.min(axis=1)
        if np.all((nearest>=1.)&(nearest<=2.)):
            return xy,rng.uniform(-np.pi,np.pi,n)
    raise ValueError('initial spacing sampling exhausted')


def eased(a,b,t,duration):
    s=float(np.clip(t/duration,0.,1.));f=10*s**3-15*s**4+6*s**5
    v=(30*s**2-60*s**3+30*s**4)/duration
    acc=(60*s-180*s**2+120*s**3)/duration**2
    return a+(b-a)*f,(b-a)*v,(b-a)*acc


def prepare(case,timestep=.0005):
    began=time.perf_counter();n=case['modules'];edges=case['edges']
    p=make_assembly_plant(n,edges,timestep=timestep);m=p.mj;view=connected_view(p,edges)
    xy,yaw=initial_xy(n,case['initial_seed']);initial=pin.neutral(p.pin)
    pq=[b['q'] for b in p.meta['bases']]
    for i,k in enumerate(pq):
        initial[k:k+2]=xy[i];initial[k+3:k+7]=pin.Quaternion(pin.rpy.rpyToMatrix(0,0,float(yaw[i]))).coeffs()
    pd=p.pin.createData();pin.framesForwardKinematics(p.pin,pd,initial)
    for i,k in enumerate(pq):
        bottom=min(float(pd.oMf[p.meta['frames'][f'm{i}/{link}']].act(center)[2]-abs(pd.oMf[p.meta['frames'][f'm{i}/{link}']].rotation[2])@half) for link,center,half in part_bounds())
        initial[k+2]=-bottom+.00005
    initial=p.configuration(initial)
    qi=np.array([int(m.jnt_qposadr[m.joint(p.pin.names[b['joint']]).id]) for b in p.meta['bases']])
    vi=np.array([int(m.jnt_dofadr[m.joint(p.pin.names[b['joint']]).id]) for b in p.meta['bases']])
    filters=DockFilters(p,edges);filters.apply(m,range(len(edges)))
    check=mujoco.MjData(m)
    sites=np.array([m.site(p.pin.frames[r['frame']].name).id for r in p.meta['rotors']])
    modules=np.array([r['module'] for r in p.meta['rotors']]);nr=len(sites)
    nominal=np.full(nr,p.meta['mass']*9.80665/nr);phases=np.zeros((nr,2));air=AirParameters()
    maximum_predicted_loss=0.
    def air_clear(data):
        nonlocal maximum_predicted_loss
        loss,_=interference(data.site_xpos[sites],data.site_xmat[sites].reshape(-1,3,3)[:,:,2],
                            nominal,modules,phases,0.,air)
        peak=float(loss.max(initial=0.))*(1+air.fluctuation)
        maximum_predicted_loss=max(maximum_predicted_loss,peak)
        # A conservative planning allowance for an uncalibrated wake, never
        # an execution stop or a change to the actual aerodynamic force.
        return peak<=.1
    def clear(q):
        check.qpos[:]=q;mujoco.mj_kinematics(m,check);mujoco.mj_collision(m,check)
        return not any(c.dist<-.00005 for c in check.contact) and air_clear(check)
    # Planar posture variables are the actual yaw hinges, not a finished
    # structure catalogue. Each independent module stays upright.
    cm=view.pin;cd=cm.createData();q=pin.neutral(cm);q[2]=1.5
    yaw_indices=[a['q'] for a in view.meta['actuators'] if a['role']=='dock' and '/yaw' in a['name']]
    root_ids=[view.meta['frames'][f'm{i}/root'] for i in range(n)]
    pairs=np.array([(i,j) for i in range(n) for j in range(i)],dtype=int)
    adjacent={tuple(sorted((e['a'],e['b']))) for e in edges}
    separation=np.array([.72 if tuple(sorted(pair)) in adjacent else .94 for pair in pairs])
    def positions(x):
        q[yaw_indices]=x;pin.framesForwardKinematics(cm,cd,q)
        return np.array([cd.oMf[f].translation.copy() for f in root_ids])
    def residual(x):
        xyz=positions(x)
        sep=np.linalg.norm(xyz[pairs[:,0]]-xyz[pairs[:,1]],axis=1) if len(pairs) else np.empty(0)
        return np.r_[10*np.maximum(separation-sep,0.),.02*x]
    candidates=[];rng=np.random.default_rng(case['initial_seed']+1)
    for attempt in range(12):
        x0=np.zeros(len(yaw_indices)) if attempt==0 else rng.uniform(-1.6,1.6,len(yaw_indices))
        fitted=least_squares(residual,x0,bounds=(-1.8,1.8),max_nfev=45)
        xyz=positions(fitted.x)
        # Rigid placement relative to measured initial centres.
        a=xyz[:,:2]-xyz[:,:2].mean(axis=0);b=xy-xy.mean(axis=0)
        angle=yaw[0] if n==1 else np.arctan2(np.sum(a[:,0]*b[:,1]-a[:,1]*b[:,0]),np.sum(a*b))
        q[:2]=xy.mean(axis=0);q[3:7]=pin.Quaternion(pin.rpy.rpyToMatrix(0,0,float(angle))).coeffs()
        pin.framesForwardKinematics(cm,cd,q)
        center=np.array([cd.oMf[f].translation for f in root_ids]).mean(axis=0)
        q[:2]+=xy.mean(axis=0)-center[:2]
        closed=p.configuration(forest_posture(p,view,q))
        if clear(closed):candidates.append(closed.copy());break
        q[:2]=0.;q[3:7]=[0.,0.,0.,1.]
    if not candidates:raise ValueError('no collision-free planar assembly posture')
    closed=candidates[0]
    from .equilibrium import static_balance
    closed_pin,_=p.observation(_data(m,closed))
    support=static_balance(p.pin,p.meta,closed_pin,payload_fraction=0.,balance_thrust=True)
    if support['status']!='static_screen_pass':raise ValueError('independent module flight support not found')
    pin_q,_=p.observation(_data(m,closed));pin.framesForwardKinematics(p.pin,pd,pin_q)
    descendants=[];directions=[]
    for edge in edges:
        nodes={edge['b']}
        while True:
            expanded=nodes|{e['b']for e in edges if e['a'] in nodes}
            if expanded==nodes:break
            nodes=expanded
        descendants.append(sorted(nodes))
        directions.append(pd.oMf[p.meta['frames'][f"m{edge['a']}/{PORTS[edge['port_a']]}"]].rotation[:,0].copy())
    gap=.18
    def pose(mask):
        result=closed.copy()
        for i in range(len(edges)):
            if not mask&(1<<i):
                for node in descendants[i]:result[qi[node]:qi[node]+3]+=gap*directions[i]
        return result
    full=(1<<len(edges))-1;poses={mask:pose(mask) for mask in range(full+1)}
    checks={};transitions={}
    @lru_cache(None)
    def solve(mask):
        if mask==full:return 0.,[]
        components=groups(n,edges,mask);remaining=[i for i in range(len(edges)) if not mask&(1<<i)]
        best=(float('inf'),[])
        for bits in range(1,1<<len(remaining)):
            chosen=[remaining[j] for j in range(len(remaining)) if bits&(1<<j)]
            occupied=[components[edges[i][key]] for i in chosen for key in ('a','b')]
            if len(set(occupied))<len(occupied):continue
            next_mask=mask|sum(1<<i for i in chosen)
            # Cached motion collision checks are independent of execution.
            key=(mask,next_mask)
            valid=all(clear((1-f)*poses[mask]+f*poses[next_mask]) for f in np.linspace(0,1,7))
            checks[str(key)]=valid
            if not valid:continue
            displacement=max(np.linalg.norm(poses[next_mask][k:k+3]-poses[mask][k:k+3])for k in qi)
            duration=max(1.5,1.875*displacement/.18,np.sqrt(5.78*displacement/.5))
            cost=duration+1.55  # align .7 + dwell .15 + capture .1 + handover .6
            tail,path=solve(next_mask)
            if cost+tail<best[0]:best=(cost+tail,[dict(edges=chosen,start_mask=mask,end_mask=next_mask,duration=duration)]+path)
        return best
    predicted,rounds=solve(0)
    if not np.isfinite(predicted):raise ValueError('no screened merge schedule')
    filters.apply(m,[])
    # Identical physical modules may fill any vertex of the fixed tree.
    # Assign the sampled launch locations to vertices by minimum squared
    # transport distance. The sampled set, yaw values and graph are unchanged.
    goals=np.array([poses[0][k:k+2] for k in qi])
    _,assignment=linear_sum_assignment(np.sum((goals[:,None]-xy[None,:])**2,axis=2))
    original=initial.copy()
    for node,physical in enumerate(assignment):
        initial[qi[node]:qi[node]+7]=original[qi[physical]:qi[physical]+7]
    hover=initial.copy()
    for k in qi:hover[k+2]=1.5
    transit=transit_paths(m,hover,poses[0],qi,vi,air_clear)
    return dict(plant=p,edges=edges,initial=initial,closed=closed,poses=poses,rounds=rounds,qi=qi,vi=vi,
        directions=directions,descendants=descendants,filters=filters,transit=transit,
        evidence=dict(merge_overhead_s=1.55,final_hover_s=1.,predicted_merge_s=predicted,predicted_transit_s=sum(t for _,_,t in transit),
                      maximum_examined_wake_loss=maximum_predicted_loss,physical_module_assignment=assignment.tolist(),independent_support=support['status'],states=solve.cache_info().currsize,transitions=len(checks),screened_paths=checks,planning_s=time.perf_counter()-began))


def transit_paths(m,hover,target,qi,vi,air_clear=lambda data:True):
    """Screen synchronized paths; add height lanes only for actual conflicts."""
    d=mujoco.MjData(m);delta=np.zeros(m.nv);n=len(qi)
    gm=[]
    for i in range(m.ngeom):
        name=m.geom(i).name or ''
        gm.append(int(name.split('/')[0][1:]) if name.startswith('m') else -1)
    def conflicts(a,b):
        pairs=set();mujoco.mj_differentiatePos(m,delta,1.,a,b)
        for t in np.linspace(0.,1.,41):
            d.qpos[:]=a;mujoco.mj_integratePos(m,d.qpos,delta,float(t))
            mujoco.mj_kinematics(m,d);mujoco.mj_collision(m,d)
            if not air_clear(d):pairs.update((i,j) for i in range(n) for j in range(i+1,n))
            for contact in d.contact:
                i,j=gm[contact.geom1],gm[contact.geom2]
                if i!=j and i>=0 and j>=0 and contact.dist<-.00005:pairs.add(tuple(sorted((i,j))))
        return pairs
    def duration(a,b,speed):
        mujoco.mj_differentiatePos(m,delta,1.,a,b)
        linear=max(np.linalg.norm(b[k:k+3]-a[k:k+3]) for k in qi)
        angular=max(np.linalg.norm(delta[j+3:j+6]) for j in vi)
        return max(1.3,1.875*linear/speed,np.sqrt(5.78*linear/.8),1.875*angular/1.2)
    pairs=conflicts(hover,target)
    if not pairs:return [(hover.copy(),target.copy(),duration(hover,target,.9))]
    colors=[]
    for i in range(n):
        forbidden={colors[j]for j in range(i) if (j,i) in pairs}
        colors.append(next(c for c in range(n) if c not in forbidden))
    high=hover.copy();cross=target.copy()
    for k,c in zip(qi,colors):high[k+2]=cross[k+2]=1.5+.5*c
    path=[hover,high,cross,target]
    if any(conflicts(a,b) for a,b in zip(path[:-1],path[1:])):
        raise ValueError('no screened transit path')
    return [(a.copy(),b.copy(),duration(a,b,.8)) for a,b in zip(path[:-1],path[1:])]


def _data(m,q):
    d=mujoco.MjData(m);d.qpos[:]=q;return d
