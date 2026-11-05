"""One-cycle physical load screens and common real rotor-direction proposal."""
import numpy as np
import mujoco
import pinocchio as pin
from .equilibrium import static_balance
from .walking import ground_support


def motion_load(plant,drive,q,v,a):
    # RNE computes the same M*a+bias without constructing the full mass and
    # rotor maps, which the planning load screen never reads. Rotor/joint
    # armature is separate from the articulated-body inertia in mj_rne.
    m,d=drive.m,drive.d;d.qpos[:]=q;d.qvel[:]=v;d.qacc[:]=a
    mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d);mujoco.mj_comVel(m,d)
    force=np.empty(m.nv);mujoco.mj_rne(m,d,1,force);force+=m.dof_armature*a
    qp,_=plant.observation(d)
    _,_,pv,mv,bases=plant.coordinate_maps;result=np.empty(plant.pin.nv);result[pv]=force[mv]
    for pos,vel,source in bases:
        result[vel:vel+3]=pin.Quaternion(qp[pos+3:pos+7]).matrix().T@force[source:source+3]
    return qp,result


def ground_contacts(controller,stance):
    d=controller.drive.d;contacts=[]
    for active,link,(bid,vertices) in zip(stance,controller.links,controller.material):
        if not active:continue
        world=vertices@d.xmat[bid].reshape(3,3).T+d.xpos[bid]
        basis,ids=ground_support(d.xmat[bid].reshape(3,3),world)
        for j in np.unique(ids):contacts.append(dict(frame=link,point=vertices[j].tolist(),normal=[0.,0.,1.],tangent=basis[:,1].tolist(),mu=.8,max_force=None))
    return contacts


def initial_directions(plant,controller,states):
 """Keep existing directions if every sampled moving load is feasible.

 A failed sample goes directly to periodic direction synthesis. No unused
 constant-angle optimization or force-map construction is performed here.
 """
 dr=controller.drive;mq=dr.qi[np.array([r['gimbal'] for r in plant.meta['rotors']])];initial=states[0][0][mq].copy();reports=[]
 for q,v,a,stance in states:
  qp,load=motion_load(plant,dr,q,v,a);contacts=ground_contacts(controller,stance)
  report=static_balance(plant.pin,plant.meta,qp,contacts,generalized_load=load,fast_lp=True)
  reports.append(report)
  if report.get('minimum_peak_effort_ratio',float('inf'))>=.9:return initial,dict(passed=False,evaluations=0,stages=reports)
 return initial,dict(passed=True,evaluations=0,stages=reports)


def support_switch_times(period,count,duty=.62,phases=None):
    """Both one-sided limits of each support change, including cycle wrap."""
    offsets=np.arange(count)/count if phases is None else np.asarray(phases)
    phase=np.r_[-offsets,duty-offsets]%1.
    # Feet that move together share a transition. Check both sides once,
    # rather than creating duplicate allocation variables for the same state.
    return np.unique(np.r_[(phase-1e-6)%1.,(phase+1e-6)%1.])*period


def support_quadrature(period,count,duty=.62,phases=None,order=3):
    """Integrate each constant support interval with its actual duration.

    Uniform samples can repeatedly land just before a support change and
    overcount the cheaper support set. Interior Gauss nodes avoid that alias;
    weights sum to one, so the result is the cycle's time average.
    """
    offsets=np.arange(count)/count if phases is None else np.asarray(phases)
    cuts=np.unique(np.r_[0.,1.,(-offsets)%1.,(duty-offsets)%1.])
    nodes,weights=np.polynomial.legendre.leggauss(order)
    widths=np.diff(cuts);middle=(cuts[:-1]+cuts[1:])/2
    times=(middle[:,None]+widths[:,None]*nodes/2).ravel()*period
    return times,(widths[:,None]*weights/2).ravel()




def support_phases(plant,controller,q):
    """For four feet, compare every two-pair partition by real load capacity.

    Alternating opposing pairs avoids an arbitrary registration-order gait.
    Lower-dimensional gaits retain sequential swing, with three-foot support
    overlap increased by the gait generator. Full motion checks remain.
    """
    count=len(controller.links)
    if count!=4:return np.arange(count)/count,dict(method='sequential_swing')
    controller.drive.observe(q,np.zeros(plant.mj.nv));qp,_=plant.observation(controller.drive.d)
    points=controller.positions(controller.drive.d);com=controller.drive.d.subtree_com[0,:2];weight=plant.meta['mass']*9.80665
    reports=[]
    for pair in [(0,1),(0,2),(0,3)]:
        complement=tuple(i for i in range(4) if i not in pair);peaks=[];distances=[]
        for group in [pair,complement]:
            contacts=ground_contacts(controller,np.array([i in group for i in range(4)]))
            r=static_balance(plant.pin,plant.meta,qp,contacts,fast_lp=True,minimum_contact_normal_force_n=.55*weight)
            peaks.append(r.get('minimum_peak_effort_ratio',100.))
            a,b=points[list(group),:2];delta=b-a;closest=a+np.clip((com-a)@delta/max(delta@delta,1e-12),0.,1.)*delta
            distances.append(float(np.linalg.norm(com-closest)))
        reports.append(dict(pair=pair,peak=max(peaks),support_distance=max(distances)))
    best=min(reports,key=lambda r:(r['peak'],r['support_distance']))
    phase=np.array([0. if i in best['pair'] else .5 for i in range(4)])
    return phase,dict(method='support_pair_motor_capacity',alternatives=reports,selected_pair=best['pair'])


def support_timing(plant,controller,q):
    """Use support capacity to propose stance duration and swing order.

    These static capacities guide motion generation, not acceptance. Periodic
    geometry, real moving loads and the selected force series are checked later.
    """
    phase,base=support_phases(plant,controller,q);count=len(phase)
    weight=plant.meta['mass']*9.80665
    controller.drive.observe(q,np.zeros(plant.mj.nv))
    qp,_=plant.observation(controller.drive.d);cache={}
    def capacity(mask):
        key=tuple(mask)
        if key not in cache:
            report=static_balance(plant.pin,plant.meta,qp,ground_contacts(controller,mask),fast_lp=True,
                                  maximize_contact_load=True,peak_effort_bound=.9)
            cache[key]=sum(c['normal_force'] for c in report.get('contacts',[]))/weight
        return cache[key]
    def distribution(offsets,duty):
        # Integrate each support set's residence time, not the number of
        # transition samples. Unequal intervals must not receive equal weight.
        cuts=np.unique(np.r_[0.,1.,(-offsets)%1.,(duty-offsets)%1.])
        mid=(cuts[:-1]+cuts[1:])/2;dt=np.diff(cuts)
        cap=np.array([capacity((p+offsets)%1.<duty) for p in mid])
        return cap@dt,dt@(cap>=.55)
    modes=[phase]
    if count==4:modes.append(np.arange(count)/count)
    choices=[];start=.7 if count==3 else .62
    for offsets in modes:
        for duty in np.arange(start,.901,.02):
            mean,fraction=distribution(offsets,duty)
            if mean>=.55 and fraction>=.6-1e-8:
                choices.append((duty,offsets,mean,fraction));break
    if not choices:
        # Initial rotor directions can be poor. Keep the seed available for
        # direction refinement rather than claiming physical impossibility.
        return phase,dict(base,duty=start,capacity_proposal_unavailable=True)
    duty,phase,mean,fraction=min(choices,key=lambda z:z[0])
    return phase,dict(base,method='load_capacity_support_timing',duty=float(duty),
        capacity_mean=float(mean),capacity_time_55=float(fraction),
        capacities=[dict(stance=k,fraction=v) for k,v in cache.items()])
