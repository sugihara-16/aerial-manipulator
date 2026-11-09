"""Generate 2--8 support roles from task stroke and ground load.

One family partitions modules along 2--4 independent root branches. A broader
family assigns up to 8 real free anchors, including internal/root bodies.
Both assign actual ports on demand; neither selects a stored robot catalogue.
"""
import copy
import numpy as np,pinocchio as pin
from functools import lru_cache
import amsrr_det.task_structure as ts
from amsrr_det.model import build_model
from amsrr_det.objects import ObjectShape
from amsrr_det.grasp_geometry import fit_contacts,aim_rotors_up,grasp_frame
from amsrr_det.walking import foot_link,rotation_z,requires_face_alignment
from .planning_preview import GrowthPreview

def select(cost,rng):
 cost=np.array(cost);w=np.exp(-(cost-cost.min())/.025**2);return int(rng.choice(len(cost),p=w/w.sum()))

@lru_cache(None)
def module_center():
    model,_=build_model();data=model.createData()
    pin.centerOfMass(model,data,pin.neutral(model))
    return data.com[0].copy()

def stroke_limited_period(spec,points,period,duty):
    """Propose a finite stroke from part scale, not a reachability certificate.

    The outside feet of a turn travel faster. Limit their stance excursion
    before selecting ports; finite kinematics and full inertial loads still
    decide whether the resulting shorter cycle is possible.
    """
    points=np.asarray(points)
    velocity=np.c_[spec.speed*(1-spec.curvature*points[:,1]),
                   spec.speed*spec.curvature*points[:,0]]
    span=2*np.max(np.linalg.norm(ts.port_origins(),axis=1))
    return float(min(period,.65*span/(np.linalg.norm(velocity,axis=1).max()*duty)))

def motion_timing(spec,points,period,duty):
    """Trade stance length against the acceleration of the quintic swing.

    Reserving 70% of single-module acceleration for support/control is a
    proposal preference, not an actuator guarantee. Exact full-body dynamics
    and the original walking conditions remain the acceptance tests.
    """
    points=np.asarray(points);v=np.max(np.linalg.norm(np.c_[spec.speed*(1-spec.curvature*points[:,1]),spec.speed*spec.curvature*points[:,0]],axis=1))
    meta=build_model()[1]
    acceleration=.3*(sum(r['maximum'] for r in meta['rotors'])/meta['mass']-9.80665)
    nominal=period
    for _ in range(3):
        period=stroke_limited_period(spec,points,nominal,duty)
        # max |d²(10s³-15s⁴+6s⁵)/ds²| = 10/sqrt(3).
        duty=min(duty,max(.52,1-np.sqrt((10/np.sqrt(3))*v/(period*acceleration))))
    return stroke_limited_period(spec,points,nominal,duty),float(duty)

def radial_generate(spec,rng):
 ports,angles,seams,pads,names,*_=ts.part_frames(True);n=spec.modules;mass=build_model()[1]['mass'];counts=np.arange(2,min(4,n-1)+1)
 power=(n/counts)**1.5;weights=np.exp(-2*(power-power.min()));count=int(rng.choice(counts,p=weights/weights.sum()));duty=float(rng.uniform(.60,.78))
 lengths=np.full(count,(n-1)//count);lengths[rng.permutation(count)[:(n-1)%count]]+=1
 period=rng.uniform(1.6,2.25)*np.sqrt(max(1.,n/(count+1)));reach=2*np.max(np.linalg.norm(ts.port_origins(),axis=1));root=np.eye(4)
 heading=rng.uniform(-np.pi,np.pi)
 # A transverse seed for articulated branches is an empirical initialization,
 # not a fixed final pose or graph. Ports still follow task load and stroke,
 # and the complete posture, finite motion and input checks remain decisive.
 root[:3,:3]=rotation_z(np.copysign(np.pi/2,heading) if np.max(lengths)>1 else heading)
 root[2,3]=rng.uniform(.95,1.3)*reach
 azimuth=rng.uniform(-np.pi,np.pi);phases=azimuth+np.arange(count)*2*np.pi/count;radius=rng.uniform(.8,1.15)*reach
 radii=radius+.42*(lengths-1);targets=np.c_[radii*np.cos(phases),radii*np.sin(phases),np.full(count,.03)]
 period,duty=motion_timing(spec,targets,period,duty)
 floor=ObjectShape('box',[20.,20.,1.],1.);source=np.array([0,0,-.5,0,0,0,1.]);poses=[root];values=[{}];edges=[];occupied=set();surfaces=[];assigned=[];trace=[];cache={}
 preview=GrowthPreview(n);preview.update(poses,edges,surfaces,targets)
 for role in rng.permutation(count):
  parent=0;tip=targets[role];velocity=np.array([spec.speed-spec.speed*spec.curvature*tip[1],spec.speed*spec.curvature*tip[0],0.]);axis=np.cross([0.,0.,1.],velocity);axis/=np.linalg.norm(axis);force=np.array([0.,spec.speed**2*spec.curvature,9.80665])*(.85*mass*n/(count*duty))
  for depth in range(int(lengths[role])):
   child=len(poses);terminal=depth==lengths[role]-1
   # Intermediate joints approach the same load path; they do not count as
   # feet. More modules add articulation before the terminal contact is fit.
   fraction=(depth+1)/lengths[role];target=(1-fraction)*root[:3,3]+fraction*tip
   blocked=occupied|{(i,j)for i in range(len(poses))if i!=parent for j in range(4)}
   if terminal:choices=ts._expansion_choices(poses,blocked,target,np.array([0.,0.,-1.]),floor,source,[],cache,.45,axis,force,velocity*period*duty,support=True,lever_scale=.08,child_weight=0.)
   else:
    # An internal module is a body/seam target, not a fictitious floor pad.
    # Confusing these roles folds a long branch back into the central body.
    target[:2]=root[:2,3]+fraction*(tip[:2]-root[:2,3]);target[2]=root[2,3]-.1*depth
    choices=[];mate=np.diag([-1.,-1.,1.,1.]);middle=len(angles)//2
    for outgoing in range(4):
     if (parent,outgoing)in occupied:continue
     for incoming in range(4):
      if ports[outgoing][:3]==ports[incoming][:3]:continue
      bases=(poses[parent]@seams[outgoing,:,None]@mate@np.linalg.inv(seams[incoming])[None]).reshape(-1,4,4);pad=next(j for j in range(4)if j!=incoming)
      costs=np.sum((bases[:,:3,3]-target)**2,axis=1)+.03**2*(1-bases[:,2,2])**2
      distances=np.linalg.norm(bases[:,None,:3,3]-np.array(poses)[None,:,:3,3],axis=-1);separation=np.full(len(poses),.6);separation[parent]=.4;costs+=4*np.sum(np.maximum(separation-distances,0.)**2,axis=1)
      costs+=ts._clearance_cost(bases,incoming,pad,floor,source,[],support=True)[:,middle]
      for owner,port in [(poses[parent][None],outgoing),(bases,incoming)]:
       axes=owner[:,:3,:3]@ts.port_axes()[port];centers=owner[:,:3,3]+owner[:,:3,:3]@ts.port_origins()[port];lever=np.cross(force,axes);nominal=np.sum((tip-centers)*lever,axis=1);sweep=.5*np.abs(lever@(velocity*period*duty));costs+=.02**2*((abs(nominal)+sweep)/1.3)**2
      for k,value in enumerate(costs):
       aa,bb=divmod(k,len(angles));choices.append((float(value),parent,outgoing,incoming,pad,aa,bb,middle,bases[k]))
   if not choices:raise ValueError('no independent branch expansion')
   cost=[z[0]for z in choices];rank=np.argsort(cost)[:6]
   if terminal and lengths[role]==1:
    from .walking_leaf import score
    reports=[score(poses[parent],z[2],z[3],z[4],angles[list(z[5:8])],velocity,period,duty)for z in [choices[j]for j in rank]]
    pick=select([cost[j]+r['cost']for j,r in zip(rank,reports)],rng);selected=int(rank[pick]);motion=reports[pick]
   else:selected=select(cost,rng);motion=None
   residual,parent,out,inc,pad,aa,bb,cc,pose=choices[selected];edges.append(dict(a=parent,port_a=ports[out],b=child,port_b=ports[inc]));occupied.update([(parent,out),(child,inc)]);values[parent][names[out]]=float(angles[aa]);values.append({names[inc]:float(angles[bb])});poses.append(pose)
   if terminal:surfaces.append((child,ports[pad]));assigned.append(int(role));occupied.add((child,pad));values[child][names[pad]]=float(angles[cc])
   trace.append(dict(module=child,parent=parent,role=int(role),depth=depth,terminal=bool(terminal),cost=residual,motion=motion));parent=child
   preview.update(poses,edges,surfaces,targets)
 model,meta=build_model(n,edges);q=pin.neutral(model);q[:3]=root[:3,3];q[3:7]=pin.Quaternion(root[:3,:3]).coeffs()
 for act in meta['actuators']:
  owner,name=act['name'].split('/');q[act['q']]=values[int(owner[1:])].get(name,0.)
 links=[foot_link(*s)for s in surfaces];normals=np.tile([0.,0.,-1.],(count,1));q,fit=fit_contacts(model,meta,q,surfaces,targets[assigned],normals,normal_weight=[requires_face_alignment(port) for _,port in surfaces],base_rotation=True,max_evaluations=60,floor_clearance=.04,floor_contact_links=links,base_upright_weight=.05,position_metric=np.diag([.2,.2,1.]))
 data=model.createData();pin.framesForwardKinematics(model,data,q);actual=np.array([data.oMf[meta['frames'][grasp_frame(*s)]].translation for s in surfaces])
 if np.max(abs(actual[:,2]-.03))>.025:raise ValueError('simultaneous_foot_fit')
 return dict(modules=n,edges=edges,surfaces=surfaces,q=aim_rotors_up(model,meta,q),period=period,generation=dict(method='task_partitioned_support_branches',trace=trace,fit=fit,lengths=lengths,feet=count,duty=duty,phases=np.arange(count)/count,targets=targets))


from .task_structure import grow_to_contacts,part_frames,port_axes,port_origins
def shared_generate(spec,rng,*,foot_count=None):
    """Speed sets stride clearance; curved travel sets the required foot axis.

    Whole trees are grown from legal single-part ports. A one-to-one remaining
    foot assignment guides branching without consuming the same port twice.
    """
    n=spec.modules;counts=np.arange(2,9)
    # Prefer independent foot-bearing bodies. Multiple feet on one body stay
    # in the sampling support but have less independent positioning freedom.
    probability=np.exp(-4*np.maximum(counts-n,0)**2-.5*(n/counts)**2);probability/=probability.sum()
    count=int(rng.choice(counts,p=probability)) if foot_count is None else int(foot_count)
    if not 2<=count<=8:raise ValueError("walking needs 2..8 feet")
    duty=float(rng.uniform(.60,.78))
    # Lift acceleration scales as inverse period squared. More modules per
    # supporting branch call for slower joint cycles, at the same task speed.
    period=float(rng.uniform(1.1,1.5)*np.sqrt(max(1.,n/count)))
    floor=ObjectShape('box',[20.,20.,1.],1.);source=np.array([0,0,-.5,0,0,0,1.])
    radius=rng.uniform(.75,1.25)*(.35+.45*max(n/count-1.,0.))+.5*spec.speed*period
    span=2*np.max(np.linalg.norm(port_origins(),axis=1))
    # Adjacent support roles need room for their owning bodies. Increasing
    # their number on an unchanged small ring otherwise starts with overlap.
    radius=max(radius,span/(2*np.sin(np.pi/count)))
    azimuth=rng.uniform(-np.pi,np.pi)
    angles=np.arange(count)*2*np.pi/count+azimuth
    points=np.c_[radius*np.cos(angles),radius*np.sin(angles),np.full(count,.03)]
    transverse=rng.random()<.5
    if transverse:
        # Place successive support roles across travel, using actual part
        # span. This makes local self-support possible without prescribing
        # any assembled graph, port, body pose or accepted walking result.
        offsets=np.array([0]+[(-1)**(i+1)*((i+1)//2) for i in range(1,count)])
        points=np.c_[np.zeros(count),span*offsets,np.full(count,.03)]
        if count>3:
            columns=int(np.ceil(count/2))
            xy=np.array([[x,y] for y in np.arange(columns)-(columns-1)/2 for x in [-.5,.5]])
            xy=xy[np.argsort(np.sum(xy**2,axis=1),kind='stable')[:count]]
            xy-=xy.mean(axis=0)
            points=np.c_[span*xy,np.full(count,.03)]
    period,duty=motion_timing(spec,points,period,duty)
    normals=np.tile([0.,0.,-1.],(count,1))
    # The midpoint velocity relative to the turning body differs at each foot.
    velocity=np.c_[spec.speed-spec.speed*spec.curvature*points[:,1],spec.speed*spec.curvature*points[:,0],np.zeros(count)]
    axes=np.cross(np.tile([0.,0.,1.],(count,1)),velocity)
    # The new branch must transmit ground load while the foot traverses its
    # stance stroke. Bias actual port choices by their hinge lever arms before
    # a complete structure exists; exact periodic dynamics remain the judge.
    module_mass=build_model()[1]['mass']
    forces=np.tile([0.,spec.speed**2*spec.curvature,9.80665],(count,1))*(.85*module_mass*n/(duty*count))
    strokes=velocity*period*duty
    def root_score(port,k,sign,R):
        arm=R@(part_frames(True)[3][port,k,:3,3]-port_origins()[port]);axis=R@port_axes()[port]
        peak=max(abs(np.cross(arm+s*strokes[0],forces[0])@axis) for s in [-.5,.5])/1.3
        self_arm=R@(module_center()-part_frames(True)[3][port,k,:3,3])
        self_moment=module_mass*9.80665*np.linalg.norm(self_arm[:2])/1.3
        return dict(score=.1*(1-R[2,2])**2+peak**2+self_moment**2)
    independent=False
    root=None
    if count<n and not transverse and (independent or rng.random()<.5):
        root=np.eye(4);root[2,3]=rng.uniform(.18,.5)
    edges,surfaces,root,values,trace=grow_to_contacts(n,points,normals,rng,shape=floor,source=source,
        motion_axis=axes[0],root_pose=root,root_shortlist=80,body_spacing=.45,
        multiple_contacts=count>n,independent_branches=independent,contact_motion_axes=axes,coverage_weight=.5,capture_distance=.12,
        root_score=root_score,contact_forces=forces,contact_strokes=strokes,support=True,lever_scale=.08,child_weight=module_mass*9.80665)
    # A central root has no initial foot. The trace then contains all roles.
    order=[item['target_role'] for item in trace if item['contact_assigned']]
    if len(order)<len(surfaces):order=[0]+order
    model,meta=build_model(n,edges);q=pin.neutral(model)
    q[:3]=root[:3,3];q[3:7]=pin.Quaternion(root[:3,:3]).coeffs()
    for a in meta['actuators']:
        module,name=a['name'].split('/');q[a['q']]=values[int(module[1:])].get(name,0.)
    links=[foot_link(*s) for s in surfaces]
    q,fit=fit_contacts(model,meta,q,surfaces,points[order],normals[order],normal_weight=[requires_face_alignment(port) for _,port in surfaces],base_rotation=True,
        max_evaluations=60,floor_clearance=.04,floor_contact_links=links,base_upright_weight=.05)
    if fit['position_error_m']>.025:
        # The task specifies a floor, not preassigned foothold coordinates.
        # Keep an already fitted seed; otherwise free its planar placement.
        q,fit=fit_contacts(model,meta,q,surfaces,points[order],normals[order],normal_weight=[requires_face_alignment(port) for _,port in surfaces],base_rotation=True,
            max_evaluations=30,floor_clearance=.04,floor_contact_links=links,base_upright_weight=.05,
            position_metric=np.diag([.2,.2,1.]))
        data=model.createData();pin.framesForwardKinematics(model,data,q)
        fitted=np.array([data.oMf[meta['frames'][grasp_frame(*s)]].translation.copy() for s in surfaces])
        fit.update(floor_height_error_m=float(np.max(abs(fitted[:,2]-points[order,2]))),fitted_points=fitted.tolist(),
                   foot_placement='floor constraint with movable planar foot positions')
        if fit['floor_height_error_m']>.025:raise ValueError('simultaneous_foot_fit')
    q=aim_rotors_up(model,meta,q)
    return dict(modules=n,edges=edges,surfaces=surfaces,q=q,period=period,
        generation=dict(phases=np.array([list(dict.fromkeys(i for i,_ in surfaces)).index(owner)/len(set(i for i,_ in surfaces)) for owner,_ in surfaces]),duty=duty,feet=count,method='foot_stroke_and_load_lever_growth',transverse=bool(transverse),independent_branches=independent,trace=trace,points=points,axes=axes,
                        forces=forces,strokes=strokes,fit=fit))



def generate(spec,rng):
    # Both proposal families are driven by task stroke and load. The broad
    # family permits 2--8 support roles, including internal/free-root pads.
    # Later finite motion and physical screens decide whether those are legs.
    selector=np.random.Generator(rng.bit_generator.jumped())
    if selector.random()>=.4:return radial_generate(spec,rng)
    initial=copy.deepcopy(rng)
    try:return shared_generate(spec,rng)
    except ValueError:
        record=radial_generate(spec,initial)
        record['generation']['family_recovery']=True
        return record
