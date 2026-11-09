"""Shared real actuator geometry and ground support refine a walking seed.

Several support sets share one posture and real gimbal angles, with separate
force allocations. This is a static proposal; periodic motion is checked later.
"""
import copy
import numpy as np,pinocchio as pin,mujoco
from .sparse_least_squares import least_squares
from scipy.sparse import csr_matrix
from .walking_load import support_phases
from .grasp_geometry import part_bounds
from .walking import requires_face_alignment


def support_durations(phases,duty):
    """Fraction of a cycle spent in each support set, including wraparound."""
    phases=np.asarray(phases)
    cuts=np.unique(np.r_[0.,1.,(-phases)%1.,(duty-phases)%1.]);duration={}
    for lo,hi in zip(cuts[:-1],cuts[1:]):
        active=tuple(np.flatnonzero(((lo+hi)/2+phases)%1.<duty))
        duration[active]=duration.get(active,0.)+hi-lo
    return duration


def propose_support_pose(plant,controller,pose,*,iterations,duty=.62,phases=None,motion=None):
    """Preserve mobile postures; repair proposals with insufficient support.

    The static capacity is a proposal heuristic, not an impossibility test.
    Both choices must pass the same complete moving-load and CAD screens.
    """
    from .walking_control import GroundControl
    from .walking_load import ground_contacts
    from .equilibrium import static_balance
    weak,report=refine_support_pose(plant,controller,pose,support_weight=5.,iterations=min(48,iterations),duty=duty,phases=phases,motion=motion)
    control=GroundControl(plant,weak,controller.links)
    phases=np.arange(len(control.links))/len(control.links) if phases is None else np.asarray(phases)
    control.drive.observe(weak,np.zeros(plant.mj.nv));qp,_=plant.observation(control.drive.d)
    weight=plant.meta['mass']*9.80665
    cuts=np.unique(np.r_[0.,1.,(-phases)%1.,(duty-phases)%1.]);dt=np.diff(cuts);cache={}
    mid=(cuts[:-1]+cuts[1:])/2
    for t in mid:
        mask=tuple((t+phases)%1<duty)
        if mask not in cache:
            force=static_balance(plant.pin,plant.meta,qp,ground_contacts(control,mask),fast_lp=True,
                                maximize_contact_load=True,peak_effort_bound=.9)
            cache[mask]=sum(z['normal_force'] for z in force.get('contacts',[]))/weight
    capacity=np.array([cache[tuple((t+phases)%1<duty)] for t in mid])
    mean=float(dt@capacity);support_time=float(dt@(capacity>=.55))
    repair=mean<.55 or support_time<.8-1e-8
    parameters=dict(support_weight=20. if repair else 5.,capacity_timing=repair,
                    initial_capacity_mean=mean,initial_capacity_time=support_time)
    used=report['nfev'];state=report.pop('_solution')
    if repair or (used>=48 and iterations>48):
        # Continue the same bounded pose/force problem from its current
        # solution. Do not repeat the first 48 iterations from the old seed.
        weak,report=refine_support_pose(plant,controller,pose,support_weight=20. if repair else 5.,
            iterations=max(16,iterations-used),duty=duty,phases=phases,initial=state,motion=motion)
        report.pop('_solution');report['nfev']+=used
    return weak,report,parameters


def refine_support_pose(plant,controller,pose,*,iterations=24,support_weight=20.,duty=.62,phases=None,initial=None,motion=None):
    model,meta=plant.pin,plant.meta;m=plant.mj;dr=controller.drive;dr.observe(pose,np.zeros(m.nv));q0,_=plant.observation(dr.d)
    actuators=meta['actuators'];qi=np.array([a['q'] for a in actuators]);vi=np.array([a['v'] for a in actuators]);limits=np.array([a['effort'] for a in actuators]);nj=len(qi);nr=len(meta['rotors']);weight=meta['mass']*9.80665
    R0=pin.Quaternion(q0[3:7]).matrix();npv=6+nj;data=model.createData();pin.framesForwardKinematics(model,data,q0)
    frames=[meta['frames'][link] for link in controller.links];points=controller.foot_points.copy();nf=len(frames)
    aligned=np.array([requires_face_alignment(link) for link in controller.links])
    nominal=np.array([data.oMf[f].act(p) for f,p in zip(frames,points)]);nominal[:,2]=0.
    radius=max(.2,float(np.max(np.linalg.norm(nominal[:,:2]-np.mean(nominal[:,:2],axis=0),axis=1))))
    base_scale=np.r_[np.full(3,100/weight),np.full(3,100/(weight*radius))]
    phase=np.arange(nf)/nf if phases is None else np.asarray(phases)
    cuts=np.unique(np.r_[0.,1.,(-phase)%1.,(duty-phase)%1.]);sets=sorted({tuple(((t+phase)%1)<duty) for t in (cuts[:-1]+cuts[1:])/2})
    duration=support_durations(phase,duty)
    # This is a numerical search window around the floating-body seed, not
    # an actuator limit. Real joint, contact and finite-motion bounds follow.
    x0=list(np.r_[q0[:3],np.zeros(3),q0[qi]]);lo=list(np.r_[q0[:3]-.5,[-1.2]*3,[a['lower']+(0.02 if a['role']=='dock' else .02) for a in actuators]]);hi=list(np.r_[q0[:3]+.5,[1.2]*3,[a['upper']-(0.02 if a['role']=='dock' else .02) for a in actuators]])
    stages=[]
    for active in sets:
        ids=np.flatnonzero(active);begin=len(x0);stages.append((ids,begin))
        x0.extend(np.r_[np.full(nr,.4*weight/nr),np.tile([.6*weight/len(ids),0.,0.],len(ids))])
        lo.extend([0.]*nr+list(np.tile([0.,-200.,-200.],len(ids))));hi.extend([r['maximum'] for r in meta['rotors']]+[200.]*(3*len(ids)))
    movement=[]
    if motion is not None:
        moving=np.r_[np.arange(6),[a['v'] for a in actuators if a['role']=='dock']]
        rated=np.r_[np.ones(3),np.full(3,3.),[a.get('rated_speed') or a['speed'] for a in actuators if a['role']=='dock']]
        local=nominal-nominal.mean(axis=0)
        swing=np.c_[motion.speed*(1-motion.curvature*local[:,1])/(1-duty),
                    motion.speed*motion.curvature*local[:,0]/(1-duty),np.full(nf,.12)]
        # Only guide the two root-foot modes here. Keep the most constrained
        # support set in each mode; every actual support phase is still used
        # in the force proposal and the complete finite-motion acceptance.
        roots={i for i,link in enumerate(controller.links) if link.startswith('m0/')}
        representatives={}
        for ids,_ in stages:
            mode=bool(roots.intersection(ids))
            if mode not in representatives or len(ids)>len(representatives[mode]):representatives[mode]=ids
        for ids in representatives.values():
            begin=len(x0);movement.append((ids,begin))
            x0.extend(np.zeros(len(moving)));lo.extend(-rated);hi.extend(rated)
    # Ground coordinates are normal, world x, world y.
    basis=np.array([[0.,1.,0.],[0.,0.,1.],[1.,0.,0.]])
    bounds=[(meta['frames'][f'm{i}/{name}'],center,half) for i in range(meta['modules']) for name,center,half in part_bounds() if f'm{i}/{name}' not in controller.links]
    collision_model=copy.copy(m);floor=m.geom('floor').id
    collision_model.geom_contype[floor]=0;collision_model.geom_conaffinity[floor]=0
    collision_data=mujoco.MjData(collision_model)
    import ctypes as ct
    from .walking_native import library
    native_clearance=library().walking_posture_clearance
    native_clearance.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_int]+[ct.c_void_p]*5
    native_clearance.restype=None
    bodies=np.array([m.body(model.frames[f].name).id for f,_,_ in bounds],np.int32)
    centers=np.array([c for _,c,_ in bounds]);halves=np.array([h for _,_,h in bounds]);cr=np.empty(len(bounds)+1);cJ=np.empty((len(bounds)+1,m.nv))
    # Coordinate conversion changes only floating translation here. Angular
    # velocities are body-local in both engines; joint order follows metadata.
    _,_,pvi,mvi,bases=plant.coordinate_maps
    rotor_wrenches=np.zeros((model.njoints,6,nr))
    for k,rotor in enumerate(meta['rotors']):
        frame=model.frames[rotor['frame']]
        rotor_wrenches[frame.parentJoint,:,k]=frame.placement.act(pin.Force(np.array([0.,0.,1.]),np.zeros(3))).vector
    mobility_data=model.createData() if movement else None
    external=[pin.Force.Zero() for _ in range(model.njoints)]
    last=None;pair=None
    def configuration(x):
        q=q0.copy();q[:3]=x[:3];q[3:7]=pin.Quaternion(pin.exp3(x[3:6])@R0).coeffs();q[qi]=x[6:npv];return q
    def point_at(frame,point):
        # Joint Jacobians and placements are already current for this pose.
        J=pin.getFrameJacobian(model,data,frame,pin.LOCAL_WORLD_ALIGNED)
        arm=data.oMf[frame].rotation@np.asarray(point)
        return J[:3]-pin.skew(arm)@J[3:]
    def evaluate(x):
        nonlocal last,pair
        if last is not None and np.array_equal(last,x):return pair
        q=configuration(x);pin.computeJointJacobians(model,data,q);pin.updateFramePlacements(model,data)
        lift=np.zeros((model.nv,npv));R=pin.Quaternion(q[3:7]).matrix();lift[:3,:3]=R.T
        phi=x[3:6];theta=np.linalg.norm(phi);S=pin.skew(phi);left=np.eye(3)+(.5 if theta<1e-7 else (1-np.cos(theta))/theta**2)*S+(1/6 if theta<1e-7 else (theta-np.sin(theta))/theta**3)*S@S
        lift[3:6,3:6]=R.T@left;lift[vi,6+np.arange(nj)]=1.
        positions=np.array([data.oMf[f].act(p) for f,p in zip(frames,points)])
        support_points=[]
        for i,(f,(_,vertices)) in enumerate(zip(frames,controller.material)):
            world=vertices@data.oMf[f].rotation.T+data.oMf[f].translation
            j=int(np.argmin(world[:,2]));support_points.append(vertices[j]);positions[i,2]=world[j,2]
        Jp=np.array([point_at(f,p) for f,p in zip(frames,support_points)])
        tracking=np.array([point_at(f,p) for f,p in zip(frames,points)]);tracking[:,2]=Jp[:,2]
        B=np.column_stack([pin.getFrameJacobian(model,data,r['frame'],pin.LOCAL_WORLD_ALIGNED)[:3].T@data.oMf[r['frame']].rotation[:,2] for r in meta['rotors']]);g=pin.computeGeneralizedGravity(model,data,q).copy()
        res=np.empty(len(row_columns));rows=np.zeros((len(row_columns),len(x)));cursor=0
        def append(value,pose_block=None,force_block=None,start=None):
            nonlocal cursor
            value=np.atleast_1d(value);size=len(value);res[cursor:cursor+size]=value
            if pose_block is not None:rows[cursor:cursor+size,:npv]=pose_block
            if force_block is not None:rows[cursor:cursor+size,start:start+force_block.shape[1]]=force_block
            cursor+=size
        w=np.array([.5,.5,500.]);append(((positions-nominal)*w).ravel(),(tracking@lift*w[None,:,None]).reshape(3*nf,npv))
        for f,required in zip(frames,aligned):
            if not required:continue
            normal=data.oMf[f].rotation[:,0]
            angular=pin.getFrameJacobian(model,data,f,pin.LOCAL_WORLD_ALIGNED)[3:]
            append(100*(normal-np.array([0.,0.,-1.])), -100*pin.skew(normal)@angular@lift)
        for ids,start in stages:
            force=x[start:start+nr];cf=x[start+nr:start+nr+3*len(ids)].reshape(-1,3);C=np.column_stack([Jp[i].T@basis for i in ids]);F=np.c_[B,C];tau=g-F@np.r_[force,cf.ravel()]
            values=rotor_wrenches@force;extra=np.zeros((model.nv,model.nv))
            for i,value in enumerate(values):external[i].vector[:]=value
            for i,fv in zip(ids,cf):
                f=frames[i];frame=model.frames[f];R=data.oMf[f].rotation;world=basis@fv;local=R.T@world;p=support_points[i]
                external[frame.parentJoint]+=frame.placement.act(pin.Force(local,np.cross(p,local)))
                J=pin.getFrameJacobian(model,data,f,pin.LOCAL_WORLD_ALIGNED)
                extra-=Jp[i].T@pin.skew(world)@J[3:]
            deriv=pin.computeRNEADerivatives(model,data,q,np.zeros(model.nv),np.zeros(model.nv),external)[0].copy()+extra
            excess=np.maximum(abs(tau[vi])/limits-.7,0.);factor=np.sign(tau[vi])*(excess>0)/limits
            excess*=1.0;factor*=1.0
            rr=np.r_[base_scale*tau[:6],excess];Jq=np.vstack([base_scale[:,None]*deriv[:6],factor[:,None]*deriv[vi]])@lift;Jf=np.vstack([-base_scale[:,None]*F[:6],-factor[:,None]*F[vi]])
            append(rr,Jq,Jf,start)
            cone=np.maximum(abs(cf[:,1])+abs(cf[:,2])-.8*cf[:,0],0.);row=np.zeros((len(ids),3*len(ids)))
            for j in range(len(ids)):row[j,3*j:3*j+3]=np.array([-.8,np.sign(cf[j,1]),np.sign(cf[j,2])])*float(cone[j]>0)
            append(cone*100/weight,force_block=row*100/weight,start=start+nr)
            deficit=max(0.,.6*weight-sum(cf[:,0]));row=np.zeros((1,3*len(ids)));row[0,::3]=-support_weight/weight*float(deficit>0)
            append(support_weight/weight*deficit,force_block=row,start=start+nr)
            # Rotor effort and joint holding load now alter the pose itself.
            # This residual is the square root of the original convex proxy.
            es=np.sqrt(20.0/(weight**1.5/np.sqrt(nr))*duration[tuple(ids)])
            soft=1e-4;rr=es*((force+soft)**.75-soft**.75)
            append(rr,force_block=np.diag(es*.75*(force+soft)**-.25),start=start)
            js=np.sqrt(20.0*.05/nj*duration[tuple(ids)])/limits
            append(js*tau[vi],js[:,None]*(deriv[vi]@lift),-js[:,None]*F[vi],start)
        for ids,start in movement:
            from .walking_mobility import residual
            velocity=np.zeros(model.nv);velocity[moving]=x[start:start+len(moving)]
            rr,Jq,Jv=residual(model,q,velocity,moving,frames,support_points,ids,[motion.speed,0.,0.],swing,data=mobility_data)
            append(rr,Jq@lift,Jv,start)
        collision_data.qpos[:]=plant.configuration(q)
        native_clearance(collision_model._address,collision_data._address,len(bounds),bodies.ctypes.data,centers.ctypes.data,halves.ctypes.data,cr.ctypes.data,cJ.ctypes.data)
        tangent=np.zeros((m.nv,npv));tangent[mvi]=lift[pvi]
        for pos,vel,source in bases:
            tangent[source:source+3]=pin.Quaternion(q[pos+3:pos+7]).matrix()@lift[vel:vel+3]
        append(cr,cJ@tangent)
        regular=np.full(len(x),1e-5);regular[:npv]=.01
        res[cursor:]=regular*(x-x0);rows[cursor+np.arange(len(x)),np.arange(len(x))]=regular
        last=x.copy();pair=res,rows;return pair
    # Fixed block pattern includes every possible pose derivative. Avoid a
    # dense nonzero scan on each Jacobian without dropping structural zeros.
    row_columns=[]
    row_columns.extend([np.arange(npv)]*(3*nf+3*sum(aligned)))
    for ids,start in stages:
        width=nr+3*len(ids);nrrows=6+nj+len(ids)+1+nr+nj
        row_columns.extend([np.r_[np.arange(npv),np.arange(start,start+width)]]*nrrows)
    for ids,start in movement:
        row_columns.extend([np.r_[np.arange(npv),np.arange(start,start+len(moving))]]*(3*nf+3*sum(aligned[ids])+3))
    row_columns.extend([np.arange(npv)]*(len(bounds)+1))
    row_columns.extend([np.array([i]) for i in range(len(x0))])
    columns=np.concatenate(row_columns);offsets=np.r_[0,np.cumsum([len(c) for c in row_columns])];ri=np.repeat(np.arange(len(row_columns)),np.diff(offsets))
    def jacobian(x):
        J=evaluate(x)[1]
        if len(J)!=len(row_columns):raise RuntimeError('posture Jacobian pattern mismatch')
        sparse=csr_matrix((J[ri,columns],columns,offsets),shape=J.shape);sparse.eliminate_zeros();return sparse
    x0=np.array(x0);start=x0 if initial is None else np.asarray(initial)
    if start.shape!=x0.shape:raise ValueError('posture continuation shape changed')
    blocks=[begin for _,begin in stages+movement]+[len(x0)]
    result=least_squares(lambda x:evaluate(x)[0],np.clip(start,lo,hi),jac=jacobian,tr_solver='lsmr',direct=True,blocks=blocks,bounds=(lo,hi),max_nfev=iterations,ftol=1e-5,xtol=1e-6,gtol=1e-5)
    return plant.configuration(configuration(result.x)),dict(nfev=result.nfev,cost=float(result.cost),method='shared_pose_real_actuator_support_sets',_solution=result.x.copy())
