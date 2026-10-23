"""Solve grasp posture from contact targets; no task-ID or finished-shape lookup."""
import numpy as np
import pinocchio as pin
from .geometry_least_squares import least_squares
from itertools import combinations
from .model import PORTS
from functools import lru_cache


@lru_cache(maxsize=1)
def part_geometry():
    from .geometry import Geometry
    return Geometry()


@lru_cache(maxsize=1)
def part_bounds():
    from .part_bounds_cache import read_bounds
    cached = read_bounds()
    if cached is not None:
        return cached
    return [(link,T.act((shape.aabb_local.min_+shape.aabb_local.max_)/2),
             abs(T.rotation)@((shape.aabb_local.max_-shape.aabb_local.min_)/2))
            for link,T,shape in part_geometry().entries]


def grasp_frame(module,port):
    return f'm{module}/'+(('pitch_grasp_contact_frame_' if port.startswith('pitch')
                          else 'yaw_connect_dummy_')+port[-1])


def free_surfaces(meta):
    occupied={(e['a'],e['port_a']) for e in meta['edges']}|{(e['b'],e['port_b']) for e in meta['edges']}
    return [(m,p) for m in range(meta['modules']) for p in PORTS if (m,p) not in occupied]


def separating_translation(inward_normals, gap):
    """Shortest translation separating all supporting planes by at least gap.

    A minimum-norm point in three dimensions has at most three independent
    active planes. Actual material clearance is checked by the caller.
    Opposing normals can make translation alone impossible; return None then.
    """
    normals = np.asarray(inward_normals, dtype=float)
    if gap <= 0 or normals.ndim != 2 or normals.shape[1] != 3:
        raise ValueError('positive gap and three-dimensional normals required')
    best = None
    for count in range(1, min(3, len(normals))+1):
        for indices in combinations(range(len(normals)), count):
            active = normals[list(indices)]
            candidate = np.linalg.lstsq(active, np.full(count, -gap), rcond=None)[0]
            if np.max(normals@candidate+gap) > 1e-10: continue
            if best is None or candidate@candidate < best@best: best = candidate
    return best


def robot_bottom(model,meta,q):
    data=model.createData();pin.framesForwardKinematics(model,data,q);lowest=float('inf')
    for module in range(meta['modules']):
        for link,center,half in part_bounds():
            T=data.oMf[meta['frames'][f'm{module}/{link}']]
            lowest=min(lowest,float(T.act(center)[2]-abs(T.rotation[2])@half))
    return lowest


def opposed_contact_terms(points,normals,friction,point_jac=None,normal_jac=None,normal_sum_limit=None):
    """Both ends can push along their connecting line inside the friction cone.

    Zero error permits a pair of equal, opposite, collinear compressive forces;
    it does not certify gravity support or full grasp stability.
    """
    delta=points[1]-points[0];length=max(float(np.linalg.norm(delta)),1e-9);axis=delta/length
    cosine=1/np.sqrt(1+friction**2)
    raw=cosine-np.array([normals[0]@axis,-normals[1]@axis])
    if normal_sum_limit is not None:
        total=normals.sum(axis=0);total_length=float(np.linalg.norm(total))
        raw=np.r_[raw,total_length-normal_sum_limit]
    error=np.maximum(raw,0.)
    if point_jac is None:return error
    axis_jac=(np.eye(3)-np.outer(axis,axis))@(point_jac[1]-point_jac[0])/length
    derivative=np.vstack([-axis@normal_jac[0]-normals[0]@axis_jac,
                           axis@normal_jac[1]+normals[1]@axis_jac])
    if normal_sum_limit is not None:
        derivative=np.vstack([derivative,total@normal_jac.sum(axis=0)/max(total_length,1e-9)])
    return error,derivative*(raw>0)[:,None]


def fit_contacts(model,meta,q0,surfaces,points,inward_normals, *, max_evaluations=100,floor_clearance=None,base_rotation=False,object_shape=None,contact_gap=-.001,avoid_pairs=(),avoid_shapes=None,obstacles=(),analytic_jacobian=True,normal_moment_axis=None,normal_moment_samples=None,opposed_friction=None):
    if not set(surfaces)<=set(free_surfaces(meta)):
        raise ValueError('contact surface is occupied or does not exist')
    active=[a for a in meta['actuators'] if a['role']=='dock']
    qi=np.array([a['q'] for a in active]);data=model.createData()
    frames=[meta['frames'][grasp_frame(*s)] for s in surfaces]
    collision_parts={link:(T,shape) for link,T,shape in part_geometry().entries} if avoid_pairs else {}
    points=np.asarray(points);normals=np.asarray(inward_normals)
    bounds=[] if floor_clearance is None and not obstacles else [(meta['frames'][f'm{i}/{link}'],center,half)
            for i in range(meta['modules']) for link,center,half in part_bounds()]
    if points.shape!=(len(surfaces),3) or normals.shape!=points.shape or not np.allclose(np.linalg.norm(normals,axis=1),1):
        raise ValueError('invalid contact targets')
    if opposed_friction is not None and (len(surfaces)!=2 or object_shape is None or opposed_friction<=0):
        raise ValueError('opposed contact fitting needs two movable surface sites and positive friction')
    opposition_limit=float(np.linalg.norm(normals.sum(axis=0))) if opposed_friction is not None else None
    if normal_moment_axis is not None:
        normal_moment_axis=np.asarray(normal_moment_axis,dtype=float)
        if (object_shape is None or normal_moment_axis.shape!=(3,) or
            not np.isclose(np.linalg.norm(normal_moment_axis),1.) or
            normal_moment_samples is None or len(normal_moment_samples)!=len(surfaces)):
            raise ValueError('normal moment objective needs a shape, unit world axis and material patches')
        normal_moment_samples=[np.asarray(p,dtype=float) for p in normal_moment_samples]
        if any(p.ndim!=2 or p.shape[1]!=3 or len(p)==0 or not np.isfinite(p).all() for p in normal_moment_samples):
            raise ValueError('invalid normal moment material samples')
    # Base translation and every Dock joint; gimbals remain control variables.
    # Initial prototype keeps the root orientation given by the task frame.
    nr=3 if base_rotation else 0
    R0=pin.Quaternion(q0[3:7]).matrix()
    x0=np.r_[q0[:3],np.zeros(nr),q0[qi]]
    lo=np.r_[q0[:3]-2,np.full(nr,-1.4),[a['lower']+.04 for a in active]]
    hi=np.r_[q0[:3]+2,np.full(nr,1.4),[a['upper']-.04 for a in active]]
    from .native_grasp_geometry import GeometryKernel,library
    kernel=None
    if analytic_jacobian and not avoid_pairs and normal_moment_axis is None and library() is not None:
        kernel=GeometryKernel(model,frames+([meta['object']['frame']] if object_shape is not None else []),
                              bounds,obstacles,floor_clearance,len(x0))
    def configuration(x):
        q=q0.copy();q[:3]=x[:3];q[qi]=x[3+nr:]
        if base_rotation:q[3:7]=pin.Quaternion(pin.exp3(x[3:6])@R0).coeffs()
        return q
    def residual(x,want_jac=False):
        q=configuration(x)
        if want_jac:
            lift=np.zeros((model.nv,len(x)));rotation=pin.Quaternion(q[3:7]).matrix()
            lift[:3,:3]=rotation.T
            if nr:
                phi=x[3:6];theta=np.linalg.norm(phi);S=pin.skew(phi)
                left=np.eye(3)+(.5 if theta<1e-7 else (1-np.cos(theta))/theta**2)*S+(1/6 if theta<1e-7 else (theta-np.sin(theta))/theta**3)*(S@S)
                lift[3:6,3:6]=rotation.T@left
            lift[[a['v'] for a in active],3+nr+np.arange(len(active))]=1
            if kernel is not None:
                fused=kernel.fit(q,lift,points,normals,object_shape,contact_gap,x-x0,opposed_friction)
                if fused is not None:return fused
                poses,J,br,bj=kernel.evaluate(q,lift)
                count=len(frames);position=poses[:count,:3];normal=poses[:count,3:].reshape(-1,3,3)[:,:,0]
                linear=J[:count,:3];angular=J[:count,3:]
                normal_jac=np.cross(angular.transpose(0,2,1),normal[:,None,:]).transpose(0,2,1)
                if object_shape is None:
                    contact=np.concatenate([position-points,.15*(normal-normals)],axis=1)
                    derivative=np.concatenate([linear,.15*normal_jac],axis=1)
                else:
                    obj_position=poses[-1,:3];obj_rotation=poses[-1,3:].reshape(3,3)
                    local=(position-obj_position)@obj_rotation
                    surface,gaps,outward=object_shape.project_surface(local)
                    h=1e-6;offset=h*np.eye(3)
                    perturb=np.concatenate([local[:,None,:]+offset,local[:,None,:]-offset],axis=1)
                    _,dg,dn=object_shape.project_surface(perturb.reshape(-1,3))
                    dg=dg.reshape(count,6);dn=dn.reshape(count,6,3)
                    gap_gradient=(dg[:,:3]-dg[:,3:])/(2*h)
                    normal_gradient=((dn[:,:3]-dn[:,3:])/(2*h)).transpose(0,2,1)
                    local_jac=obj_rotation.T@linear
                    contact=np.concatenate([(gaps-contact_gap)[:,None],
                                            .15*(normal+outward@obj_rotation.T),.03*(position-points)],axis=1)
                    derivative=np.concatenate([(gap_gradient[:,None,:]@local_jac),
                        .15*(normal_jac+obj_rotation@normal_gradient@local_jac),.03*linear],axis=1)
                extra=np.empty(0);extra_jac=np.empty((0,len(x)))
                if opposed_friction is not None:
                    sj=(np.eye(3)-outward[:,:,None]*gap_gradient[:,None,:]-gaps[:,None,None]*normal_gradient)@local_jac
                    extra,extra_jac=opposed_contact_terms(surface,-outward,opposed_friction,sj,-normal_gradient@local_jac,opposition_limit)
                return (np.r_[contact.ravel(),.15*extra,br,.0001*(x-x0)],
                        np.vstack([derivative.reshape(-1,len(x)),.15*extra_jac,bj,.0001*np.eye(len(x))]))
            pin.computeJointJacobians(model,data,q);pin.updateFramePlacements(model,data)
        else:pin.framesForwardKinematics(model,data,q)
        r=[];rows=[];jacobians={}
        def jac(frame):
            if frame not in jacobians:jacobians[frame]=pin.getFrameJacobian(model,data,frame,pin.LOCAL_WORLD_ALIGNED)@lift
            return jacobians[frame]
        if object_shape is not None:
            obj=data.oMf[meta['object']['frame']]
            local_points=np.array([obj.actInv(data.oMf[f].translation) for f in frames])
            surface,gaps,outwards=object_shape.project_surface(local_points)
            if want_jac:
                h=1e-6;offset=h*np.eye(3)
                perturb=np.concatenate([local_points[:,None,:]+offset,local_points[:,None,:]-offset],axis=1)
                _,dg,dn=object_shape.project_surface(perturb.reshape(-1,3))
                dg=dg.reshape(len(frames),6);dn=dn.reshape(len(frames),6,3)
                gap_gradients=(dg[:,:3]-dg[:,3:])/(2*h)
                normal_gradients=((dn[:,:3]-dn[:,3:])/(2*h)).transpose(0,2,1)
        for pad,(f,p,n) in enumerate(zip(frames,points,normals)):
            T=data.oMf[f]
            if object_shape is None:
                r.extend(T.translation-p);r.extend(.15*(T.rotation[:,0]-n))
                if want_jac:
                    J=jac(f);rows.extend(J[:3]);rows.extend(-.15*pin.skew(T.rotation[:,0])@J[3:])
            else:
                r.append(gaps[pad]-contact_gap)
                r.extend(.15*(T.rotation[:,0]+obj.rotation@outwards[pad]))
                r.extend(.03*(T.translation-p))
                if want_jac:
                    # Differentiate only the three-coordinate surface query;
                    # propagate through the analytic whole-robot Jacobian.
                    J=jac(f);local_jac=obj.rotation.T@J[:3]
                    rows.append(gap_gradients[pad]@local_jac)
                    rows.extend(.15*(-pin.skew(T.rotation[:,0])@J[3:]+obj.rotation@normal_gradients[pad]@local_jac))
                    rows.extend(.03*J[:3])
                if normal_moment_axis is not None:
                    # Each known material sample contributes independently.
                    # This soft geometric preference bounds no contact force
                    # and does not certify a complete finite contact region.
                    material=normal_moment_samples[pad]
                    world=material@T.rotation.T+T.translation
                    local_material=(world-obj.translation)@obj.rotation
                    axis=obj.rotation.T@normal_moment_axis
                    _,_,outward_material=object_shape.project_surface(local_material)
                    scale=1/np.sqrt(len(material))
                    r.extend(scale*(np.cross(local_material,outward_material)@axis))
                    if want_jac:
                        perturb=np.concatenate([local_material[:,None,:]+offset,
                                                local_material[:,None,:]-offset],axis=1)
                        _,_,dn=object_shape.project_surface(perturb.reshape(-1,3))
                        arms=np.cross(perturb,dn.reshape(perturb.shape))@axis
                        gradients=(arms[:,:3]-arms[:,3:])/(2*h)
                        for point,gradient in zip(material,gradients):
                            point_jac=J[:3]-pin.skew(T.rotation@point)@J[3:]
                            rows.append(scale*gradient@obj.rotation.T@point_jac)
        if opposed_friction is not None:
            if want_jac:
                local_jac=np.array([obj.rotation.T@jac(f)[:3] for f in frames])
                sj=(np.eye(3)-outwards[:,:,None]*gap_gradients[:,None,:]-gaps[:,None,None]*normal_gradients)@local_jac
                extra,extra_jac=opposed_contact_terms(surface,-outwards,opposed_friction,sj,-normal_gradients@local_jac,opposition_limit)
                rows.extend(.15*extra_jac)
            else:extra=opposed_contact_terms(surface,-outwards,opposed_friction,normal_sum_limit=opposition_limit)
            r.extend(.15*extra)
        if bounds:
            # Keep each transform evaluation in the original arithmetic order.
            # Batch only the box-overlap and inactive-row bookkeeping.
            placements=[data.oMf[frame] for frame,_,_ in bounds]
            centers=np.array([T.act(center) for T,(_,center,_) in zip(placements,bounds)])
            halves=np.array([abs(T.rotation)@half for T,(_,_,half) in zip(placements,bounds)])
            overlap_columns=[];axis_columns=[];sign_columns=[]
            if floor_clearance is not None:
                overlap_columns.append(floor_clearance-centers[:,2]+halves[:,2])
                axis_columns.append(np.full(len(bounds),2));sign_columns.append(np.ones(len(bounds)))
            if obstacles:
                boxes=np.asarray(obstacles)
                delta=centers[:,None,:]-boxes[None,:,0,:]
                depth=halves[:,None,:]+boxes[None,:,1,:]+.035-abs(delta)
                axes=np.argmin(depth,axis=2)
                overlaps=np.take_along_axis(depth,axes[:,:,None],axis=2)[:,:,0]
                signs=np.take_along_axis(np.sign(delta),axes[:,:,None],axis=2)[:,:,0]
                overlap_columns.extend(overlaps.T);axis_columns.extend(axes.T);sign_columns.extend(signs.T)
            overlaps=np.column_stack(overlap_columns);axes=np.column_stack(axis_columns);signs=np.column_stack(sign_columns)
            r.extend((2*np.maximum(0.,overlaps)).ravel())
            if want_jac:
                derivatives=np.zeros((*overlaps.shape,len(x)))
                for index in np.flatnonzero(np.any(overlaps>0,axis=1)):
                    frame,center,half=bounds[index];T=placements[index]
                    J=jac(frame);center_jac=J[:3]-pin.skew(T.rotation@center)@J[3:]
                    half_jac=np.zeros((3,len(x)))
                    for axis in range(3):
                        half_jac-=np.sign(T.rotation[:,axis])[:,None]*half[axis]*(pin.skew(T.rotation[:,axis])@J[3:])
                    for column in np.flatnonzero(overlaps[index]>0):
                        axis=axes[index,column];sign=signs[index,column]
                        derivatives[index,column]=2*(half_jac[axis]-sign*center_jac[axis])
                rows.extend(derivatives.reshape(-1,len(x)))
        if avoid_pairs:
            import coal
            request=coal.DistanceRequest();request.enable_signed_distance=True
            for first,second in avoid_pairs:
                transforms=[];shapes=[]
                for name in [first,second]:
                    if avoid_shapes and name in avoid_shapes:local,shape=avoid_shapes[name]
                    else:local,shape=collision_parts[name.split('/',1)[1]]
                    T=data.oMf[meta['frames'][name]]*local if name in meta['frames'] else local
                    transforms.append(coal.Transform3s(T.rotation,T.translation));shapes.append(shape)
                result=coal.DistanceResult()
                distance=coal.distance(shapes[0],transforms[0],shapes[1],transforms[1],request,result)
                r.append(2*max(0.,.03-distance))
                if want_jac:
                    if distance<.03:
                        point_jacs=[]
                        for name,point in zip([first,second],[result.getNearestPoint1(),result.getNearestPoint2()]):
                            if name not in meta['frames']:
                                point_jacs.append(np.zeros((3,len(x))));continue
                            frame=meta['frames'][name];J=jac(frame)
                            point_jacs.append(J[:3]-pin.skew(np.asarray(point)-data.oMf[frame].translation)@J[3:])
                        rows.append(-2*np.asarray(result.normal)@(point_jacs[1]-point_jacs[0]))
                    else:rows.append(np.zeros(len(x)))
        r.extend(.0001*(x-x0))
        if want_jac:
            rows.extend(.0001*np.eye(len(x)));return np.asarray(r),np.asarray(rows)
        return np.asarray(r)
    # The solver requests the residual and derivative at the same posture.
    # Share that evaluation, retaining only the last posture within this fit.
    cached_x=None;cached_pair=None
    def evaluate(x):
        nonlocal cached_x,cached_pair
        if cached_x is None or not np.array_equal(x,cached_x):
            cached_pair=residual(x,True);cached_x=x.copy()
        return cached_pair
    objective=(lambda x:evaluate(x)[0]) if analytic_jacobian else residual
    jacobian=(lambda x:evaluate(x)[1]) if analytic_jacobian else '2-point'
    result=least_squares(objective,np.clip(x0,lo,hi),bounds=(lo,hi),max_nfev=max_evaluations,
                         jac=jacobian,ftol=1e-9,xtol=1e-9,gtol=1e-9)
    q=configuration(result.x);pin.framesForwardKinematics(model,data,q)
    report=dict(position_error_m=max(np.linalg.norm(data.oMf[f].translation-p) for f,p in zip(frames,points)),
                normal_error=max(np.linalg.norm(data.oMf[f].rotation[:,0]-n) for f,n in zip(frames,normals)),
                evaluations=result.nfev,scope='kinematic_fit_only',task_success=False)
    if object_shape is not None:
        obj=data.oMf[meta['object']['frame']]
        surface,gaps,outward=object_shape.project_surface(np.array([obj.actInv(data.oMf[f].translation) for f in frames]))
        report.update(position_error_m=float(np.max(abs(gaps-contact_gap))),
            normal_error=max(np.linalg.norm(data.oMf[f].rotation[:,0]+obj.rotation@n) for f,n in zip(frames,outward)),
            object_points=surface.tolist(),inward_normals=(-outward).tolist())
        if opposed_friction is not None:
            report.update(opposed_friction=opposed_friction,
                opposition_normal_sum_limit=opposition_limit,
                opposition_error=float(opposed_contact_terms(surface,-outward,opposed_friction,normal_sum_limit=opposition_limit).max()))
    return q,report


def aim_rotors_up(model,meta,q):
    """Choose allowed gimbal angles that best align each rotor with world up."""
    q=q.copy();data=model.createData();pin.framesForwardKinematics(model,data,q)
    up=np.array([0.,0.,1.])
    for rotor in meta['rotors']:
        a=meta['actuators'][rotor['gimbal']]
        direction=data.oMf[rotor['frame']].rotation[:,2]
        axis=data.oMi[a['joint']].rotation@pin.computeJointJacobian(model,data,q,a['joint'])[3:,a['v']]
        # Every gimbal belongs to its own rotor branch; updating one does not
        # change any other rotor's placement used in this pass.
        theta=np.arctan2(up@np.cross(axis,direction),up@direction-(up@axis)*(axis@direction))
        candidates=[np.clip(q[a['q']]+theta+turn*2*np.pi,a['lower']+.02,a['upper']-.02) for turn in [-1,0,1]]
        def score(angle):
            delta=angle-q[a['q']]
            rotated=direction*np.cos(delta)+np.cross(axis,direction)*np.sin(delta)+axis*(axis@direction)*(1-np.cos(delta))
            return up@rotated
        q[a['q']]=max(candidates,key=score)
    return q
