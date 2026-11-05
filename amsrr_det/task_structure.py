"""Task contact requirements drive incremental port-labelled tree synthesis.

Single-module kinematics are prepared once. Exact repeated expansion scores
can be shared within a planning call; every random expansion draw is retained.
A finite joint grid guides generation; continuous physical screens accept it.
"""
from functools import lru_cache
import time
import numpy as np
import pinocchio as pin
from .model import build_model, PORTS
from .grasp_geometry import grasp_frame, fit_contacts, part_bounds
from .planning_memo import evaluate as memoized
from .native_grasp_geometry import growth_clearance


@lru_cache(maxsize=2)
def part_frames(support=False):
    model, meta = build_model()
    data = model.createData()
    ports = tuple(PORTS)
    # A support branch can need near-right-angle folds to keep the added
    # body's rotors upright. Stay inside the authored +/- pi/2 Dock range.
    angles = np.linspace(-np.pi/2+.02, np.pi/2-.02, 5) if support else np.linspace(-1.2,1.2,5)
    seams = np.empty((4, len(angles), 4, 4))
    pads = np.empty_like(seams)
    parts = part_bounds()
    boxes = np.empty((4,len(angles),len(parts),2,3))
    names = []
    for i, port in enumerate(ports):
        name = ('pitch' if port.startswith('pitch') else 'yaw') + '_dock_mech_joint' + port[-1]
        names.append(name)
        actuator = next(a for a in meta['actuators'] if a['name'] == 'm0/' + name)
        for j, angle in enumerate(angles):
            q = pin.neutral(model); q[actuator['q']] = angle
            pin.framesForwardKinematics(model, data, q)
            seams[i, j] = data.oMf[meta['frames']['m0/' + PORTS[port]]].homogeneous
            pads[i, j] = data.oMf[meta['frames'][grasp_frame(0, port)]].homogeneous
            for k,(link,center,half) in enumerate(parts):
                frame=data.oMf[meta['frames']['m0/'+link]]
                boxes[i,j,k]=[frame.act(center),abs(frame.rotation)@half]
    moving=[next(k for k,(link,_,_) in enumerate(parts) if link==('pitch' if p.startswith('pitch') else 'yaw')+'_dock_mech'+p[-1]) for p in ports]
    return ports, angles, seams, pads, names, boxes, moving


def _frame(normal):
    x = np.asarray(normal); z = np.array([0., 0., 1.])
    z = z - x * (x @ z)
    if np.linalg.norm(z) < 1e-8:
        z = np.array([1., 0., 0.]); z -= x * (x @ z)
    z /= np.linalg.norm(z)
    return np.column_stack([x, np.cross(z, x), z])


def _clearance_cost(base, incoming, pad, shape, source, obstacles,support=False):
    """Bounding-box overlap guides partial growth; final exact checks remain."""
    _,angles,_,_,_,boxes,moving=part_frames(support); k=len(angles)
    fast=growth_clearance(base,boxes,incoming,pad,moving[pad],pin.Quaternion(source[3:]).matrix(),source[:3],shape.size,obstacles)
    if fast is not None:return fast
    parts=np.broadcast_to(boxes[incoming,np.tile(np.arange(k),k),None],(k*k,k,*boxes.shape[2:])).copy()
    parts[:,:,moving[pad]]=boxes[pad,:,moving[pad]][None]
    centers=np.einsum('bij,bcpj->bcpi',base[:,:3,:3],parts[:,:,:,0])+base[:,None,None,:3,3]
    halves=np.einsum('bij,bcpj->bcpi',abs(base[:,:3,:3]),parts[:,:,:,1])
    R=pin.Quaternion(source[3:]).matrix()
    local=(centers-source[:3])@R
    local_half=np.einsum('bij,bcpj->bcpi',abs(np.einsum('ij,bjk->bik',R.T,base[:,:3,:3])),parts[:,:,:,1])
    depth=shape.size/2+local_half-abs(local)
    margin=np.full(parts.shape[2],.03);margin[moving[pad]]=-.002
    cost=np.sum(np.maximum(0.,np.min(depth,axis=-1)+margin)**2,axis=-1)
    for center,half in obstacles:
        depth=halves+half+.035-abs(centers-center)
        cost+=np.sum(np.maximum(0.,np.min(depth,axis=-1))**2,axis=-1)
    return 4*cost


def _port_geometry(pose, out, shape, source, obstacles,support=False):
    """Geometry of one parent port, independent of the next contact target."""
    ports, angles, seams, pads, _, _, _ = part_frames(support)
    mate = np.diag([-1., -1., 1., 1.]); rows = []
    for incoming in range(4):
        if ports[out][:3] == ports[incoming][:3]:
            continue
        base = pose @ seams[out, :, None] @ mate @ np.linalg.inv(seams[incoming])[None, :]
        base = base.reshape(-1, 4, 4)
        for pad in range(4):
            if pad == incoming:
                continue
            frames = base[:, None] @ pads[pad][None]
            clearance = None if shape is None else _clearance_cost(base,incoming,pad,shape,source,obstacles,support)
            rows.append((incoming,pad,base,frames[..., :3, 3],frames[..., :3, 0],clearance))
    return rows


def _expansion_choices(poses, occupied, target, normal, shape, source, obstacles, geometry_cache=None,body_spacing=0.,motion_axis=None,contact_force=None,contact_stroke=None,*,support=False,lever_scale=.02,child_weight=0.):
    """All expansions from this exact partial geometry, before a random draw."""
    ports, angles, _, _, _, _, _ = part_frames(support); choices = []
    for parent in range(len(poses)):
        for out in range(4):
            if (parent, out) in occupied:
                continue
            # A growth call never changes an existing module pose or its
            # environment. Reuse each port's geometry across target roles
            # without repeatedly deserializing the same shared memo entry.
            key=(parent,out)
            geometry=None if geometry_cache is None else geometry_cache.get(key)
            if geometry is None:
                inputs=dict(operation='port_expansion_geometry',pose=poses[parent].tolist(),out=out,support=bool(support),
                            shape=None if shape is None else shape.payload(),
                            source=None if source is None else np.asarray(source).tolist(),
                            obstacles=np.asarray(obstacles).tolist())
                geometry=memoized(inputs,lambda:_port_geometry(poses[parent],out,shape,source,obstacles,support))
                if geometry_cache is not None:geometry_cache[key]=geometry
            for incoming,pad,base,position,direction,clearance in geometry:
                error = np.sum((position-target)**2, axis=-1) + .15**2*np.sum((direction-normal)**2, axis=-1)
                error += (.15 if contact_force is None else .03)**2*(1-base[:, 2, 2, None])**2
                if motion_axis is not None:
                    axis=base[:,:3,:3]@port_axes()[pad]
                    error+=.15**2*(1-abs(axis@motion_axis))[:,None]**2
                if contact_force is not None:
                    # Task force through the proposed branch, at both ends of
                    # the required foot stroke. This is a lever-arm preference,
                    # not a replacement for whole-body load acceptance.
                    stroke=np.zeros(3) if contact_stroke is None else np.asarray(contact_stroke)
                    lever_cost=np.zeros(error.shape)
                    for oi,(owner,port) in enumerate([(poses[parent][None],out),(base,incoming),(base,pad)]):
                        axes=owner[:,:3,:3]@port_axes()[port]
                        centers=owner[:,:3,3]+owner[:,:3,:3]@port_origins()[port]
                        # (r x force) . axis = r . (force x axis).
                        # Both stroke endpoints reduce to |a| + |b|.
                        moment_axes=np.cross(contact_force,axes)
                        nominal=np.einsum('bkj,bj->bk',position-centers[:,None],moment_axes)
                        if oi<2 and child_weight:
                            nominal+=np.einsum("bj,bj->b",np.cross(base[:,:3,3]-centers,[0.,0.,-child_weight]),axes)[:,None]
                        sweep=.5*abs(moment_axes@stroke)
                        lever_cost=np.maximum(lever_cost,(abs(nominal)+sweep[:,None])/1.3)
                    error+=lever_scale**2*lever_cost**2
                if body_spacing>0:
                    distance=np.linalg.norm(base[:,None,:3,3]-np.asarray(poses)[None,:,:3,3],axis=-1)
                    error+=4*np.sum(np.maximum(body_spacing-distance,0.)**2,axis=1)[:,None]
                if clearance is not None:error += clearance
                index = np.unravel_index(np.argmin(error), error.shape)
                seam_index, pad_index = index
                a, b = divmod(seam_index, len(angles))
                choices.append((float(error[index]), parent, out, incoming,
                                pad, a, b, pad_index, base[seam_index]))
    return choices


@lru_cache(maxsize=1)
def port_axes():
    """Single-part hinge axes, used before any connection graph exists."""
    model,meta=build_model();data=model.createData();q=pin.neutral(model)
    pin.forwardKinematics(model,data,q);result=[]
    for name in part_frames()[4]:
        a=next(a for a in meta['actuators'] if a['name']=='m0/'+name)
        result.append(data.oMi[a['joint']].rotation@pin.computeJointJacobian(model,data,q,a['joint'])[3:,a['v']])
    return np.array(result)


@lru_cache(maxsize=1)
def port_origins():
    model,meta=build_model();data=model.createData();pin.forwardKinematics(model,data,pin.neutral(model))
    return np.array([data.oMi[next(a['joint'] for a in meta['actuators'] if a['name']=='m0/'+name)].translation.copy()
                     for name in part_frames()[4]])


def _support_choices(poses,occupied,shape,source,obstacles):
    """Unused branches need clearance and load transmission, not another pad."""
    ports,angles,*_=part_frames(True);middle=len(angles)//2;choices=[]
    for parent,pose in enumerate(poses):
        for out in range(4):
            if (parent,out) in occupied:continue
            seen=set()
            for incoming,pad,base,_,_,clearance in _port_geometry(pose,out,shape,source,obstacles,True):
                if incoming in seen:continue
                seen.add(incoming)
                cost=.15**2*(1-base[:,2,2])**2
                if clearance is not None:cost+=clearance[:,middle]
                distance=np.linalg.norm(base[:,None,:3,3]-np.asarray(poses)[None,:,:3,3],axis=-1)
                cost+=4*np.sum(np.maximum(.25-distance,0.)**2,axis=1)
                for k,value in enumerate(cost):
                    a,b=divmod(k,len(angles))
                    choices.append((float(value),parent,out,incoming,pad,a,b,middle,base[k]))
    return choices


def _remaining_contact_cost(poses, occupied, points, normals, motion_axes):
    """One-to-one free-port assignment; a shared port cannot serve two feet.

    This is a growth preference, not a proof of simultaneous reachability.
    Existing occupied joints are fixed; each remaining free hinge is distinct.
    """
    from scipy.optimize import linear_sum_assignment
    if not len(points):
        return 0.
    pads = part_frames()[3]
    frames = [pose @ pads[port] for parent, pose in enumerate(poses)
              for port in range(4) if (parent, port) not in occupied]
    if len(frames) < len(points):
        return 100. * (len(points) - len(frames))
    frames = np.asarray(frames)
    cost = np.sum((frames[:, :, None, :3, 3] - points)**2, axis=-1)
    cost += .15**2 * np.sum((frames[:, :, None, :3, 0] - normals)**2, axis=-1)
    cost = cost.min(axis=1)
    if motion_axes is not None:
        axes = np.array([pose[:3, :3] @ port_axes()[port]
                         for parent, pose in enumerate(poses)
                         for port in range(4) if (parent, port) not in occupied])
        cost += .15**2 * (1 - abs(axes @ motion_axes.T))**2
    rows, cols = linear_sum_assignment(cost)
    return float(cost[rows, cols].sum())


def grow_to_contacts(modules, points, normals, rng, *, shape=None,source=None,obstacles=(),motion_axis=None,root_score=None,root_scores=None,support_score=None,root_shortlist=4,body_spacing=0.,multiple_contacts=False,contact_motion_axes=None,root_pose=None,coverage_weight=0.,capture_distance=.01,contact_forces=None,contact_strokes=None,independent_branches=False,support=False,lever_scale=.02,child_weight=0.):
    """Choose each seam from its progress towards uncovered contact frames.

    Returns a connected tree, contact roles and actual joint-grid seeds. No
    assembled graph, fitted posture, simulation or previous trial is scored.
    """
    ports, angles, seams, pads, names, _, _ = part_frames(support)
    if not 1 <= len(points) <= (2*modules+2 if multiple_contacts else modules):
        raise ValueError('guided growth needs one distinct module per contact role')
    points, normals = np.asarray(points), np.asarray(normals)
    for values in [contact_forces,contact_strokes]:
        if values is not None and (np.shape(values)!=points.shape or not np.isfinite(values).all()):
            raise ValueError('one finite force/stroke vector per contact required')
    if contact_motion_axes is not None:
        contact_motion_axes=np.asarray(contact_motion_axes,dtype=float)
        if contact_motion_axes.shape!=points.shape or not np.isfinite(contact_motion_axes).all() or np.any(np.linalg.norm(contact_motion_axes,axis=1)<1e-8):
            raise ValueError('one finite nonzero motion axis per contact required')
        contact_motion_axes=contact_motion_axes/np.linalg.norm(contact_motion_axes,axis=1)[:,None]
    first = int(rng.integers(4)); middle = len(angles)//2
    root = np.eye(4); root[:3, :3] = _frame(normals[0]) @ _frame(pads[first, middle, :3, 0]).T
    if motion_axis is not None and root_pose is None:
        # A turning task needs changing contact normals. Orient a candidate
        # Dock's hinge along the requested rotation, before any tree exists.
        # Prefer an upright module over using body roll for the same sweep.
        model,meta=build_model();q=pin.neutral(model);data=model.createData()
        pin.forwardKinematics(model,data,q)
        axis=np.asarray(motion_axis);axis=axis/np.linalg.norm(axis)
        n=normals[0];axis=axis-n*(axis@n);axis/=np.linalg.norm(axis)
        world=np.column_stack([n,np.cross(axis,n),axis]);choices=[]
        for port in range(4):
            a=next(a for a in meta['actuators'] if a['name']=='m0/'+names[port])
            joint_axis=data.oMi[a['joint']].rotation@pin.computeJointJacobian(model,data,q,a['joint'])[3:,a['v']]
            for angle_index in range(len(angles)):
                local=pads[port,angle_index,:3,0]
                for sign in [-1.,1.]:
                    z=sign*joint_axis;basis=np.column_stack([local,np.cross(z,local),z])
                    rotation=world@basis.T
                    choices.append((float((1-rotation[2,2])**2),port,angle_index,rotation,sign))
        # Upright is a preference, not a flight constraint: vectoring rotors
        # can support tilted modules. Keep those physically screened options.
        if root_score is not None:
            # A fixed random shortlist of single-part configurations, not
            # assembled structures. Task physics directs the very first
            # port/orientation before any connection graph is constructed.
            choices=[choices[i] for i in rng.choice(len(choices),size=min(root_shortlist,len(choices)),replace=False)]
            costs=[]
            for _,port,k,rotation,sign in choices:
                report=root_score(port,k,sign,rotation)
                costs.append(report['score'])
                if root_scores is not None:root_scores.append(dict(port=ports[port],angle=float(angles[k]),axis_sign=sign,**report))
            cost=np.asarray(costs)
        else:cost=np.array([c[0] for c in choices])
        weight=np.maximum(np.exp(-(cost-cost.min())/1.),1e-8)
        selected=int(rng.choice(len(choices),p=weight/weight.sum()))
        _,first,middle,root[:3,:3],_=choices[selected]
        if root_scores:root_scores[selected]['selected']=True
    root[:3, 3] = points[0] - root[:3, :3] @ pads[first, middle, :3, 3]
    poses = [root]; values = [{names[first]:float(angles[middle])}]; occupied = {(0, first)}
    surfaces = [(0, ports[first])]; edges = []; trace = []
    remaining = list(range(1, len(points)));geometry_cache={}
    if root_pose is not None:
        root=np.asarray(root_pose,dtype=float).copy()
        if root.shape!=(4,4) or not np.isfinite(root).all():raise ValueError('finite root transform required')
        poses=[root];values=[{}];occupied=set();surfaces=[];remaining=list(range(len(points)))
    def assign_existing(force=False):
        # A free moving Dock can supply another foot without adding a body.
        # Only that free joint changes; occupied seams and existing poses stay.
        while remaining:
            if not force and 4*len(poses)-len(occupied)<=1:return
            choices=[]
            for parent,pose in enumerate(poses):
                for port in range(4):
                    if (parent,port) in occupied:continue
                    frames=pose@pads[port]
                    for role in remaining:
                        cost=np.sum((frames[:,:3,3]-points[role])**2,axis=1)+.15**2*np.sum((frames[:,:3,0]-normals[role])**2,axis=1)
                        if contact_motion_axes is not None:
                            cost+=.15**2*(1-abs((pose[:3,:3]@port_axes()[port])@contact_motion_axes[role]))**2
                        k=int(np.argmin(cost));choices.append((float(cost[k]),parent,port,k,role))
            if not choices:raise ValueError('no free contact port')
            cost,parent,port,k,role=min(choices)
            if not force and cost>.04**2:return
            surfaces.append((parent,ports[port]));occupied.add((parent,port));remaining.remove(role)
            values[parent][names[port]]=float(angles[k]);geometry_cache.pop((parent,port),None)
            trace.append(dict(module=parent,target_role=role,residual_m2=cost,contact_assigned=True,
                              kind='existing_free_port',port=ports[port]))
    branch_role={}
    for child in range(1, modules):
        if multiple_contacts:assign_existing()
        choices = []
        support_reports=None
        if not remaining and support_score is not None:
            support_angles=part_frames(True)[1]
            expansions=_support_choices(poses,occupied,shape,source,obstacles)
            geometric=np.array([c[0] for c in expansions])
            weight=np.maximum(np.exp(-(geometric-geometric.min())/.1**2),1e-8)
            shortlist=rng.choice(len(expansions),min(2,len(expansions)),replace=False,p=weight/weight.sum())
            support_reports=[]
            for index in shortlist:
                choice=expansions[index];cost,parent,out,incoming,pad,a,b,c,pose=choice
                proposed=edges+[dict(a=parent,port_a=ports[out],b=child,port_b=ports[incoming])]
                angles0=[dict(v) for v in values]+[{names[incoming]:float(support_angles[b])}]
                angles0[parent][names[out]]=float(support_angles[a])
                report=support_score(proposed,surfaces,root,angles0)
                support_reports.append(report)
                # Finite preferences retain every shortlisted legal option.
                score=report['score']+cost/.1**2
                choices.append((score,*choice[1:],len(points)-1))
        # Contact-force selection fixes a set, not a required connection order.
        # Use actual seam reachability to assign the next uncovered role.
        for role in ([] if support_reports is not None else remaining or [len(points)-1]):
            target, normal = points[role], normals[role]
            inputs=dict(operation='expansion',poses=np.asarray(poses).tolist(),occupied=sorted(occupied),support=bool(support),lever_scale=lever_scale,child_weight=child_weight,
                        target=target.tolist(),normal=normal.tolist(),
                        shape=None if shape is None else shape.payload(),
                        source=None if source is None else np.asarray(source).tolist(),
                        obstacles=np.asarray(obstacles).tolist())
            if body_spacing:inputs['body_spacing']=body_spacing
            axis=None if contact_motion_axes is None else np.asarray(contact_motion_axes[role])
            if axis is not None:inputs['motion_axis']=axis.tolist()
            force=None if contact_forces is None else np.asarray(contact_forces[role])
            stroke=None if contact_strokes is None else np.asarray(contact_strokes[role])
            if force is not None:inputs['contact_force']=force.tolist()
            if stroke is not None:inputs['contact_stroke']=stroke.tolist()
            expansions = memoized(inputs,lambda:_expansion_choices(poses,occupied,target,normal,shape,source,obstacles,geometry_cache,body_spacing,axis,force,stroke,support=support,lever_scale=lever_scale,child_weight=child_weight))
            choices.extend((*choice,role) for choice in expansions)
        if independent_branches and remaining and root_pose is not None:
            # Preserve separate force paths from the root while growing feet.
            # Ports, branch lengths and joint angles still follow task loads.
            choices=[c for c in choices if c[1]==0 or branch_role.get(c[1])==c[-1]]
        if not choices:
            raise ValueError('no unoccupied seam for contact-directed growth')
        if coverage_weight and remaining:
            ranked=[]
            for choice in choices:
                cost,parent,out,incoming,pad,a,b,c,pose,role=choice
                used=occupied|{(parent,out),(child,incoming)}
                assigned=(modules-child<=len(remaining) or cost<capture_distance**2)
                roles=[r for r in remaining if not (assigned and r==role)]
                if assigned:used.add((child,pad))
                coverage=_remaining_contact_cost(poses+[pose],used,points[roles],normals[roles],
                    None if contact_motion_axes is None else contact_motion_axes[roles])
                ranked.append((cost+coverage_weight*coverage,*choice[1:]))
            choices=ranked
        order=np.argsort([c[0] for c in choices],kind='stable');choices=[choices[i] for i in order]
        if support_reports is not None:support_reports=[support_reports[i] for i in order]
        # Randomness only diversifies equally useful grammatical expansions.
        costs = np.array([c[0] for c in choices])
        weight = np.maximum(np.exp(-(costs-costs[0])/(.2 if support_reports is not None else .02**2)),1e-8 if support_reports is not None else 0.)
        selected=int(rng.choice(len(choices), p=weight/weight.sum()));chosen=choices[selected]
        cost, parent, out, incoming, pad, a, b, c, pose, role = chosen
        branch_role[child]=role
        edges.append(dict(a=parent, port_a=ports[out], b=child, port_b=ports[incoming]))
        occupied.update([(parent, out), (child, incoming)])
        joint_angles=angles if support_reports is None else support_angles
        values[parent][names[out]] = float(joint_angles[a])
        values.append({names[incoming]:float(joint_angles[b])})
        poses.append(pose)
        # Reserve enough remaining modules for all still-unassigned roles.
        # Capture uses the selected contact's geometric residual, independent
        # of other feet's coverage cost, which is only a branching preference.
        contact_frame=pose@pads[pad,c]
        capture_cost=np.sum((contact_frame[:3,3]-points[role])**2)+.15**2*np.sum((contact_frame[:3,0]-normals[role])**2)
        assigned = bool(remaining) and (modules-child <= len(remaining) or (capture_cost if coverage_weight else cost)<capture_distance**2)
        if assigned:
            surfaces.append((child, ports[pad])); occupied.add((child, pad))
            values[child][names[pad]] = float(angles[c])
            remaining.remove(role)
        trace.append(dict(module=child, target_role=role, residual_m2=cost,
                          contact_assigned=assigned,
                          parent=parent, port_a=ports[out], port_b=ports[incoming]))
        if support_reports is not None:
            trace[-1].update(criterion='task_load_support',support_scores=support_reports,selected=selected)
            trace[-1]['support_score']=trace[-1].pop('residual_m2')
    if multiple_contacts:assign_existing(force=True)
    return edges, surfaces, root, values, trace


def fit_generated(modules, shape, source, obstacles, points, normals, rng, *, fit_pose=True, opposed=True,motion_axis=None,root_score=None,support_score=None,preserve_box_faces=False):
    """Materialize a task-generated tree and use the original continuous fit."""
    started = time.perf_counter()
    R = pin.Quaternion(source[3:]).matrix()
    world = np.asarray(points) @ R.T + source[:3]
    wn = np.asarray(normals) @ R.T
    root_scores=[]
    edges, surfaces, root, values, trace = grow_to_contacts(modules, world+.001*wn, wn, rng,
                                                         shape=shape,source=source,obstacles=obstacles,motion_axis=motion_axis,
                                                         root_score=root_score,root_scores=root_scores,support_score=support_score)
    order = [0]+[item['target_role'] for item in trace if item['contact_assigned']]
    world, wn = world[order], wn[order]
    points, normals = np.asarray(points)[order], np.asarray(normals)[order]
    generated=dict(method='contact_directed_growth', expansion_trace=trace,
                   target_points=np.asarray(points).tolist(),target_normals=np.asarray(normals).tolist())
    if motion_axis is not None:generated['contact_normal_rotation_axis']=np.asarray(motion_axis).tolist()
    if root_scores:generated['root_task_scores']=root_scores
    if not fit_pose:
        return dict(group='task_growth',modules=modules,edges=edges,surfaces=surfaces,
                    wall_s=time.perf_counter()-started,generation=generated)
    model, meta = build_model(modules, edges, shape.payload())
    q = pin.neutral(model); q[meta['object']['q']:] = source
    q[:3] = root[:3, 3]; q[3:7] = pin.Quaternion(root[:3, :3]).coeffs()
    for actuator in meta['actuators']:
        module, name = actuator['name'].split('/')
        q[actuator['q']] = values[int(module[1:])].get(name, 0.)
    inputs=dict(operation='fit_task_growth',modules=modules,edges=edges,surfaces=surfaces,
                shape=shape.payload(),q=q.tolist(),points=(world+.001*wn).tolist(),normals=wn.tolist(),
                obstacles=np.asarray(obstacles).tolist(),opposed=opposed,preserve_box_faces=preserve_box_faces)
    closed, fit = memoized(inputs,lambda:fit_contacts(model, meta, q, surfaces, world+.001*wn, wn,
        object_shape=shape, floor_clearance=.03, base_rotation=True, obstacles=obstacles, max_evaluations=45,
        opposed_friction=4.5 if len(points)==2 and opposed else None,avoid_object=True,preserve_box_faces=preserve_box_faces))
    if len(points)==2:fit['initial_object_points']=np.asarray(points).tolist()
    return dict(group='task_growth', modules=modules, edges=edges, surfaces=surfaces,
                q=closed.tolist(), fit=fit, wall_s=time.perf_counter()-started,
                generation=generated)
