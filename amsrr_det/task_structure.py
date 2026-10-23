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


@lru_cache(maxsize=1)
def part_frames():
    model, meta = build_model()
    data = model.createData()
    ports = tuple(PORTS)
    angles = np.linspace(-1.2, 1.2, 5)
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


def _clearance_cost(base, incoming, pad, shape, source, obstacles):
    """Bounding-box overlap guides partial growth; final exact checks remain."""
    _,angles,_,_,_,boxes,moving=part_frames(); k=len(angles)
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


def _port_geometry(pose, out, shape, source, obstacles):
    """Geometry of one parent port, independent of the next contact target."""
    ports, angles, seams, pads, _, _, _ = part_frames()
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
            clearance = None if shape is None else _clearance_cost(base,incoming,pad,shape,source,obstacles)
            rows.append((incoming,pad,base,frames[..., :3, 3],frames[..., :3, 0],clearance))
    return rows


def _expansion_choices(poses, occupied, target, normal, shape, source, obstacles):
    """All expansions from this exact partial geometry, before a random draw."""
    ports, angles, _, _, _, _, _ = part_frames(); choices = []
    for parent in range(len(poses)):
        for out in range(4):
            if (parent, out) in occupied:
                continue
            inputs=dict(operation='port_expansion_geometry',pose=poses[parent].tolist(),out=out,
                        shape=None if shape is None else shape.payload(),
                        source=None if source is None else np.asarray(source).tolist(),
                        obstacles=np.asarray(obstacles).tolist())
            geometry=memoized(inputs,lambda:_port_geometry(poses[parent],out,shape,source,obstacles))
            for incoming,pad,base,position,direction,clearance in geometry:
                error = np.sum((position-target)**2, axis=-1) + .15**2*np.sum((direction-normal)**2, axis=-1)
                error += .15**2*(1-base[:, 2, 2, None])**2
                if clearance is not None:error += clearance
                index = np.unravel_index(np.argmin(error), error.shape)
                seam_index, pad_index = index
                a, b = divmod(seam_index, len(angles))
                choices.append((float(error[index]), parent, out, incoming,
                                pad, a, b, pad_index, base[seam_index]))
    return choices


def grow_to_contacts(modules, points, normals, rng, *, shape=None,source=None,obstacles=()):
    """Choose each seam from its progress towards uncovered contact frames.

    Returns a connected tree, contact roles and actual joint-grid seeds. No
    assembled graph, fitted posture, simulation or previous trial is scored.
    """
    ports, angles, seams, pads, names, _, _ = part_frames()
    if not 1 <= len(points) <= modules:
        raise ValueError('guided growth needs one distinct module per contact role')
    points, normals = np.asarray(points), np.asarray(normals)
    first = int(rng.integers(4)); middle = len(angles)//2
    root = np.eye(4); root[:3, :3] = _frame(normals[0]) @ _frame(pads[first, middle, :3, 0]).T
    root[:3, 3] = points[0] - root[:3, :3] @ pads[first, middle, :3, 3]
    poses = [root]; values = [{}]; occupied = {(0, first)}
    surfaces = [(0, ports[first])]; edges = []; trace = []
    for child in range(1, modules):
        role = min(len(surfaces), len(points)-1)
        target, normal = points[role], normals[role]
        inputs=dict(operation='expansion',poses=np.asarray(poses).tolist(),occupied=sorted(occupied),
                    target=target.tolist(),normal=normal.tolist(),
                    shape=None if shape is None else shape.payload(),
                    source=None if source is None else np.asarray(source).tolist(),
                    obstacles=np.asarray(obstacles).tolist())
        choices = memoized(inputs,lambda:_expansion_choices(poses,occupied,target,normal,shape,source,obstacles))
        if not choices:
            raise ValueError('no unoccupied seam for contact-directed growth')
        choices.sort(key=lambda row: row[0])
        # Randomness only diversifies equally useful grammatical expansions.
        costs = np.array([c[0] for c in choices])
        weight = np.exp(-(costs-costs[0])/.02**2)
        chosen = choices[int(rng.choice(len(choices), p=weight/weight.sum()))]
        cost, parent, out, incoming, pad, a, b, c, pose = chosen
        edges.append(dict(a=parent, port_a=ports[out], b=child, port_b=ports[incoming]))
        occupied.update([(parent, out), (child, incoming)])
        values[parent][names[out]] = float(angles[a])
        values.append({names[incoming]:float(angles[b])})
        poses.append(pose)
        # Reserve enough remaining modules for all still-unassigned roles.
        if modules-child <= len(points)-len(surfaces) or (len(surfaces)<len(points) and cost<.01**2):
            surfaces.append((child, ports[pad])); occupied.add((child, pad))
            values[child][names[pad]] = float(angles[c])
        trace.append(dict(module=child, target_role=role, residual_m2=cost,
                          parent=parent, port_a=ports[out], port_b=ports[incoming]))
    return edges, surfaces, root, values, trace


def fit_generated(modules, shape, source, obstacles, points, normals, rng, *, fit_pose=True):
    """Materialize a task-generated tree and use the original continuous fit."""
    started = time.perf_counter()
    R = pin.Quaternion(source[3:]).matrix()
    world = np.asarray(points) @ R.T + source[:3]
    wn = np.asarray(normals) @ R.T
    edges, surfaces, root, values, trace = grow_to_contacts(modules, world+.001*wn, wn, rng,
                                                         shape=shape,source=source,obstacles=obstacles)
    generated=dict(method='contact_directed_growth', expansion_trace=trace,
                   target_points=np.asarray(points).tolist(),target_normals=np.asarray(normals).tolist())
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
                obstacles=np.asarray(obstacles).tolist())
    closed, fit = memoized(inputs,lambda:fit_contacts(model, meta, q, surfaces, world+.001*wn, wn,
        object_shape=shape, floor_clearance=.03, base_rotation=True, obstacles=obstacles, max_evaluations=45,
        opposed_friction=4.5 if len(points)==2 else None))
    if len(points)==2:fit['initial_object_points']=np.asarray(points).tolist()
    return dict(group='task_growth', modules=modules, edges=edges, surfaces=surfaces,
                q=closed.tolist(), fit=fit, wall_s=time.perf_counter()-started,
                generation=generated)
