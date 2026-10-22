"""Finite outward reference path captured once, with no object-velocity feedback.

This is a kinematic target generator. Actual separation, loads and collisions
are measured separately; a solved target is not a contact-release guarantee.
"""
import numpy as np
import pinocchio as pin
from .grasp_geometry import fit_contacts
from .numerics import advance_world_orientation


def rest_to_rest_durations(model,meta,path,distances,normal_speed=.006):
    """Conservative quintic durations; preserve the geometric path exactly.

    Limits are the existing reference speeds/accelerations, not actuator
    capability guarantees. Stopping at each corner avoids a velocity jump.
    """
    durations=[];peak_speed=1.875;peak_acceleration=10/np.sqrt(3.)
    speeds=np.broadcast_to(normal_speed,(len(path)-1,))
    if not np.all(np.isfinite(speeds)&(speeds>0)):raise ValueError('invalid normal path speed')
    for first,last,distance,speed in zip(path[:-1],path[1:],np.diff(distances),speeds):
        delta=pin.difference(model,first,last)
        bounds=[peak_speed*distance/speed]
        for base in meta['bases']:
            j=base['v'];linear,angular=delta[j:j+3],delta[j+3:j+6]
            bounds.extend([peak_speed*np.linalg.norm(linear)/.06,peak_speed*np.linalg.norm(angular)/.15,
                np.sqrt((peak_acceleration*np.linalg.norm(linear)+peak_speed**2*np.linalg.norm(np.cross(angular,linear)))/.1),
                np.sqrt(peak_acceleration*np.linalg.norm(angular)/.3)])
        for actuator in meta['actuators']:
            amount=abs(delta[actuator['v']]);speed=.2 if actuator['role']=='dock' else .3
            bounds.extend([peak_speed*amount/speed,np.sqrt(peak_acceleration*amount/.5)])
        durations.append(max(.01,*bounds))
    return np.asarray(durations)


def object_avoidance_hull(shape):
    """Conservative object geometry for repairing a detected part collision."""
    import coal
    if shape.vertices is not None:
        vertices=coal.StdVec_Vec3s();faces=coal.StdVec_Triangle()
        for v in shape.vertices:vertices.append(v)
        for f in shape.faces:faces.append(coal.Triangle(*map(int,f)))
        mesh=coal.BVHModelOBBRSS();mesh.beginModel(len(faces),len(vertices));mesh.addSubModel(vertices,faces);mesh.endModel()
        mesh.buildConvexHull(True,'Qt');return mesh.convex
    if shape.family in ['box','elongated_box']:return coal.Box(*map(float,shape.size))
    if shape.family=='cylinder':return coal.Cylinder(float(shape.size[0]/2),float(shape.size[2]))
    if shape.family=='ellipsoid':return coal.Ellipsoid(*map(float,shape.size/2))
    raise ValueError('unknown collision-repair object shape')


def task_clearance_gap(shape,captured,destination, *, separation=.02,position_tolerance=.03,angle_tolerance=.2):
    """Frozen-object distance that covers the declared final pose set.

    Triangle inequalities bound translation and rotation from the captured
    pose. The radius bounds every object material point about its pose origin.
    This is a geometric target, not a bound on robot tracking/contact force.
    """
    radius=float(np.linalg.norm(shape.vertices,axis=1).max() if shape.vertices is not None else np.linalg.norm(shape.size/2))
    dp=position_tolerance+np.linalg.norm(captured[:3]-destination[:3])
    angle=np.linalg.norm(pin.log3(pin.Quaternion(captured[3:]).matrix()@pin.Quaternion(destination[3:]).matrix().T))
    return float(separation+dp+2*radius*np.sin(min(np.pi,angle_tolerance+angle)/2))


class FixedRetreat:
    def __init__(self,grasp,q,opening_order='simultaneous', *, opening_gap_m=.002,final_gap_m=.03,
                 obstacles=(),maximum_material_step=None,prepared=None,avoid_pairs=()):
        if opening_order not in ['simultaneous','sides_first']:raise ValueError('unknown opening order')
        if not 0<opening_gap_m<=final_gap_m or not np.isfinite(final_gap_m):
            raise ValueError('retreat gaps must be finite, positive and ordered')
        if maximum_material_step is not None and not 0<maximum_material_step<np.inf:
            raise ValueError('invalid material path spacing')
        self.grasp=grasp;self.model=grasp.model;self.meta=grasp.meta
        self.reference=grasp.reference.copy();self.distance=0.;self.speed=0.;self.elapsed=0.
        self.initial_velocity=grasp.last_velocity.copy()
        acceleration=np.r_[np.full(3,.1),np.full(3,.3),np.full(len(grasp.docks),.5)]
        self.bridge_time=max(.25,8*float(np.max(abs(self.initial_velocity)/acceleration)))
        _,_,_,self.initial_compression=grasp.geometry(q)
        obj=self.meta['object'];self.reference[obj['q']:obj['q']+7]=q[obj['q']:obj['q']+7]
        data=self.model.createData();pin.framesForwardKinematics(self.model,data,self.reference)
        frames=[frame for frame,_ in grasp.samples]
        centers=np.array([data.oMf[f].translation for f in frames])
        normals=np.array([data.oMf[f].rotation[:,0] for f in frames])
        # Account for the difference between the loaded reference and the
        # observed material. A 2 mm change from a penetrating reference need
        # not put that reference outside the captured object at all.
        from .contact_geometry import close_material
        def gaps_at(configuration):
            pin.framesForwardKinematics(self.model,data,configuration)
            if grasp.shape is None:return np.zeros(len(frames)),-normals
            obj_pose=data.oMf[obj['frame']];gaps=[];outward=[]
            for frame,triangles in zip(frames,grasp.triangles):
                _,_,gap,direction=close_material(grasp.shape,obj_pose.inverse()*data.oMf[frame],triangles)
                k=int(np.argmin(gap));gaps.append(gap[k]);outward.append(obj_pose.rotation@direction[k])
            return np.array(gaps),np.array(outward)
        initial_gaps,outward=gaps_at(self.reference)
        self.initial_reference_gaps=initial_gaps.copy()
        if prepared is not None:
            self.path=prepared['path'];self.distances=prepared['distances']
            self.segment_durations=prepared.get('segment_durations')
            if self.segment_durations is not None and np.max(abs(self.initial_velocity))>1e-9:
                raise ValueError('rest_to_rest_opening_requires_stationary_reference')
            # A worker plans from the held reference. Never silently splice a
            # path planned from a different robot configuration into control.
            delta=pin.difference(self.model,self.reference,self.path[0])
            if np.max(abs(np.r_[delta[:6],delta[grasp.vi]]))>1e-10:
                raise ValueError('opening_reference_changed_during_planning')
            self.opening_distance=self.distances[-1]
            self.opening_duration=max(2.5,1.875*self.opening_distance/.001)
            return
        self.distances=[0.];self.path=[self.reference.copy()];previous_centers=centers
        near=np.full(len(frames),opening_gap_m);gap_goals=[]
        upper=outward[:,2]>.5
        if opening_order=='sides_first' and np.any(upper) and np.any(~upper):
            hold=near.copy();hold[upper]=np.minimum(near[upper],-self.initial_compression[upper])
            gap_goals.append(hold)
        gap_goals.append(near);opening_index=len(gap_goals)
        if final_gap_m>opening_gap_m:gap_goals.append(np.full(len(frames),final_gap_m))
        self.goal_gaps=gap_goals
        expanded=[];previous_gap=initial_gaps.copy();expanded_opening_index=0
        for index,gap_goal in enumerate(gap_goals):
            count=1 if maximum_material_step is None else max(1,int(np.ceil(np.max(abs(gap_goal-previous_gap))/maximum_material_step)))
            expanded.extend((1-u)*previous_gap+u*gap_goal for u in np.linspace(0,1,count+1)[1:])
            previous_gap=gap_goal
            if index+1==opening_index:expanded_opening_index=len(expanded)
        previous_gaps=initial_gaps.copy()
        avoid_shapes=({'object':(pin.SE3.Identity(),object_avoidance_hull(grasp.shape))}
                      if any('object' in pair for pair in avoid_pairs) else None)
        if any(name in ['support','destination_support'] for pair in avoid_pairs for name in pair):
            import coal
            avoid_shapes=dict(avoid_shapes or {})
            for name,(center,half) in zip(['support','destination_support'],obstacles):
                avoid_shapes[name]=(pin.SE3(np.eye(3),np.asarray(center)),coal.Box(*map(float,2*np.asarray(half))))
        for waypoint,gap_goal in enumerate(expanded):
            goals=centers+np.maximum(gap_goal-initial_gaps,0.)[:,None]*outward
            # A finite refinement budget is numerical, not a material-gap
            # tolerance. Four-pad paths can need more than six corrections.
            for _ in range(24):
                target,fit=fit_contacts(self.model,self.meta,self.path[-1],grasp.surfaces,
                    goals,normals,base_rotation=True,max_evaluations=100,
                    floor_clearance=.03 if obstacles else None,obstacles=obstacles,avoid_pairs=avoid_pairs,avoid_shapes=avoid_shapes)
                # Frame sites/normals are numerical guidance during release;
                # the material distance below is the actual endpoint test.
                if grasp.shape is None and (fit['position_error_m']>1e-5 or fit['normal_error']>1e-3):
                    error=ValueError('finite_retreat_kinematics_not_found')
                    error.evidence=dict(fit=fit,goal_gap_m=gap_goal.tolist(),completed_waypoints=len(self.path)-1)
                    raise error
                gaps,directions=gaps_at(target)
                if grasp.shape is None or min(gaps-gap_goal)>=-1e-5:break
                goals+=np.maximum(gap_goal-gaps,0.)[:,None]*directions
            else:
                # Intermediate spacing is a numerical construction, not a
                # required material distance. Keep an outward intermediate
                # pose even if obstacle avoidance shifts its nominal sites.
                if waypoint==len(expanded)-1 or np.min(gaps-previous_gaps)<-1e-7:
                    error=ValueError('finite_retreat_material_gap_not_found')
                    error.evidence=dict(fit=fit,goal_gap_m=gap_goal.tolist(),actual_gap_m=gaps.tolist(),completed_waypoints=len(self.path)-1)
                    raise error
            actual_centers=np.array([data.oMf[f].translation for f in frames])
            self.distances.append(self.distances[-1]+float(np.linalg.norm(actual_centers-previous_centers,axis=1).max()))
            previous_centers=actual_centers;previous_gaps=gaps;self.path.append(target)
        self.distances=np.array(self.distances)
        self.opening_distance=self.distances[expanded_opening_index]
        self.opening_duration=max(2.5,1.875*self.opening_distance/.001)

    def at(self,distance):
        index=min(len(self.path)-2,int(np.searchsorted(self.distances,distance,side='right')-1))
        index=max(0,index)
        fraction=np.clip((distance-self.distances[index])/(self.distances[index+1]-self.distances[index]),0.,1.)
        return pin.interpolate(self.model,self.path[index],self.path[index+1],float(fraction))

    def update(self,q,v,object_target, *, preload,normal_speed=.001,**kwargs):
        g=self.grasp
        if getattr(self,'segment_durations',None) is not None:
            from .numerics import quintic_reference
            ends=np.r_[0.,np.cumsum(self.segment_durations)]
            index=min(len(self.path)-2,int(np.searchsorted(ends,self.elapsed,side='right')-1))
            local=self.elapsed-ends[index];duration=self.segment_durations[index]
            target,tv,_,fraction=quintic_reference(self.model,self.meta,self.path[index],self.path[index+1],local,duration)
            self.distance=float((1-fraction)*self.distances[index]+fraction*self.distances[index+1])
            u=np.clip(local/duration,0.,1.)
            self.speed=float((self.distances[index+1]-self.distances[index])*30*u*u*(1-u)**2/duration)
            self.elapsed+=g.dt;self.reference=target.copy()
            g.reference=target.copy();g.last_velocity=np.r_[tv[:6],tv[g.vi]]
            _,_,_,compression=g.geometry(q)
            g.last_diagnostics=dict(compression=compression.tolist(),preload=np.asarray(preload).tolist(),
                fixed_retreat_distance_m=self.distance,object_velocity_drives_reference=False,
                fixed_retreat_speed_m_s=self.speed,
                reference_tracking_error_rad=float(abs(target[g.qi]-q[g.qi]).max()),
                rest_to_rest_segment=index)
            return target,tv
        # This amount is set by the phase's bounded opening schedule. Geometry
        # observations below are diagnostics, never a moving retreat target.
        if normal_speed>.001:
            requested=self.distances[-1]
        else:
            progress=float(np.clip(np.max((g.preload-np.asarray(preload))/(g.preload+.002)),0.,1.))
            requested=progress*self.opening_distance
        requested=np.clip(requested,0.,self.distances[-1])
        acceleration=.002
        desired_speed=min(normal_speed,np.sqrt(2*acceleration*max(0.,requested-self.distance)))
        self.speed+=float(np.clip(desired_speed-self.speed,-acceleration*g.dt,acceleration*g.dt))
        distance=min(self.distances[-1],self.distance+self.speed*g.dt)
        self.elapsed+=g.dt
        def moving_target(amount):
            target=self.at(amount)
            # Compact initial-velocity bridge: value 0 at both ends, initial
            # derivative 1 and final derivative 0. Never reset to physical q.
            u=min(1.,self.elapsed/self.bridge_time)
            blend=self.bridge_time*u*(1-u)**2
            target[:3]+=blend*self.initial_velocity[:3]
            target[3:7]=advance_world_orientation(target[3:7],blend*self.initial_velocity[3:6])
            target[g.qi]+=blend*self.initial_velocity[6:]
            return target
        target=moving_target(distance)
        delta=pin.difference(self.model,self.reference,target)
        rotation=pin.Quaternion(self.reference[3:7]).matrix()
        velocity=np.r_[rotation@delta[:3],rotation@delta[3:6],delta[g.vi]]/g.dt
        gimbals=np.array([a['v'] for a in self.meta['actuators'] if a['role']=='vectoring'])
        ratio=max(float(np.max(abs(velocity)/g.scale)),float(np.max(abs(delta[gimbals])))/(.3*g.dt))
        if ratio>1:
            distance=self.distance+(distance-self.distance)/ratio
            target=moving_target(distance);delta=pin.difference(self.model,self.reference,target)
            velocity=np.r_[rotation@delta[:3],rotation@delta[3:6],delta[g.vi]]/g.dt
        self.distance=float(distance);self.reference=target.copy()
        g.reference=target.copy();g.last_velocity=velocity.copy()
        _,_,_,compression=g.geometry(q)
        g.last_diagnostics=dict(compression=compression.tolist(),preload=np.asarray(preload).tolist(),
            fixed_retreat_distance_m=self.distance,object_velocity_drives_reference=False,
            fixed_retreat_speed_m_s=self.speed,
            reference_tracking_error_rad=float(abs(target[g.qi]-q[g.qi]).max()))
        tv=np.zeros(self.model.nv);tv[:6]=velocity[:6];tv[g.vi]=velocity[6:]
        return target,tv
