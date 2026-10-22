"""Continuous unloading followed by task-derived, empty-flight withdrawal.

Only pose/velocity and known motor commands enter this controller. The simulator
separately verifies load transfer and task success; load=0 is not force evidence.
"""
import time
import numpy as np
import pinocchio as pin
from .equilibrium import static_balance
from .free_flight_trim import refine_support_path,scheduled_support,free_flight_trim
from .retreat import FixedRetreat,task_clearance_gap,rest_to_rest_durations
from .numerics import advance_world_orientation


def smooth(t):
    t=np.clip(t,0.,1.);return t*t*t*(10+t*(-15+6*t))


def warm_planner():
    return True


def check_opening_clearance(meta,shape,surfaces,path,obstacles, *, prepared_plant=None):
    """Bound motion between geometry queries along every interpolated segment."""
    import xml.etree.ElementTree as ET
    import mujoco
    from .mujoco_model import make_plant,numbers
    from .clearance import MotionBoundedClearance,PART_CLEARANCE,ENVIRONMENT_CLEARANCE
    if prepared_plant is None:
        plant=make_plant(meta['modules'],meta['edges'],payload=shape.payload(),mesh_cache=meta.get('material_mesh_cache'),floor=True)
        doc=ET.fromstring(plant.xml)
        for name,(center,half) in zip(['support','destination_support'],obstacles):
            ET.SubElement(doc.find('worldbody'),'geom',name=name,type='box',pos=numbers(center),size=numbers(half))
        plant.mj=mujoco.MjModel.from_xml_string(ET.tostring(doc,encoding='unicode'))
    else:
        # Reuse this candidate's full compiled geometry, never its pruned
        # contact-dynamics model. The checker still owns a separate copy.
        plant=prepared_plant
        for name,(center,half) in zip(['support','destination_support'],obstacles):
            geom=plant.mj.geom(name)
            if not np.array_equal(geom.pos,center) or not np.array_equal(geom.size,half):
                raise ValueError('prepared clearance support mismatch')
    # Adding static geometry does not change the joint order.
    links=[f'm{m}/'+('yaw_dock_mech' if port.startswith('yaw') else 'pitch_dock_mech')+port[-1] for m,port in surfaces]
    check=MotionBoundedClearance(plant.mj,links);motion=check.joint_motion
    if motion is None:raise ValueError('opening_motion_bound_unavailable')
    free={int(plant.mj.jnt_qposadr[mjid]):plant.pin.joints[jid] for jid,mjid in plant.joints if plant.pin.joints[jid].nq==7}
    body_rows=np.zeros(plant.mj.nbody,dtype=int);body_rows[check.bodies]=np.arange(len(check.bodies))
    roots=np.full(len(check.bodies),-1,dtype=int)
    for k,(rows,_) in motion.roots.items():roots[rows]=k
    from .interval_bounds import IntervalPairBounds
    try:pair_bounds=IntervalPairBounds(check,body_rows,roots)
    except ValueError:pair_bounds=None
    checked=0
    for segment,(first,last) in enumerate(zip(path[:-1],path[1:])):
        qfirst,qlast=plant.configuration(first),plant.configuration(last)
        hinge_change=abs(qlast[motion.qi]-qfirst[motion.qi]);hinge_bound=hinge_change@motion.levers.T
        tangent=pin.difference(plant.pin,first,last)
        pending=[(0.,1.,0)]
        while pending:
            low,high,depth=pending.pop();fraction=(low+high)/2;half=(high-low)/2
            q=pin.interpolate(plant.pin,first,last,float(fraction))
            qpos=plant.configuration(q);queries=check.queries
            bad,part,environment=check.check(qpos);checked+=1
            reused=check.queries==queries
            if reused:
                # A conservative reused lower bound can lie at the threshold.
                # Refresh before asking it to certify a nonzero path interval.
                check.anchor=None;bad,part,environment=check.check(qpos)
            if bad:
                pair=bad[0];distance=min(float(c.dist)for c in check.data.contact if tuple(sorted((check.names[c.geom1],check.names[c.geom2])))==pair)
                error=ValueError('empty_opening_sampled_clearance_violation')
                required=ENVIRONMENT_CLEARANCE if any(name in ['floor','support','destination_support']for name in pair)else PART_CLEARANCE
                error.evidence=dict(segment=segment,fraction=float(fraction),parts=list(pair),distance_m=distance,required_m=required,configuration=q.tolist(),queries=check.queries,intervals=checked)
                raise error
            # Along pin.interpolate, hinge angles are linear and each free
            # pose follows a constant body twist. Speed times interval bounds
            # origin translation even when the SE(3) path is curved.
            root_movement=np.zeros(len(check.bodies))
            for k,(rows,radius) in motion.roots.items():
                twist=tangent[free[k].idx_v:free[k].idx_v+6]
                root_movement[rows]=half*np.linalg.norm(twist[:3])+2*radius*np.sin(min(np.pi,half*np.linalg.norm(twist[3:]))/2)
            movement=half*hinge_bound+root_movement
            # For observed near pairs, common ancestor motion is a rigid
            # transformation of both parts and cannot change their distance.
            intended,limits=check.classifiers[True](check.data)
            pairs=np.column_stack([check.data.contact.geom1,check.data.contact.geom2])[~intended]
            rows=body_rows[plant.mj.geom_bodyid[pairs]]
            a,b=rows[:,0],rows[:,1];la,lb=motion.levers[a],motion.levers[b]
            pair_movement=half*((la+lb)*np.logical_xor(la>0,lb>0))@hinge_change
            pair_movement+=np.where(roots[a]==roots[b],0.,root_movement[a]+root_movement[b])
            # Absent pairs can use a tighter conservative certificate: common
            # ancestor motion cancels, enclosing boxes bound separation, and
            # exact convex support proves any remaining positive gap.
            absent_safe=(2*movement.max(initial=0.)<=check.halo+1e-6 if pair_bounds is None else
                         pair_bounds.certifies(check.data,half,hinge_change,root_movement))
            if (absent_safe
                    and np.all(pair_movement<=np.asarray(check.data.contact.dist)[~intended]-limits[~intended]+1e-6)):
                continue
            if depth>=20 or checked>=32768:
                error=ValueError('opening_clearance_not_certified')
                error.evidence=dict(segment=segment,fraction=float(fraction),depth=depth,checks=checked,part_gap_m=part,environment_gap_m=environment,interval_movement_bound_m=float(2*movement.max(initial=0.)))
                raise error
            pending.extend([(fraction,high,depth+1),(low,fraction,depth+1)])
    return dict(queries=check.queries,reuses=check.reused,intervals=checked)


def plan_opening(model,meta,shape,q,reference,surfaces,points,normals,obstacles,destination, *, withdrawal='auto'):
    """Pure geometric/static planning, suitable for a separate CPU process."""
    start=time.perf_counter()
    if withdrawal=='auto':
        rejected=[]
        for mode in ['lift','spread']:
            try:
                result=plan_opening(model,meta,shape,q,reference,surfaces,points,normals,obstacles,destination,withdrawal=mode)
                result.update(planning_s=time.perf_counter()-start,route_rejections=rejected)
                return result
            except ValueError as error:
                rejected.append(dict(mode=mode,reason=str(error),evidence=getattr(error,'evidence',{})))
                if mode=='spread':raise
    if withdrawal not in ['lift','spread']:raise ValueError('unknown withdrawal mode')
    from .grasp_control import GraspReferences
    if isinstance(shape,dict):
        import trimesh
        from .mesh_surface import MeshSurface
        known=shape;shape=MeshSurface(known['nominal'],trimesh.Trimesh(known['vertices'],known['faces'],process=False))
        shape.convexes=known['convexes']
        if shape.convexes is not None:shape.bounds=np.array([bounds for bounds,_ in shape.convexes])
    grasp=GraspReferences(model,meta,q,reference,surfaces,points,normals,shape=shape)
    oq=meta['object']['q'];gap=task_clearance_gap(shape,q[oq:oq+7],destination)
    gimbals=np.array([a['q'] for a in meta['actuators'] if a['role']=='vectoring'])
    avoid_pairs=[];repairs=[]
    for attempt in range(3):
        translated=False
        opening_gap=min(.02,gap) if withdrawal=='lift' else gap
        path=FixedRetreat(grasp,q,opening_gap_m=opening_gap,final_gap_m=opening_gap,obstacles=obstacles,maximum_material_step=.01,avoid_pairs=avoid_pairs)
        trims=[]
        for i,pose in enumerate(path.path):
            if i:
                pose=pose.copy();pose[gimbals]=path.path[i-1][gimbals]
                pose,trim=free_flight_trim(model,meta,pose)
            else:trim=static_balance(model,meta,pose,payload_fraction=0.,balance_thrust=True)
            if trim['status']!='static_screen_pass':raise ValueError('empty_opening_support_not_found')
            path.path[i]=pose;trims.append(trim)
        path.path,path.distances,schedules,support_repairs=refine_support_path(model,meta,path.path,path.distances,trims)
        try:
            clearance=dict(opening=check_opening_clearance(meta,shape,surfaces,path.path,obstacles))
            if withdrawal=='lift':
                from .grasp_geometry import robot_bottom
                from .free_flight_trim import support_schedule
                # A known-shape bounding sphere and the full robot bottom
                # give an endpoint beyond the same final object pose set.
                radius=float(np.linalg.norm(shape.vertices,axis=1).max() if shape.vertices is not None else np.linalg.norm(shape.size/2))
                last=path.path[-1];raised=last.copy()
                rise=max(0.,q[oq+2]+radius+gap-robot_bottom(model,meta,last))
                raised[2]+=rise
                # All pads have opened. They are no longer intended contacts
                # during the rigid translation, including its interior.
                clearance['withdrawal']=check_opening_clearance(meta,shape,[],[last,raised],obstacles)
                if rise>1e-9:
                    trim=schedules[-1][-1][1]
                    schedules.append(support_schedule(model,meta,last,raised,trim,trim))
                    path.path.append(raised);path.distances=np.r_[path.distances,path.distances[-1]+rise]
                    translated=True
        except ValueError as error:
            evidence=getattr(error,'evidence',{});parts=evidence.get('parts',[])
            pair=tuple(name.rsplit('/',1)[0] for name in parts)
            known=set(meta['frames'])|set(['support','destination_support'][:len(obstacles)])
            if attempt==2 or len(pair)!=2 or any(name not in known for name in pair) or pair in avoid_pairs:raise
            # Generate a constraint from the offending links. The existing
            # convex-hull fit is conservative; exact owned part geometry is
            # checked again afterwards with the unchanged physical clearance.
            avoid_pairs.append(pair);repairs.append(evidence)
            continue
        break
    speeds=np.full(len(path.path)-1,.006)
    if translated:speeds[-1]=.06
    duration=sum(max(float(b-a)/speed,float(abs(last[gimbals]-first[gimbals]).max())/.3)
                 for a,b,first,last,speed in zip(path.distances[:-1],path.distances[1:],path.path[:-1],path.path[1:],speeds))
    return dict(path=path.path,distances=path.distances,schedules=schedules,minimum_duration_s=duration,planning_s=time.perf_counter()-start,avoid_pairs=avoid_pairs,clearance_repairs=repairs,support_repairs=support_repairs,clearance=clearance,withdrawal=withdrawal,segment_normal_speeds=speeds)


class SupportedRelease:
    def __init__(self,plan,grasp,flight,q,executor=None,rest_to_rest=False):
        start=time.perf_counter();self.plan,self.grasp,self.flight=plan,grasp,flight
        self.model,self.meta=grasp.model,grasp.meta;self.dt=flight.dt
        self.first=grasp.reference.copy()
        # Grasp integration updates Dock/base references, leaving gimbal entries
        # at the old seed. Pair the captured thrust with its issued gimbal
        # commands so Flight's allocation center does not jump on handoff.
        self.first[flight.qi[flight.ai]]=flight.command[flight.ai]
        self.end=q.copy();self.reference=self.first.copy();self.elapsed=0.
        self.nu=np.zeros(self.model.nv);self.nu[:6]=grasp.last_velocity[:6];self.nu[grasp.vi]=grasp.last_velocity[6:]
        self.bias0=flight.kp*(flight.command-self.first[flight.qi])-flight.kd*self.nu[flight.vi]
        self.force0=flight.force.copy()
        # Empty flight may require a different rotor direction. Dock/base stay
        # fixed in this endpoint fit; all changes are executed continuously.
        self.end,trim=free_flight_trim(self.model,self.meta,self.end)
        if trim['status']!='static_screen_pass':
            error=ValueError('empty_release_support_not_found');error.evidence=trim
            raise error
        self.bias1=np.array([j['effort_nm'] for j in trim['joints']]);self.force1=np.array(trim['thrust'])
        delta=pin.difference(self.model,self.first,self.end)
        self.duration=max(2.,1.875*float(np.max(abs(np.r_[delta[:6],delta[grasp.vi]])/grasp.scale)))
        gimbals=[a['v'] for a in self.meta['actuators'] if a['role']=='vectoring']
        self.duration=max(self.duration,1.875*float(np.max(abs(delta[gimbals]),initial=0.))/.3)
        acceleration=np.r_[np.full(3,.1),np.full(3,.3),np.full(len(grasp.vi),.5)]
        self.bridge=max(.25,8*float(np.max(abs(grasp.last_velocity)/acceleration)))
        self.retreat=None;self.schedules=None;self.planning_times=[time.perf_counter()-start]
        self.executor=executor;self.pending=None;self.waiting=0.;self.opening_timeout=None
        self.rest_to_rest=rest_to_rest

    def unload(self):
        t=self.elapsed;s=float(smooth(t/self.duration));target=pin.interpolate(self.model,self.first,self.end,s)
        u=min(1.,t/self.bridge);blend=self.bridge*u*(1-u)**2
        target[:3]+=blend*self.nu[:3];target[3:7]=advance_world_orientation(target[3:7],blend*self.nu[3:6]);target[self.grasp.qi]+=blend*self.nu[self.grasp.vi]
        difference=pin.difference(self.model,self.reference,target)/self.dt;R=pin.Quaternion(self.reference[3:7]).matrix()
        tv=difference.copy();tv[:3]=R@difference[:3];tv[3:6]=R@difference[3:6]
        if t==0:tv=self.nu.copy()
        self.reference=target;self.grasp.reference=target.copy();self.grasp.last_velocity=np.r_[tv[:6],tv[self.grasp.vi]]
        self.elapsed+=self.dt
        return target,tv,1-s,(1-s)*self.bias0+s*self.bias1,(1-s)*self.force0+s*self.force1

    @property
    def unloaded_reference_settled(self):
        return self.elapsed>=self.duration+2.

    def prepare_opening(self,q):
        plan=self.plan;p=plan.plant;shape=plan.shape
        obstacles=[(p.mj.geom(name).pos.copy(),p.mj.geom(name).size.copy()) for name in ['support','destination_support']]
        from .mesh_surface import MeshSurface
        if isinstance(shape,MeshSurface):
            shape=dict(nominal=shape.nominal,vertices=shape.vertices,faces=shape.faces,convexes=shape.convexes)
        arguments=(self.model,self.meta,shape,q.copy(),self.reference.copy(),plan.surfaces,plan.points,plan.normals,obstacles,plan.destination)
        if self.executor is None:self.accept_opening(plan_opening(*arguments),q)
        else:self.pending=self.executor.submit(plan_opening,*arguments)

    def accept_opening(self,result,q):
        if self.rest_to_rest:
            result=dict(result)
            result['segment_durations']=rest_to_rest_durations(self.model,self.meta,result['path'],result['distances'],normal_speed=result.get('segment_normal_speeds',.006))
        self.retreat=FixedRetreat(self.grasp,q,prepared=result)
        self.schedules=result['schedules']
        self.opening_timeout=self.waiting+1.875*result['minimum_duration_s']+.006/.002+2.
        if not self.rest_to_rest:
            # The legacy distance follower retains its uniform 6 mm/s speed.
            self.opening_timeout=max(self.opening_timeout,self.waiting+1.875*result['distances'][-1]/.006+.006/.002+2.)
        self.planning_times.append(result['planning_s'])
        self.opening_evidence=dict(path=[q.tolist() for q in result['path']],distances=result['distances'].tolist(),avoid_pairs=result.get('avoid_pairs',[]),clearance_repairs=result.get('clearance_repairs',[]),support_repairs=result.get('support_repairs',[]),planning_s=result['planning_s'],clearance=result.get('clearance'),withdrawal=result.get('withdrawal'),route_rejections=result.get('route_rejections',[]))
        if self.rest_to_rest:
            self.opening_timeout=self.waiting+float(sum(result['segment_durations']))+2.01
            self.opening_evidence['segment_durations_s']=result['segment_durations'].tolist()

    def open(self,q,v):
        if self.retreat is None and self.pending is None:self.prepare_opening(q)
        if self.retreat is None:
            if self.pending.done():self.accept_opening(self.pending.result(),q)
            else:
                self.waiting+=self.dt
                return self.reference.copy(),np.zeros(self.model.nv),0.,self.bias1,self.force1
        path=self.retreat;target,tv=path.update(q,v,self.plan.destination,preload=np.full(len(self.grasp.preload),-.03),normal_speed=.006)
        index=max(0,min(len(self.schedules)-1,int(np.searchsorted(path.distances,path.distance,side='right')-1)))
        fraction=(path.distance-path.distances[index])/(path.distances[index+1]-path.distances[index])
        joint,force=scheduled_support(self.schedules[index],fraction)
        return target,tv,0.,joint,force

    @property
    def opening_reference_complete(self):
        return self.retreat is not None and self.retreat.distance>=self.retreat.distances[-1]-1e-8
