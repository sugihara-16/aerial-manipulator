"""Observed flight to dock frames, then guarded rigid-seam engagement.

Matches the copied source's selected-pair collision-filter fallback. This is
not a simulation of the latch teeth or funnel. Joint servos remain movable;
only the two connect frames become constrained, without resetting state.
"""
import xml.etree.ElementTree as ET
import numpy as np
import pinocchio as pin
import mujoco
from dataclasses import dataclass
from .model import build_model, PORTS
from .mujoco_model import PlantModel, make_plant,CONTACT_CACHE


def mechanism(module, port):
    return f'm{module}/'+('pitch_dock_mech' if port.startswith('pitch') else 'yaw_dock_mech')+port[-1]


def make_assembly_plant(n, edges, payload=None, *, timestep=.001, meshes=True):
    plant=make_plant(n,[],payload,timestep=timestep,meshes=meshes,
                     mesh_cache=CONTACT_CACHE,floor=True)
    doc=ET.fromstring(plant.xml);equality=ET.SubElement(doc,'equality')
    for i,e in enumerate(edges):
        first=f"m{e['a']}/{PORTS[e['port_a']]}";second=f"m{e['b']}/{PORTS[e['port_b']]}"
        body=next(b for b in doc.findall('.//body') if b.get('name')==second)
        ET.SubElement(body,'site',name=f'mate{i}',quat='0 0 0 1',size='.002',rgba='0 0 0 0')
        ET.SubElement(equality,'weld',name=f'seam{i}',site1=first,site2=f'mate{i}',active='false',
                      solref='.005 1',solimp='.999 .999 .001',torquescale='1')
        # This exact Dock-body pair is the source's accepted contactless fallback.
        # All other body pairs retain physical collision and separation checks.
        ET.SubElement(doc.find('contact'),'exclude',body1=mechanism(e['a'],e['port_a']),
                      body2=mechanism(e['b'],e['port_b']))
    plant.xml=ET.tostring(doc,encoding='unicode');plant.mj=mujoco.MjModel.from_xml_string(plant.xml)
    return plant


def connected_view(plant, edges, payload=None):
    """Different controller coordinates over the SAME simulation state/model."""
    m,meta=build_model(plant.meta['modules'],edges,payload)
    meta['material_mesh_cache']=plant.meta.get('material_mesh_cache')
    joints=[(i,mujoco.mj_name2id(plant.mj,mujoco.mjtObj.mjOBJ_JOINT,m.names[i])) for i in range(1,m.njoints)]
    if any(j<0 for _,j in joints):raise ValueError('controller view has an absent physical joint')
    return PlantModel(m,meta,plant.mj,plant.xml,joints)


def actuator_indices(plant):
    names=[a['name'] for a in plant.meta['actuators']]+[f'rotor{i}' for i in range(len(plant.meta['rotors']))]
    return np.array([mujoco.mj_name2id(plant.mj,mujoco.mjtObj.mjOBJ_ACTUATOR,n) for n in names])


def forest_posture(forest, connected, q, data=None):
    """Use only to initialize or form a target, never to teleport live bodies."""
    m,d=connected.pin,connected.pin.createData() if data is None else data
    pin.forwardKinematics(m,d,q);pin.updateFramePlacements(m,d)
    result=pin.neutral(forest.pin)
    for b in forest.meta['bases']:
        T=d.oMf[connected.meta['frames'][f"m{b['module']}/root"]];k=b['q']
        result[k:k+3]=T.translation;result[k+3:k+7]=pin.Quaternion(T.rotation).coeffs()
    values={a['name']:q[a['q']] for a in connected.meta['actuators']}
    for a in forest.meta['actuators']:result[a['q']]=values[a['name']]
    if forest.meta['object']:
        result[forest.meta['object']['q']:]=q[connected.meta['object']['q']:]
    return result


def dock_error(plant,q,v,edge,data=None):
    m,d=plant.pin,plant.pin.createData() if data is None else data
    pin.forwardKinematics(m,d,q,v);pin.updateFramePlacements(m,d)
    ids=[plant.meta['frames'][f"m{edge[key]}/{PORTS[edge[port]]}"] for key,port in [('a','port_a'),('b','port_b')]]
    A,B=[d.oMf[i] for i in ids];delta=A.rotation.T@(B.translation-A.translation)
    angular=pin.log3(A.rotation@pin.rpy.rpyToMatrix(0,0,np.pi)@B.rotation.T)
    va,vb=[pin.getFrameVelocity(m,d,i,pin.ReferenceFrame.LOCAL_WORLD_ALIGNED) for i in ids]
    # The origin offset matters during approach; subtract the velocity of the
    # same point transported by the leader's rotating frame.
    relative=vb.linear-va.linear-np.cross(va.angular,B.translation-A.translation)
    return dict(axial_m=float(delta[0]),transverse_m=float(np.linalg.norm(delta[1:])),
                attitude_rad=float(np.linalg.norm(angular)),linear_speed_m_s=float(np.linalg.norm(relative)),
                angular_speed_rad_s=float(np.linalg.norm(vb.angular-va.angular)))


def ready_to_attach(error):
    return (abs(error['axial_m'])<=.003 and error['transverse_m']<=.002 and error['attitude_rad']<=np.deg2rad(.5)
            and error['linear_speed_m_s']<=.01 and error['angular_speed_rad_s']<=.03)


def following_base_target(plant,q,edge,offset=0.,data=None):
    """Align the follower with observed ports, including joint deflection."""
    data=plant.pin.createData() if data is None else data
    pin.framesForwardKinematics(plant.pin,data,q)
    leader=data.oMf[plant.meta['frames'][f"m{edge['a']}/{PORTS[edge['port_a']]}"]]
    follower=data.oMf[plant.meta['frames'][f"m{edge['b']}/{PORTS[edge['port_b']]}"]]
    base=data.oMf[plant.meta['frames'][f"m{edge['b']}/root"]]
    desired=leader*pin.SE3(pin.rpy.rpyToMatrix(0,0,np.pi),np.zeros(3))
    desired.translation+=offset*leader.rotation[:,0]
    target=desired*(base.inverse()*follower).inverse()
    return np.r_[target.translation,pin.Quaternion(target.rotation).coeffs()]


class DockFilters:
    """Only attached seams and the currently approached seam are filtered."""
    def __init__(self,plant,edges):
        self.original=plant.mj.exclude_signature.copy();self.seams=[]
        for e in edges:
            a,b=sorted([plant.mj.body(mechanism(e['a'],e['port_a'])).id,plant.mj.body(mechanism(e['b'],e['port_b'])).id])
            self.seams.append((a<<16)+b)

    def apply(self,model,indices):
        disabled=[s for i,s in enumerate(self.seams) if i not in indices]
        # MuJoCo's lookup expects sorted signatures. Keep the array dimension;
        # zero names the non-colliding world/world pair.
        model.exclude_signature[:]=np.sort(np.where(np.isin(self.original,disabled),0,self.original))


@dataclass
class AssemblyPlan:
    plant: object
    connected: object
    edges: list
    posture: np.ndarray
    parked: np.ndarray
    staging: list
    component_targets: list
    component_trims: list
    morph_path: list
    morph_trims: list
    morph_schedules: list
    filters: object
    evidence: dict


def prepare_assembly(task):
    """A bounded search over task-derived planar starts and coordinate paths."""
    import time
    from .grasp_geometry import robot_bottom
    from .free_flight_trim import free_flight_trim,check_fixed_thrust,support_schedule
    from .clearance import proximity_model,ContactClassifier
    from .mujoco_model import coat_grippers
    start=time.perf_counter();edges=task.plant.meta['edges'];n=task.plant.meta['modules']
    p=make_assembly_plant(n,edges,task.shape.payload(),timestep=task.plant.mj.opt.timestep)
    doc=ET.fromstring(p.xml);original=ET.fromstring(task.plant.xml)
    for geom in original.findall('./worldbody/geom'):
        if geom.get('name') in ['support','destination_support']:doc.find('worldbody').append(geom)
    doc.find('option').set('impratio','100')
    links=[mechanism(module,port) for module,port in task.surfaces]
    p.xml=coat_grippers(ET.tostring(doc,encoding='unicode'),links);p.mj=mujoco.MjModel.from_xml_string(p.xml)
    connected=connected_view(p,edges,task.shape.payload());m,meta=connected.pin,connected.meta
    near=proximity_model(p.mj);nd=mujoco.MjData(near);filters=DockFilters(p,edges)
    names=[near.geom(i).name for i in range(near.ngeom)]
    classify=ContactClassifier(near,names,[])
    def clear(fq):
        nd.qpos[:]=p.configuration(fq);mujoco.mj_fwdPosition(near,nd)
        intended,limits=classify(nd)
        return not np.any(~intended & (np.asarray(nd.contact.dist)<limits-1e-6))
    def path_clear(view,path):
        for a,b in zip(path[:-1],path[1:]):
            for t in np.linspace(0,1,21):
                q=pin.interpolate(view.pin,a,b,float(t))
                if not clear(forest_posture(p,view,q)):return False
        return True
    goal=task.initial_q.copy();goal[2]+=max(0.,1.-robot_bottom(m,meta,goal))
    attempts=[]
    for alpha in [-.5,-1.,0.,.5,1.]:
        row=dict(yaw_seed_scale=alpha);attempts.append(row)
        q=pin.neutral(m);q[:3]=goal[:3];q[meta['object']['q']:]=task.source
        q[3:7]=pin.Quaternion(pin.rpy.rpyToMatrix(0,0,pin.rpy.matrixToRpy(pin.Quaternion(goal[3:7]).matrix())[2])).coeffs()
        for a in meta['actuators']:
            if 'yaw' in a['name']:q[a['q']]=alpha*goal[a['q']]
        filters.apply(near,range(len(edges)))
        if not clear(forest_posture(p,connected,q)):row['rejected']='planar_collision';continue
        rough=None
        orders=[['pitch','yaw'],['yaw','pitch'],[f'm{i}/' for i in range(n-1,-1,-1)],
                [f'm{i}/' for i in range(n)]]
        for order in orders:
            candidate=[q.copy()]
            for group in order+['all']:
                pose=candidate[-1].copy()
                for a in meta['actuators']:
                    if group=='all' or group in a['name']:pose[a['q']]=goal[a['q']]
                if group=='all':pose[:7]=goal[:7]
                candidate.append(pose)
            if path_clear(connected,candidate):rough=candidate;row['coordinate_order']=order;break
        if rough is None:row['rejected']='morph_collision';continue
        components=[];trims=[]
        independent=connected_view(p,[],task.shape.payload())
        single_q,single_trim=free_flight_trim(independent.pin,independent.meta,forest_posture(independent,connected,q))
        if single_trim['status']!='static_screen_pass':row['rejected']='independent_hover_not_found';continue
        q=forest_posture(connected,independent,single_q)
        for count in range(len(edges)+1):
            view=connected_view(p,edges[:count],task.shape.payload())
            target=forest_posture(view,connected,q)
            trim=check_fixed_thrust(view.pin,view.meta,target,single_trim['thrust'])
            if trim['status']!='static_screen_pass':break
            components.append(target);trims.append(trim)
        if len(trims)!=len(edges)+1:row['rejected']='component_trim_not_found';continue
        path=[q.copy()];path_trims=[trims[-1]];gimbal=np.array([a['q'] for a in meta['actuators'] if a['role']=='vectoring'])
        failed=False
        for first,last in zip(rough[:-1],rough[1:]):
            for t in np.linspace(0,1,6)[1:]:
                target=pin.interpolate(m,first,last,float(t));target[gimbal]=path[-1][gimbal]
                target,trim=free_flight_trim(m,meta,target)
                if trim['status']!='static_screen_pass':failed=True;break
                path.append(target);path_trims.append(trim)
            if failed:break
        if failed:row['rejected']='morph_trim_not_found';continue
        # End with the task's actual gimbal posture and stationary load screen.
        from .equilibrium import static_balance
        last_trim=static_balance(m,meta,goal,payload_fraction=0.,balance_thrust=True)
        if last_trim['status']!='static_screen_pass':row['rejected']='handoff_trim_not_found';continue
        path.extend([goal,task.initial_q.copy()]);path_trims.extend([last_trim,last_trim])
        if not path_clear(connected,path):row['rejected']='trimmed_path_collision';continue
        final_forest=forest_posture(p,connected,q);pd=m.createData();pin.framesForwardKinematics(m,pd,q)
        parking=final_forest.copy();staging=[]
        for e in edges:
            direction=pd.oMf[meta['frames'][f"m{e['a']}/{PORTS[e['port_a']]}"]].rotation[:,0]
            base=next(b for b in p.meta['bases'] if b['module']==e['b']);k=base['q']
            parking[k:k+3]+=.35*direction+np.array([0,0,.5+.35*e['b']])
            stage=final_forest[k:k+7].copy();stage[:3]+=.15*direction;staging.append(stage)
        state=parking.copy();failed=False
        for i,e in enumerate(edges):
            filters.apply(near,range(i+1));k=next(b['q'] for b in p.meta['bases'] if b['module']==e['b'])
            staged=state.copy();staged[k:k+7]=staging[i];docked=state.copy();docked[k:k+7]=final_forest[k:k+7]
            if not path_clear(p,[state,staged,docked]):failed=True;break
            state=docked
        if failed:row['rejected']='assembly_approach_collision';continue
        try:
            schedules=[support_schedule(m,meta,first,last,a,b) for first,last,a,b in
                       zip(path[:-1],path[1:],path_trims[:-1],path_trims[1:])]
        except ValueError as error:row['rejected']=str(error);continue
        row['status']='accepted'
        return AssemblyPlan(p,connected,edges,q,parking,staging,components,trims,path,path_trims,schedules,filters,
                            dict(scope='sampled_geometry_and_stationary_loads_only',attempts=attempts,wall_s=time.perf_counter()-start))
    error=ValueError('no assembly found in planar task-derived search');error.evidence=dict(attempts=attempts,wall_s=time.perf_counter()-start)
    raise error

