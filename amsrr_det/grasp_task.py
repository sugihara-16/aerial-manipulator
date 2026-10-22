"""Geometric candidates and observation-driven grasp/transport execution.

The force screen does not establish a realized contact equilibrium. Physical
execution uses the actual drives and contact model and is scored separately.
"""
from dataclasses import dataclass
from collections import deque
from pathlib import Path
import time
import xml.etree.ElementTree as ET
import numpy as np
from .clearance import ContactClassifier
import pinocchio as pin
import mujoco
from .model import build_model,ROOT
from .mujoco_model import make_plant,coat_grippers,numbers,CONTACT_CACHE
from .grasp_geometry import fit_contacts,aim_rotors_up,robot_bottom,separating_translation
from .grasp_control import GraspReferences,surface_points
from .equilibrium import static_balance,restoring_screen
from .flight import FlightController
from .objects import ObjectShape
from .clearance import proximity_model, classify_contacts
from .numerics import advance_world_orientation


def smooth(value):
    t=np.clip(value,0.,1.);return t*t*t*(10+t*(-15+6*t))


def quintic_path_velocity(displacement,elapsed,duration):
    """World twist of the known translation path, not observed object motion."""
    u=float(np.clip(elapsed/duration,0.,1.))
    return np.r_[np.asarray(displacement)*(30*u*u*(1-u)**2/duration),np.zeros(3)]


def placement_height(lift_height,elapsed,extent_change,gap):
    """Goal height and its time derivative with observed orientation held fixed."""
    height=lift_height*(1-smooth(elapsed/10))
    rate=quintic_path_velocity([0.,0.,-lift_height],elapsed,10.)[2]
    u=float(np.clip((.02-height)/.02,0.,1.))
    correction=extent_change+gap
    return height+smooth(u)*correction,rate*(1-30*u*u*(1-u)**2*correction/.02)


# 0.1 micrometre distance-query tolerance, independent of successful loads.
# Source contact-onset sweeps resolve 0.07/2**22 m; this is not force confidence
# or a bound on the separately recorded collision-mesh approximation error.
CONTACT_QUERY_TOLERANCE=1e-7
PROBE_HEIGHT=.005
NOMINAL_CONTACT_GAP=0.


def object_height_at_clearance(shape,orientation,support_top,clearance):
    """COM height for the requested bottom clearance at this orientation."""
    return support_top+clearance-shape.lowest_z(np.r_[0.,0.,0.,orientation])


def probe_goal_height(shape,source,pose,support_top,elapsed):
    """Interpolate bottom clearance, then account for the observed orientation."""
    initial_clearance=shape.lowest_z(source)-support_top
    clearance=initial_clearance+(PROBE_HEIGHT-initial_clearance)*smooth(elapsed/3)
    return object_height_at_clearance(shape,pose[3:],support_top,clearance)


def observed_extent_rate(shape,pose,body_angular_velocity,period=.01):
    """Downward extent change if observed angular velocity persists one period.

    This is a geometric prediction, not guaranteed future object motion. Using
    the whole shape handles changes of the lowest vertex without choosing one.
    """
    omega=pin.Quaternion(pose[3:]).matrix()@body_angular_velocity
    rotated=np.array(pose,copy=True)
    rotated[3:]=advance_world_orientation(pose[3:],period*omega)
    return (shape.lowest_z(pose)-shape.lowest_z(rotated))/period


def probe_acquired(shape,source,pose,velocity,support_top,compression,near_gap_m=CONTACT_QUERY_TOLERANCE):
    clearance=shape.lowest_z(pose)-support_top
    rotation=np.linalg.norm(pin.log3(pin.Quaternion(source[3:]).matrix()@pin.Quaternion(pose[3:]).matrix().T))
    return (pose[2]-source[2]>=.003 and clearance>=.003
            and np.linalg.norm(pose[:2]-source[:2])<=.002 and abs(clearance-PROBE_HEIGHT)<=.002 and rotation<=.05
            and np.linalg.norm(velocity[:3])<=.005 and np.linalg.norm(velocity[3:])<=.02
            and np.all(compression>=-near_gap_m))


def quiet_object(velocity,linear_limit=.02,angular_limit=.05):
    """Separate m/s and rad/s; this is a phase guard, not a force guarantee."""
    return np.linalg.norm(velocity[:3])<linear_limit and np.linalg.norm(velocity[3:])<angular_limit


def free_object_orientation_gain(clearance):
    """Enable pose correction during the 5 mm probe once off the support.

    Use geometry only. Zero feedback at the probe height would make the
    probe's rotation acceptance condition unreachable after a yaw disturbance.
    """
    return float(smooth((clearance-.0002)/(PROBE_HEIGHT-.0002)))


def bounded_pose_window(poses,position_span=.001,angle_span=.01):
    """Bound oscillation/drift, without averaging away instantaneous motion."""
    if len(poses)<31:return False
    rows=np.asarray(poses)
    # Half the pairwise bound from one reference bounds every pair by the
    # triangle inequality. Quaternion signs do not affect the rotation test.
    position=np.linalg.norm(rows[:,:3]-rows[0,:3],axis=1).max()
    angle=2*np.arccos(np.clip(abs(rows[:,3:]@rows[0,3:]),0.,1.)).max()
    return position<=position_span/2 and angle<=angle_span/2


@dataclass
class GraspPlan:
    plant: object
    shape: ObjectShape
    source: np.ndarray
    destination: np.ndarray
    support_top: float
    open_q: np.ndarray
    closed_q: np.ndarray
    initial_q: np.ndarray
    surfaces: list
    points: np.ndarray
    normals: np.ndarray
    samples: list
    trims: list
    wall_s: float
    fit: dict
    lift_height: float = .12
    approach_height: float = .15
    operating_point: dict | None = None


def plan_seed(shape,source,destination,support_top, *, timestep=.001,mode='yaw',mesh_cache=CONTACT_CACHE,axis_world=None,geometry_only=False,contact_tilt=0.,base_roll=0.):
    start=time.perf_counter();source=np.asarray(source);destination=np.asarray(destination)
    edges=[dict(a=0,port_a='pitch1',b=1,port_b='yaw1'),dict(a=0,port_a='pitch2',b=2,port_b='yaw2')]
    surfaces=[(1,'yaw2'),(2,'yaw1')]
    if mode=='pitch':
        edges=[dict(a=0,port_a='yaw1',b=1,port_b='pitch1'),dict(a=0,port_a='yaw2',b=2,port_b='pitch2')]
        surfaces=[(1,'pitch2'),(2,'pitch1')]
    if geometry_only:
        # Seed fitting reads kinematics and material geometry only. Its accepted
        # result is compiled with full collision geometry by the ordinary screen.
        from types import SimpleNamespace
        model,meta=build_model(3,edges,shape.payload())
        meta['material_mesh_cache']=str(Path(mesh_cache or ROOT/'build/collision_meshes').resolve())
        p=SimpleNamespace(pin=model,meta=meta,xml='')
    else:
        p=make_plant(3,edges,payload=shape.payload(),mesh_cache=mesh_cache,timestep=timestep,floor=True)
    obstacles=support_obstacles(shape,source,destination,support_top)
    q=pin.neutral(p.pin);oq=p.meta['object']['q'];q[oq:oq+7]=source
    # Use a horizontal object-frame axis as the first contact direction.
    R=pin.Quaternion(source[3:]).matrix();directions=[R[:,i] for i in range(3)]
    axis=max(directions,key=lambda n:np.linalg.norm(n[:2]) if abs(n[1])>.1 else 0.).copy() if axis_world is None else np.array(axis_world,dtype=float)
    axis[2]=0;axis/=np.linalg.norm(axis)
    if np.cross(axis,[0,0,1])@(destination[:3]-source[:3])<0:axis=-axis
    yaw=np.arctan2(-axis[0],axis[1])
    if not np.isfinite(base_roll):raise ValueError('invalid initial base roll')
    q[3:7]=pin.Quaternion(pin.exp3(axis*base_roll)@pin.rpy.rpyToMatrix(0.,0.,yaw)).coeffs()
    points,normals=shape.opposed_sites(source,axis,contact_tilt)
    world=points@R.T+source[:3];world_normals=normals@R.T
    pd=p.pin.createData();pin.framesForwardKinematics(p.pin,pd,q)
    samples=[surface_points(p.pin,p.meta,q,s) for s in surfaces]
    from .contact_geometry import material_triangles
    for _,port in surfaces:material_triangles(port,p.meta.get('material_mesh_cache'))
    q[:3]+=source[:3]-np.mean([pd.oMf[f].translation for f,_ in samples],axis=0)
    if mode=='pitch':q[2]=source[2]+.3
    # A small set of initial joint folds is a numerical initialisation, not a
    # catalogue of finished structures. Every target and posture is re-solved.
    base_xml=p.xml;failures=[];last_error=None;best_coarse=None
    for attempt in range(4):
        initial=q.copy()
        if attempt:
            fold=[-1.4,.8,-.8][attempt-1]
            for a in p.meta['actuators']:
                if a['role']=='dock' and 'yaw' in a['name']:
                    if a['name'].startswith('m1/'):initial[a['q']]=fold
                    if a['name'].startswith('m2/'):initial[a['q']]=-fold
        closed,fit=fit_contacts(p.pin,p.meta,initial,surfaces,world+.001*world_normals,world_normals,
                               floor_clearance=.03,base_rotation=attempt>0,obstacles=obstacles)
        score=fit['position_error_m']+.15*fit['normal_error']
        if best_coarse is None or score<best_coarse[0]:
            best_coarse=(score,closed.copy(),dict(fit))
        # This is only a seed filter; material gaps and full geometry below
        # decide acceptance. Tangential site offsets may be harmless.
        if fit['position_error_m']>.03 or fit['normal_error']>.15:continue
        contact_points=points.copy();contact_normals=normals.copy()
        try:
            opened,closed,fit=refine_material_posture(p,shape,source,closed,surfaces,contact_points,contact_normals,
                                                    base_rotation=attempt>0,obstacles=obstacles)
            fit['contact_tilt_rad']=float(contact_tilt)
            fit['initial_base_roll_rad']=float(base_roll)
            if geometry_only:
                return dict(plant=p,opened=opened,closed=closed,surfaces=surfaces,points=contact_points,
                            normals=contact_normals,fit=fit,wall_s=time.perf_counter()-start)
            p.xml=base_xml
            plan=finish_plan(p,shape,source,destination,support_top,opened,closed,surfaces,contact_points,contact_normals,fit,start)
            plan.fit['seed_attempt']=attempt;plan.fit['prior_rejections']=failures
            return plan
        except ValueError as error:
            last_error=error;failures.append(str(error))
    if geometry_only and best_coarse is not None:
        # A short initial arm may not reach a valid task. Keep its best coarse
        # pose as a growth seed; it is not a feasible/accepted structure.
        _,closed,fit=best_coarse
        # Surface locations are search variables, not task-prescribed anchors.
        # If fixed seeds cannot be reached, allow tangential movement before
        # growing links. A later material/whole-geometry screen still decides.
        closed,fit=fit_contacts(p.pin,p.meta,closed,surfaces,world,world_normals,
            floor_clearance=.03,base_rotation=True,obstacles=obstacles,
            object_shape=shape,max_evaluations=65)
        points=np.asarray(fit.pop('object_points'));normals=np.asarray(fit.pop('inward_normals'))
        fit.update(contact_tilt_rad=float(contact_tilt),growth_seed_only=True,
                   initial_base_roll_rad=float(base_roll),prior_rejections=failures)
        return dict(plant=p,opened=closed.copy(),closed=closed,surfaces=surfaces,
                    points=points.copy(),normals=normals.copy(),fit=fit,
                    wall_s=time.perf_counter()-start)
    if last_error is not None:raise last_error
    raise ValueError('seed has no accurate contact posture')


def support_obstacles(shape,source,destination,support_top):
    R=pin.Quaternion(source[3:]).matrix();half=(abs(R)@(shape.size/2))[:2]+.04
    if shape.vertices is not None:half=np.max(abs(shape.vertices@R.T),axis=0)[:2]+.04
    delta=np.asarray(destination[:3])-source[:3]
    physical=[np.r_[pose[:2],support_top/2] for pose in [source,destination]]
    # Test the same posture at pickup and placement. Translating obstacles in
    # the opposite direction expresses both constraints in pickup coordinates.
    centers={tuple(center-shift) for center in physical for shift in [np.zeros(3),delta]}
    return [(np.array(center),np.r_[half,support_top/2]) for center in sorted(centers)]


def refine_material_posture(p,shape,source,closed,surfaces,points,normals, *, base_rotation=True,obstacles=()):
    from .contact_geometry import material_triangles,close_material
    R=pin.Quaternion(source[3:]).matrix();world=points@R.T+source[:3];world_normals=normals@R.T
    pd=p.pin.createData();samples=[surface_points(p.pin,p.meta,closed,s) for s in surfaces]
    fit=dict(position_error_m=0.,normal_error=0.,scope='material_contact_fit',task_success=False)
    # Contact frames lie at authored coordinates, but a curved object can enter
    # a pad's real recess before touching its rim. Correct the frame targets from
    # closest material gaps; never fill the recess with an imagined plane.
    from .contact_geometry import close_material
    for _ in range(6):
        pin.framesForwardKinematics(p.pin,pd,closed);obj=pd.oMf[p.meta['object']['frame']]
        patches=[close_material(shape,obj.inverse()*pd.oMf[frame],material_triangles(port,p.meta.get('material_mesh_cache')))
                 for (frame,_),(_,port) in zip(samples,surfaces)]
        gap=np.array([patch[2].min() for patch in patches])
        patch_centers=[];material_normals=[]
        for _,targets,gaps,outward in patches:
            weights=np.maximum(0,.00015-(gaps-gaps.min()));weights/=weights.sum()
            patch_centers.append(obj.act(weights@targets))
            normal=-(weights@outward);normal/=np.linalg.norm(normal)
            material_normals.append(obj.rotation@normal)
        error=world-np.array(patch_centers)
        tangent=error-np.sum(error*world_normals,axis=1)[:,None]*world_normals
        material_normals=np.array(material_normals)
        frame_normals=np.array([pd.oMf[frame].rotation[:,0] for frame,_ in samples])
        alignment=np.linalg.norm(frame_normals-material_normals,axis=1)
        if np.max(abs(gap-NOMINAL_CONTACT_GAP))<.00001 and alignment.max()<.1:break
        centers=np.array([pd.oMf[frame].translation for frame,_ in samples])
        # Contact sites may move along a face. Do not sacrifice the actual gap
        # to restore an arbitrary patch centroid or improve an already allowed
        # normal alignment. The resulting points are screened for loads below.
        # The actual closest patch can move on a curved object. Align with
        # its normal, rather than the original numerical site's normal.
        preserve=alignment<.1
        fit_normals=np.where(preserve[:,None],frame_normals,material_normals)
        closed,fit=fit_contacts(p.pin,p.meta,closed,surfaces,centers+(gap-NOMINAL_CONTACT_GAP)[:,None]*material_normals,fit_normals,
                               floor_clearance=.03,base_rotation=base_rotation,obstacles=obstacles)
    # Surface sites are search targets, not task-prescribed attachment points.
    # A valid material contact may slide along a face. Judge the resulting gap
    # and normal instead of rejecting a harmless tangent shift of the frame.
    pin.framesForwardKinematics(p.pin,pd,closed);obj=pd.oMf[p.meta['object']['frame']]
    patches=[close_material(shape,obj.inverse()*pd.oMf[frame],material_triangles(port,p.meta.get('material_mesh_cache')))
             for (frame,_),(_,port) in zip(samples,surfaces)]
    gap=np.array([patch[2].min() for patch in patches])
    actual_points=[];actual_normals=[];alignment=[]
    for (frame,_),(_,targets,gaps,outward) in zip(samples,patches):
        weights=np.maximum(0,.00015-(gaps-gaps.min()));weights/=weights.sum()
        n=-(weights@outward);n/=np.linalg.norm(n)
        actual_points.append(weights@targets);actual_normals.append(n)
        alignment.append(np.linalg.norm(pd.oMf[frame].rotation[:,0]-obj.rotation@n))
    fit.update(material_gap_m=gap.tolist(),material_normal_error=max(alignment),
               nominal_contact_gap_m=NOMINAL_CONTACT_GAP,physical_contact_equilibrium_verified=False,
               contact_target_shift_m=np.linalg.norm(np.array(actual_points)-points,axis=1).tolist())
    if np.max(abs(gap-NOMINAL_CONTACT_GAP))>.00005 or max(alignment)>.1:
        error=ValueError('material contact fit failed')
        error.evidence=dict(fit=fit,q=closed.tolist(),surfaces=surfaces,points=points.tolist(),normals=normals.tolist(),material_gap=gap.tolist(),tangent_error=tangent.tolist())
        raise error
    points[:]=actual_points;normals[:]=actual_normals;world_normals=normals@R.T
    pin.framesForwardKinematics(p.pin,pd,closed)
    centers=np.array([pd.oMf[frame].translation for frame,_ in samples])
    opened,openfit=fit_contacts(p.pin,p.meta,closed,surfaces,centers-.045*world_normals,world_normals,
                               floor_clearance=.03,base_rotation=base_rotation,obstacles=obstacles)
    pin.framesForwardKinematics(p.pin,pd,opened);obj=pd.oMf[p.meta['object']['frame']]
    open_gaps=[close_material(shape,obj.inverse()*pd.oMf[frame],material_triangles(port,p.meta.get('material_mesh_cache')))[2].min()
               for (frame,_),(_,port) in zip(samples,surfaces)]
    fit['opening_method']='joint_fit'
    if min(open_gaps)<.025:
        # Matching offset contact points can unnecessarily prohibit tangential
        # motion. A common separating translation needs only the actual gaps.
        displacement=separating_translation(world_normals,.045)
        if displacement is not None:
            trial=closed.copy();trial[:3]+=displacement
            pin.framesForwardKinematics(p.pin,pd,trial);obj=pd.oMf[p.meta['object']['frame']]
            trial_gaps=[close_material(shape,obj.inverse()*pd.oMf[frame],material_triangles(port,p.meta.get('material_mesh_cache')))[2].min()
                        for (frame,_),(_,port) in zip(samples,surfaces)]
            if min(trial_gaps)>=.025:
                opened=trial;open_gaps=trial_gaps;fit['opening_method']='separating_translation'
    fit['open_material_gap_m']=list(map(float,open_gaps))
    if min(open_gaps)<.025:raise ValueError('material contact opening failed')
    return opened,closed,fit


def material_force_contacts(p,shape,closed,surfaces):
    """The same finite material points, usable before compiling dynamics."""
    from .contact_geometry import material_triangles,close_material
    pd=p.pin.createData();pin.framesForwardKinematics(p.pin,pd,closed)
    obj=pd.oMf[p.meta['object']['frame']];contacts=[]
    for surface in surfaces:
        frame,_=surface_points(p.pin,p.meta,closed,surface);port=surface[1]
        vertices,targets,gaps,outward=close_material(shape,obj.inverse()*pd.oMf[frame],material_triangles(port,p.meta.get('material_mesh_cache')))
        for point,target,normal in zip(vertices,targets,outward):
            # Equal/opposite forces share one world point, including soft
            # reference overlap, so no fictitious internal couple is added.
            world_point=(pd.oMf[frame].act(point)+obj.act(target))/2
            contacts.append(dict(frame=p.pin.frames[frame].name,point=pd.oMf[frame].actInv(world_point),target_frame='object',
                target_point=obj.actInv(world_point),normal=obj.rotation@normal,mu=2.25,
                max_force=30.,pad_force_limit=30.))
    return contacts


def prepare_static_trims(p,shape,opened,closed,surfaces,points,normals, *, loaded_trim=None,contacts=None):
    """Compute the unchanged trim screen before compiling collision geometry.

    Keep caller postures intact so original clearance checks use the original
    gimbals. Only a successful result is transferred into finish_plan.
    """
    opened=opened.copy();closed=closed.copy()
    if contacts is None:contacts=material_force_contacts(p,shape,closed,surfaces)
    # Gimbal changes cannot repair an impossible object-force projection.
    # Reject only the current conservative material-contact problem, not its
    # graph or other geometries. This replaces repeated identical failures.
    from .equilibrium import contact_normal_obstruction
    obstruction=None if loaded_trim is not None else contact_normal_obstruction(contacts,shape.mass*p.pin.gravity.linear,.5)
    if obstruction is not None:
        error=ValueError('material contact force directions cannot support this load')
        error.evidence=obstruction;raise error
    trims=None;screen_attempts=[]
    gimbals=[a for a in p.meta['actuators'] if a['role']=='vectoring']
    aimed=aim_rotors_up(p.pin,p.meta,closed)
    rng=np.random.default_rng(194)
    for attempt in range(16 if loaded_trim is None else 1):
        if attempt:
            for a in gimbals:
                angle=np.clip(aimed[a['q']]+rng.normal(0,.15),a['lower']+.02,a['upper']-.02)
                closed[a['q']]=opened[a['q']]=angle
        candidate=[static_balance(p.pin,p.meta,opened,payload_fraction=0.),
                   static_balance(p.pin,p.meta,closed,payload_fraction=0.),
                   static_balance(p.pin,p.meta,closed,contacts,minimum_pad_normal_force_n=.5) if loaded_trim is None else loaded_trim]
        screen_attempts.append(candidate)
        if all(t['status']=='static_screen_pass' for t in candidate[:2]) and candidate[2]['status'] in ['static_screen_pass','coupled_point_pass']:
            # Preserve the minimum worst-joint burden before smoothing the
            # rotor distribution; the second objective may add at most .02.
            trims=[static_balance(p.pin,p.meta,opened,payload_fraction=0.,balance_thrust=True,peak_slack=.02),
                   static_balance(p.pin,p.meta,closed,payload_fraction=0.,balance_thrust=True,peak_slack=.02),
                   static_balance(p.pin,p.meta,closed,contacts,balance_thrust=True,peak_slack=.02,minimum_pad_normal_force_n=.5) if loaded_trim is None else loaded_trim]
            break
    if trims is None:
        error=ValueError('seed lacks a load-feasible trim')
        error.evidence=dict(closed_q=closed.tolist(),open_q=opened.tolist(),surfaces=surfaces,points=points.tolist(),normals=normals.tolist(),contacts=[{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in c.items()} for c in contacts],screens=screen_attempts,model=p.meta)
        raise error
    return opened,closed,trims


def finish_plan(p,shape,source,destination,support_top,opened,closed,surfaces,points,normals,fit,start, *, loaded_trim=None,static_preparation=None):
    R=pin.Quaternion(source[3:]).matrix()
    world_normals=normals@R.T;pd=p.pin.createData()
    samples=[surface_points(p.pin,p.meta,closed,s) for s in surfaces]
    # Support footprint is generated from the object dimensions, with clearance.
    half_xy=(abs(R)@(shape.size/2))[:2]+.04
    if shape.vertices is not None:half_xy=np.max(abs(shape.vertices@R.T),axis=0)[:2]+.04
    doc=ET.fromstring(p.xml);option=doc.find('option');option.set('integrator','discrete');option.set('impratio','100')
    w=doc.find('worldbody')
    # Rigid supports use a 2 ms numerical contact time constant. At the
    # execution step of 0.25 ms this resolves support compliance without
    # changing the authored coating on the grasp pads.
    for name,pose in [('support',source),('destination_support',destination)]:
        ET.SubElement(w,'geom',name=name,type='box',pos=numbers([*pose[:2],support_top/2]),
                      size=numbers([*half_xy,support_top/2]),rgba='.4 .4 .4 1',
                      priority='1',solref='.002 1')
    links=[f'm{m}/'+('yaw_dock_mech' if port.startswith('yaw') else 'pitch_dock_mech')+port[-1] for m,port in surfaces]
    # Only world-fixed supports and numerical options were added above;
    # the existing compiled model already contains the required inertias.
    xml=coat_grippers(ET.tostring(doc,encoding='unicode'),links,reference_model=p.mj)
    p.xml=xml;p.mj=mujoco.MjModel.from_xml_string(xml)
    from .contact_geometry import material_triangles,close_material
    for _,port in surfaces:material_triangles(port,p.meta.get('material_mesh_cache'))
    # Reject unintended overlap before spending time on force allocation.
    cm=proximity_model(p.mj);md=mujoco.MjData(cm)
    names=[p.mj.geom(i).name for i in range(p.mj.ngeom)]
    classifiers={True:ContactClassifier(cm,names,links),False:ContactClassifier(cm,names,[])}
    for check_q in [opened,closed]:
        md.qpos[:]=p.configuration(check_q);mujoco.mj_fwdPosition(cm,md)
        intended,limits=classifiers[True](md)
        bad=np.flatnonzero(~intended & (np.asarray(md.contact.dist)<limits-1e-6))
        if len(bad):
            c=md.contact[bad[0]]
            raise ValueError('candidate has insufficient clearance: '+str([names[c.geom1],names[c.geom2],float(c.dist)]))
    contacts=material_force_contacts(p,shape,closed,surfaces)
    prepared=static_preparation if static_preparation is not None else prepare_static_trims(
        p,shape,opened,closed,surfaces,points,normals,loaded_trim=loaded_trim,contacts=contacts)
    opened[:],closed[:],trims=prepared
    # Same .5 N per-pad floor as the carried-load audit. This is only a
    # necessary allocation constraint; execution must realize strictly >.5 N.
    fit['restoring_load_screen']=restoring_screen(p.pin,p.meta,closed,contacts,minimum_pad_normal_force_n=.5)
    if not fit['restoring_load_screen']['passed']:raise ValueError('insufficient modeled restoring load margin')
    # Clear the whole robot, not only the object's bottom, above both supports.
    lift_height=max(.12,support_top+.035-robot_bottom(p.pin,p.meta,closed))
    # The approach is above the whole payload, not only above its table.
    object_top=(source[2]+np.max(shape.vertices@R[2]) if shape.vertices is not None
                else 2*source[2]-shape.lowest_z(source))
    approach_height=max(.15,object_top+.035-robot_bottom(p.pin,p.meta,opened))
    initial=opened.copy();rootR=pin.Quaternion(opened[3:7]).matrix()
    retreat=rootR[:,0].copy();retreat[2]=0
    if np.linalg.norm(retreat)<1e-8:retreat=rootR[:,1].copy();retreat[2]=0
    retreat/=np.linalg.norm(retreat)
    initial[:3]+=-.25*retreat+np.array([0,0,approach_height])
    # Recheck after the final gimbal allocation, including interpolated approach
    # and opening. These are sampled geometric checks, not tracking guarantees.
    above=opened.copy();above[2]+=approach_height
    fit['approach_durations_s']=[max(5.,1.875*np.linalg.norm(above[:3]-initial[:3])/.08),
                                 max(5.,1.875*np.linalg.norm(opened[:3]-above[:3])/.08)]
    checkpoints=[initial,above,opened,closed]
    lift=closed.copy();lift[2]+=lift_height;lift[p.meta['object']['q']+2]+=lift_height
    carried=lift.copy();carried[:3]+=destination[:3]-source[:3]
    carried[p.meta['object']['q']:p.meta['object']['q']+3]+=destination[:3]-source[:3]
    placed=closed.copy();placed[:3]+=destination[:3]-source[:3]
    placed[p.meta['object']['q']:p.meta['object']['q']+7]=destination
    released=opened.copy();released[:3]+=destination[:3]-source[:3]
    released[p.meta['object']['q']:p.meta['object']['q']+7]=destination
    checkpoints.extend([lift,carried,placed,released]);checked=0
    for segment,(first,last) in enumerate(zip(checkpoints,checkpoints[1:])):
        for t in np.linspace(0,1,9):
            check_q=pin.interpolate(p.pin,first,last,float(t))
            md.qpos[:]=p.configuration(check_q);mujoco.mj_fwdPosition(cm,md)
            intended,limits=classifiers[segment>=2](md)
            bad=np.flatnonzero(~intended & (np.asarray(md.contact.dist)<limits-1e-6));checked+=1
            if len(bad):
                c=md.contact[bad[0]]
                raise ValueError('candidate path clearance failed: '+str([segment,names[c.geom1],names[c.geom2],float(c.dist)]))
    fit['sampled_path_configurations']=checked
    return GraspPlan(p,shape,source,destination,support_top,opened,closed,initial,surfaces,points,normals,samples,trims,time.perf_counter()-start,fit,lift_height,approach_height)


class GraspTask:
    periods=dict(APPROACH=10.,ACQUIRE=18.,LOAD=2.,PROBE=3.,LIFT=8.,CARRY=12.,PLACE=10.,UNLOAD=2.,RELEASE=6.)
    timeouts=dict(APPROACH=20.,ACQUIRE=28.,LOAD=6.,PROBE=10.,LIFT=16.,CARRY=22.,PLACE=18.,UNLOAD=6.,RELEASE=14.)
    stages=list(periods)

    def __init__(self,plan,minimum_preload_m=.00015,acquisition_gap_m=CONTACT_QUERY_TOLERANCE,placement_gap_m=.0001,release_load_mode='sequential',release_pose_mode='captured',release_method='legacy',release_executor=None,guard_servo=False,preserve_joint_action=False,allow_guard_braking=False,guard_vectoring=False,measured_thrust=False,preserve_preload=False,posture_tracking=False,input_planner=None,trajectory_feedforward=False,rest_to_rest_retreat=False,coupled_free_flight=False,relative_acquisition=False):
        if not np.isfinite(minimum_preload_m) or minimum_preload_m<0:raise ValueError("minimum preload must be a finite nonnegative distance")
        self.minimum_preload_m=float(minimum_preload_m)
        if not np.isfinite(acquisition_gap_m) or acquisition_gap_m<0:raise ValueError('invalid acquisition proximity')
        self.acquisition_gap_m=float(acquisition_gap_m)
        if placement_gap_m<0 or not np.isfinite(placement_gap_m):raise ValueError('invalid placement gap')
        if release_load_mode not in ['sequential','simultaneous','resting']:raise ValueError('invalid release load mode')
        self.placement_gap_m=float(placement_gap_m);self.release_load_mode=release_load_mode
        if release_pose_mode not in ['captured','task']:raise ValueError('invalid release pose mode')
        self.release_pose_mode=release_pose_mode
        if release_method not in ['legacy','supported']:raise ValueError('invalid release method')
        self.release_method=release_method;self.supported_release=None;self.release_executor=release_executor
        self.guard_servo=guard_servo;self.allow_guard_braking=allow_guard_braking
        self.preserve_preload=preserve_preload
        self.input_planner=input_planner
        self.trajectory_feedforward=trajectory_feedforward
        self.rest_to_rest_retreat=rest_to_rest_retreat
        self.guard_braking_counts={};self.command_guard_checks=0;self.maximum_command_step_ratio=0.
        self.pose_window=deque(maxlen=31)
        self.plan=plan;p=plan.plant
        self.relative_acquisition=bool(relative_acquisition)
        if self.relative_acquisition and release_executor is None:
            raise ValueError('relative acquisition requires a background planning executor')
        self.pickup_pending=None;self.pickup_path_evidence=None
        self.pickup_reference=plan.source.copy()
        self.flight=FlightController(p.pin,p.meta,plan.initial_q,preserve_joint_action=preserve_joint_action,guard_vectoring=guard_vectoring,measured_thrust=measured_thrust)
        self.flight_switch=None
        if coupled_free_flight:
            if input_planner is not None:raise ValueError('free flight switch does not combine with input planner')
            from .free_flight_switch import FreeFlightSwitch
            self.flight_switch=FreeFlightSwitch(plan,self.flight)
            self.flight=self.flight_switch.active
        self.stage='APPROACH';self.stage_start=0.;self.dwell=0.;self.history=[]
        self.grasp=None;self.reason=None;self.finished=False;self.load=0.
        self.load_start=None;self.load_contact_dwell=0.
        self.f0,self.fg,self.f1=(np.array(t['thrust']) for t in plan.trims)
        self.j0,self.jg,self.j1=(np.array([a['effort_nm'] for a in t['joints']]) for t in plan.trims)
        self.acquisition_feedback=False;self.release_start=None
        self.support_release_start=None;self.release_elapsed=0.;self.unload_elapsed=0.
        self.release_object_goal=None
        self.release_duration=None;self.release_timeout=None
        self.release_exit_timeout=None
        self.release_rest_dwell=0.
        self.active_only=False
        self.posture_tracking=posture_tracking
        self.preload_relief=False
        self.drive_bias_offset=np.zeros(len(self.flight.actuators))
        self.operating_transfer=None
        if getattr(plan,'operating_point',None) is not None:
            from .operating_transfer import OperatingTransfer
            self.operating_transfer=OperatingTransfer(plan.operating_point)

    def drive_guard(self):
        if not self.guard_servo:return None
        dock=self.flight.dock
        if self.operating_transfer is not None and self.operating_transfer.started:
            return self.operating_transfer.inputs(self.load)[1][dock],self.flight.command[dock]
        return ((1-self.load)*self.jg+self.load*self.j1+self.drive_bias_offset)[dock],self.flight.command[dock]

    def update(self,time_s,q,v, *, freeze_load_transfer=False,carry_goal=None,carry_complete=None):
        plan=self.plan;p=plan.plant;elapsed=time_s-self.stage_start
        if self.flight_switch is not None:
            self.flight=self.flight_switch.select(self.stage,self.grasp,q,time_s)
        oq,ov=p.meta['object']['q'],p.meta['object']['v'];obj=q[oq:oq+7];object_velocity=v[ov:ov+6]
        self.pose_window.append(obj.copy())
        tv=None;target=plan.closed_q.copy();goal=self.pickup_reference.copy();phase=self.stage
        path_velocity=None
        if self.pickup_pending is not None and self.pickup_pending.done():
            try:self.pickup_path_evidence=self.pickup_pending.result()
            except ValueError as error:
                self.reason='captured_pickup_path_rejected:'+str(error)
                self.failure_evidence=getattr(error,'evidence',None)
                return self.flight.command.copy(),self.flight.force.copy()
            self.pickup_pending=None
        done=False;release_support=None
        previous_reference=self.grasp.reference.copy() if self.grasp and self.operating_transfer else None
        if phase=='APPROACH':
            above=plan.open_q.copy();above[2]+=plan.approach_height
            first_duration,last_duration=plan.fit.get('approach_durations_s',[5.,5.])
            first,last,progress=(plan.initial_q,above,elapsed/first_duration) if elapsed<first_duration else (above,plan.open_q,(elapsed-first_duration)/last_duration)
            target=pin.interpolate(p.pin,first,last,float(smooth(progress)))
            error=pin.difference(p.pin,q,plan.open_q)
            done=elapsed>=first_duration+last_duration and np.linalg.norm(error[:6])<.008 and np.linalg.norm(v[:6])<.02
        elif phase=='ACQUIRE':
            target=pin.interpolate(p.pin,plan.open_q,plan.closed_q,float(smooth(elapsed/18)))
            if self.grasp is None:
                self.grasp=GraspReferences(p.pin,p.meta,q,target,plan.surfaces,plan.points,plan.normals,samples=plan.samples,shape=plan.shape,active_only=self.active_only)
            _,_,_,delta=self.grasp.geometry(q)
            if self.input_planner is None and elapsed>=18 and (not np.all(delta>=-CONTACT_QUERY_TOLERANCE) or self.acquisition_feedback):
                if not self.acquisition_feedback:
                    self.grasp=GraspReferences(p.pin,p.meta,q,target,plan.surfaces,plan.points,plan.normals,samples=plan.samples,shape=plan.shape,active_only=self.active_only)
                    self.grasp.preload=np.maximum(delta,self.minimum_preload_m);self.acquisition_feedback=True
                following_velocity=None
                if self.relative_acquisition:
                    goal[:3]=obj[:3]
                    R=pin.Quaternion(obj[3:]).matrix()
                    following_velocity=np.r_[R@object_velocity[:3],np.zeros(3)]
                target,tv=self.grasp.update(q,v,goal,desired_twist=following_velocity,orientation_feedback=0.,drive_guard=self.drive_guard(),allow_guard_braking=self.allow_guard_braking,previous_joint_bias=(self.flight.last_joint_bias[self.flight.dock] if self.allow_guard_braking else None))
                _,_,_,delta=self.grasp.geometry(q)
            # Near material is permission to try a small lift, not proof of
            # load-bearing contact. PROBE verifies actual unsupported motion.
            done=(elapsed>=18 and np.all(delta>=-self.acquisition_gap_m) and quiet_object(object_velocity)
                  and self.grasp.relative_material_speed(q,v)<=.02)
        elif phase in ['UNLOAD','RELEASE'] and self.release_method=='supported':
            from .release import SupportedRelease
            goal=plan.destination.copy()
            try:
                if self.supported_release is None:
                    self.supported_release=SupportedRelease(plan,self.grasp,self.flight,q,self.release_executor,rest_to_rest=self.rest_to_rest_retreat)
                    self.release_timeout=self.supported_release.duration+4.
                release=self.supported_release
                if phase=='UNLOAD':
                    target,tv,self.load,joint,force=release.unload()
                    done=(release.unloaded_reference_settled and abs(plan.shape.lowest_z(obj)-plan.support_top)<=.0011
                          and np.linalg.norm(obj[:3]-goal[:3])<.03 and quiet_object(object_velocity))
                else:
                    target,tv,self.load,joint,force=release.open(q,v)
                    self.release_exit_timeout=release.opening_timeout
                    done=(release.opening_reference_complete and np.all(self.grasp.geometry(q)[-1]<-.02)
                          and np.linalg.norm(obj[:3]-goal[:3])<.03
                          and quiet_object(object_velocity,angular_limit=.10))
                    # Final completion uses the declared .02 m/s/.10 rad/s
                    # limits, held for two seconds below. The tighter window
                    # is a support-transfer guard, not a second final contract.
                release_support=(joint,force)
            except ValueError as error:
                self.failure_evidence=getattr(error,'evidence',None)
                self.reason='supported_release_not_found:'+str(error)
                return self.flight.command.copy(),self.flight.force.copy()
        else:
            if phase=='LOAD':
                if not freeze_load_transfer:
                    if self.load_start is None:
                        # ACQUIRE admits a small positive gap. Wait for all
                        # observed surfaces before increasing the lift demand;
                        # geometry alone does not certify load-bearing contact.
                        near=np.all(self.grasp.geometry(q)[-1]>=-CONTACT_QUERY_TOLERANCE)
                        ready=(near and quiet_object(object_velocity)
                               and self.grasp.relative_material_speed(q,v)<=.02)
                        self.load_contact_dwell=self.load_contact_dwell+.01 if ready else 0.
                        if self.load_contact_dwell>=.3:self.load_start=time_s
                    self.load=0. if self.load_start is None else smooth((time_s-self.load_start)/2)
                done=(not freeze_load_transfer and self.load_start is not None
                      and time_s-self.load_start>=2 and quiet_object(object_velocity)
                      and (not self.relative_acquisition or self.pickup_path_evidence is not None))
            elif phase=='PROBE':
                # Raise the lowest material point by 5 mm. A small support-edge
                # rotation raises the COM without freeing the object; a fixed
                # COM-only target can therefore stall while still on the table.
                goal[2]=probe_goal_height(plan.shape,self.pickup_reference,obj,plan.support_top,elapsed)
                initial_clearance=plan.shape.lowest_z(plan.source)-plan.support_top
                path_velocity=quintic_path_velocity([0.,0.,PROBE_HEIGHT-initial_clearance],elapsed,3.)
                path_velocity[2]+=observed_extent_rate(plan.shape,obj,v[ov+3:ov+6],self.grasp.dt)
                _,_,_,compression=self.grasp.geometry(q)
                done=elapsed>=3 and probe_acquired(plan.shape,self.pickup_reference,obj,v[ov:ov+6],plan.support_top,compression,self.acquisition_gap_m)
            elif phase=='LIFT':
                clearance=PROBE_HEIGHT+(plan.lift_height-PROBE_HEIGHT)*smooth(elapsed/8)
                goal[2]=object_height_at_clearance(plan.shape,obj[3:],plan.support_top,clearance)
                path_velocity=quintic_path_velocity([0.,0.,plan.lift_height-PROBE_HEIGHT],elapsed,8.)
                done=elapsed>=8 and plan.shape.lowest_z(obj)-plan.support_top>=.10 and np.linalg.norm(obj[:3]-goal[:3])<.006 and quiet_object(object_velocity)
            elif phase=='CARRY':
                goal[:3]=(1-smooth(elapsed/12))*self.pickup_reference[:3]+smooth(elapsed/12)*plan.destination[:3]
                goal[2]=object_height_at_clearance(plan.shape,obj[3:],plan.support_top,plan.lift_height)
                path_velocity=quintic_path_velocity(plan.destination[:3]-self.pickup_reference[:3],elapsed,12.)
                if carry_goal is not None:goal=np.asarray(carry_goal).copy()
                if carry_goal is not None:path_velocity=None
                done=(elapsed>=12 if carry_complete is None else carry_complete) and np.linalg.norm(obj[:3]-goal[:3])<.006 and quiet_object(object_velocity)
            elif phase=='PLACE':
                goal=plan.destination.copy()
                rotated=plan.destination.copy();rotated[3:]=obj[3:]
                # Place the currently oriented object's lowest material above
                # the table. A nominal COM height can push a tilted corner in.
                extent_change=plan.support_top-plan.shape.lowest_z(rotated)
                height,height_rate=placement_height(plan.lift_height,elapsed,extent_change,self.placement_gap_m)
                goal[2]+=height
                path_velocity=np.array([0.,0.,height_rate,0.,0.,0.])
                done=elapsed>=10 and abs(plan.shape.lowest_z(obj)-plan.support_top)<self.placement_gap_m+.0001 and np.linalg.norm(obj[:3]-goal[:3])<.01 and quiet_object(object_velocity,linear_limit=.01)
            elif phase=='UNLOAD':
                goal=(plan.destination if self.release_pose_mode=='task' else self.release_object_goal).copy()
                _,_,_,compression=self.grasp.geometry(q)
                rotation_error=np.linalg.norm(pin.log3(pin.Quaternion(goal[3:]).matrix()@pin.Quaternion(obj[3:]).matrix().T))
                position_limit,angle_limit=(.03,.2) if self.release_pose_mode=='task' else (.01,.1)
                advance=(not freeze_load_transfer and np.linalg.norm(obj[:3]-goal[:3])<position_limit
                         and rotation_error<angle_limit and quiet_object(object_velocity))
                if self.release_pose_mode=='task':
                    advance=advance and abs(plan.shape.lowest_z(obj)-plan.support_top)<=self.placement_gap_m+.0001
                if advance:self.release_elapsed+=.01
                # The default waits for pad separation. The resting comparison
                # instead starts the bounded transfer after near-table rest,
                # while contacts may remain. Neither observes support force.
                separated=np.all(compression<-.0005)
                resting=abs(plan.shape.lowest_z(obj)-plan.support_top)<.001 and quiet_object(object_velocity)
                near_rest=(abs(plan.shape.lowest_z(obj)-plan.support_top)<.0002 and advance
                           and bounded_pose_window(self.pose_window))
                self.release_rest_dwell=self.release_rest_dwell+.01 if near_rest else 0.
                transfer_ready=(self.release_rest_dwell>=.3 if self.release_load_mode=='resting' else
                                separated and resting and bounded_pose_window(self.pose_window))
                if not freeze_load_transfer and self.support_release_start is None and elapsed>=2 and transfer_ready:
                    self.support_release_start=elapsed
                if advance and self.support_release_start is not None:self.unload_elapsed+=.01
                progress=self.unload_elapsed/2
                if self.release_load_mode=='simultaneous':
                    duration=self.release_duration or max(2.5,1.875*(float(self.grasp.preload.max())+.002)/.001)
                    progress=self.release_elapsed/duration
                if not freeze_load_transfer:self.load=1-smooth(progress)
                done=not freeze_load_transfer and progress>=1 and separated and np.linalg.norm(obj[:3]-goal[:3])<(.03 if self.release_pose_mode=='task' else .02) and quiet_object(object_velocity)
            elif phase=='RELEASE':
                goal=plan.destination.copy();_,_,_,delta=self.grasp.geometry(q)
                # Linear and angular speeds have different units and declared
                # limits. Do not compare their concatenated norm with m/s.
                done=(np.all(delta<-.02) and np.linalg.norm(obj[:3]-goal[:3])<.03
                      and np.linalg.norm(v[ov:ov+3])<.02 and np.linalg.norm(v[ov+3:ov+6])<.1)
            if phase=='RELEASE':
                # Continue outward from the actual material geometry. Returning
                # to a nominal open posture can sweep a pad through an object
                # whose placed pose differs from the nominal acquisition pose.
                target,tv=self.grasp.update(q,v,goal,preload=np.full(len(self.grasp.preload),-.03),
                    normal_speed=.006,object_feedback=0.,orientation_feedback=0.,releasing=True)
            elif self.input_planner is not None and phase not in ['UNLOAD','RELEASE']:
                # The alternate input planner keeps an explicit finite shape
                # reference. It does not integrate the geometric velocity law.
                target[:3]+=goal[:3]-plan.source[:3]
            else:
                # Transfer support while relaxing the squeeze. Keeping the full
                # grasp preload while removing lift makes the robot press the
                # object into its destination support. The full rollout still
                # checks placement, contact loads and the final settled pose.
                release_preload=None
                release_progress=0.
                if phase=='UNLOAD' and not freeze_load_transfer:
                    duration=self.release_duration or max(2.5,1.875*(float(self.grasp.preload.max())+.002)/.001)
                    release_progress=smooth(self.release_elapsed/duration)
                    release_preload=(1-release_progress)*self.grasp.preload-release_progress*.002
                if self.operating_transfer is not None and self.operating_transfer.started and phase not in ['UNLOAD','RELEASE']:
                    self.grasp.preload=self.operating_transfer.inputs(self.load)[0]
                reference_update=self.grasp.update
                if self.preload_relief and phase in ['LOAD','PROBE','LIFT','CARRY','PLACE']:
                    from functools import partial
                    from .preload_relief import update_with_preload_relief
                    reference_update=partial(update_with_preload_relief,self.grasp)
                target,tv=reference_update(q,v,goal,
                    drive_guard=self.drive_guard(),
                    # Retain object/posture feedback and recover only missing
                    # compression during carry; do not raise the preload target.
                    recover_compression_only=phase=='CARRY',
                    allow_guard_braking=self.allow_guard_braking,previous_joint_bias=(self.flight.last_joint_bias[self.flight.dock] if self.allow_guard_braking or self.preload_relief else None),
                    preload=release_preload,
                    preserve_preload=self.preserve_preload and phase in ['LOAD','PROBE','LIFT','CARRY','PLACE'],
                    object_feedback=1-release_progress,
                    releasing=phase=='UNLOAD',
                    desired_twist=(np.array([0.,0.,-.001*smooth((plan.shape.lowest_z(obj)-plan.support_top-self.placement_gap_m)/.0001),0.,0.,0.]) if phase=='PLACE' and elapsed>=10 else
                                   path_velocity if self.trajectory_feedforward else None),
                    orientation_feedback=free_object_orientation_gain(plan.shape.lowest_z(obj)-plan.support_top) if phase in ['LOAD','PROBE','LIFT','PLACE','UNLOAD'] else 1.)
        if self.grasp and not self.grasp.last_diagnostics.get('preload_reference_feasible',True):
            self.reason='preload_motion_not_found'
            return self.flight.command.copy(),self.flight.force.copy()
        if self.grasp and not self.grasp.last_diagnostics.get('reference_command_feasible',True):
            self.reason='reference_command_bounds_incompatible'
            return self.flight.command.copy(),self.flight.force.copy()
        closing=smooth(elapsed/18) if phase=='ACQUIRE' else (0. if phase=='APPROACH' else 1.)
        if phase=='RELEASE':closing=1-smooth(elapsed/6)
        unloaded_joint=(1-closing)*self.j0+closing*self.jg
        unloaded_force=(1-closing)*self.f0+closing*self.fg
        joint_bias,force_reference=((1-self.load)*unloaded_joint+self.load*self.j1,
                                    (1-self.load)*unloaded_force+self.load*self.f1) if release_support is None else release_support
        gimbal_reference=(plan.closed_q if release_support is None else target)[self.flight.qi[self.flight.ai]]
        if self.operating_transfer is not None and self.operating_transfer.started and phase not in ['UNLOAD','RELEASE']:
            _,joint_bias,force_reference,gimbal_reference=self.operating_transfer.inputs(self.load)
            try:
                target,tv=self.operating_transfer.compose(q,previous_reference,target,tv,self.load,self.flight.dt)
            except ValueError as error:
                self.reason=str(error);return self.flight.command.copy(),self.flight.force.copy()
        guard_check=(self.grasp.last_diagnostics if self.allow_guard_braking and self.grasp and tv is not None and release_support is None else {})
        if 'command_step_limits' in guard_check:previous_command,previous_force=self.flight.command.copy(),self.flight.force.copy()
        if self.input_planner is not None and phase not in ['UNLOAD','RELEASE']:
            cmd,f,diagnostic=self.input_planner.update(self,time_s,q,v,target,goal,joint_bias,force_reference)
            self.flight.command=cmd.copy();self.flight.force=f.copy()
            if not diagnostic['usable']:
                self.reason='input_plan_not_found';self.failure_evidence=diagnostic
                return cmd,f
        else:
            cmd,f=self.flight.update(q,v,target,target_velocity=tv,payload_mass=self.load*plan.shape.mass,
                    payload_com=obj[:3],joint_bias=joint_bias+self.drive_bias_offset,thrust_reference=force_reference,
                    gimbal_reference=gimbal_reference)
        if getattr(self.flight,'allocation_failure',None):
            self.reason=self.flight.allocation_failure
            if self.flight.requested_action_diagnostics is not None:
                self.failure_evidence=self.flight.requested_action_diagnostics.copy()
            return cmd,f
        if self.flight_switch is not None and not self.flight_switch.check(q,v):
            self.reason='free_flight_predicted_clearance'
            self.failure_evidence=self.flight_switch.last_forecast
            return cmd,f
        if 'command_step_limits' in guard_check:
            dock=self.flight.dock;raw=self.flight.kp[dock]*(cmd[dock]-q[self.flight.qi[dock]])-self.flight.kd[dock]*v[self.flight.vi[dock]]
            valid=(np.all(abs(cmd[dock]-guard_check['previous_commands'])<=np.array(guard_check['command_step_limits'])+1e-9)
                   and np.all(raw>=np.array(guard_check['drive_request_lower'])-1e-9) and np.all(raw<=np.array(guard_check['drive_request_upper'])+1e-9))
            guard_check['issued_command_guard_feasible']=bool(valid)
            self.command_guard_checks+=1
            self.maximum_command_step_ratio=max(self.maximum_command_step_ratio,float(np.max(abs(cmd[dock]-guard_check['previous_commands'])/guard_check['command_step_limits'])))
            for name in guard_check.get('reference_acceleration_relaxed_docks',[]):self.guard_braking_counts[name]=self.guard_braking_counts.get(name,0)+1
            if not valid:
                self.reason='issued_command_guard_violation';self.flight.command=previous_command;self.flight.force=previous_force
                return previous_command,previous_force
        if phase in ['PLACE','UNLOAD','RELEASE']:
            rotation_error=np.linalg.norm(pin.log3(pin.Quaternion(goal[3:]).matrix()@pin.Quaternion(obj[3:]).matrix().T))
            done=done and rotation_error<=.20
        if phase in ['LOAD','PLACE'] or (phase=='UNLOAD' and self.release_method!='supported'):
            done=done and bounded_pose_window(self.pose_window)
        self.dwell=self.dwell+.01 if done else 0.
        required=2. if phase=='RELEASE' else (.5 if phase=='PROBE' else .3)
        if self.dwell>=required:
            self.history.append(dict(stage=phase,end=time_s,duration=elapsed))
            i=self.stages.index(phase)+1
            if i==len(self.stages):self.finished=True
            else:
                self.stage=self.stages[i];self.stage_start=time_s;self.dwell=0.;self.pose_window.clear()
                if phase=='UNLOAD':self.release_start=target.copy()
                if phase=='PLACE':
                    self.release_object_goal=obj.copy()
                    if self.operating_transfer is not None:self.operating_transfer.release(self.grasp,target,tv)
                if phase=='PLACE' and self.input_planner is not None:
                    # Start supported release at the observed finite shape,
                    # preserving the issued commands for its continuity check.
                    self.grasp=GraspReferences(p.pin,p.meta,q,q.copy(),plan.surfaces,plan.points,plan.normals,samples=plan.samples,shape=plan.shape)
                if phase=='ACQUIRE':
                    if self.relative_acquisition:
                        # Capture once after acquisition. Source geometry and the
                        # final destination remain immutable; only the horizontal
                        # start of the observed grasp changes.
                        self.pickup_reference[:2]=obj[:2]
                        from .pickup import check_pickup_path
                        obstacles=[(p.mj.geom(name).pos.copy(),p.mj.geom(name).size.copy())
                                   for name in ['support','destination_support']]
                        self.pickup_pending=self.release_executor.submit(check_pickup_path,
                            p.meta,plan.shape,plan.surfaces,plan.closed_q.copy(),plan.source.copy(),
                            self.pickup_reference.copy(),plan.destination.copy(),plan.lift_height,obstacles)
                    self.grasp=GraspReferences(p.pin,p.meta,q,target,plan.surfaces,plan.points,plan.normals,samples=plan.samples,shape=plan.shape,active_only=self.active_only,posture_tracking=self.posture_tracking)
                    if self.operating_transfer is None:
                        self.grasp.preload=np.maximum(self.grasp.preload,self.minimum_preload_m)
                    else:
                        self.operating_transfer.start(self.grasp,self.flight,q,tv)
                        if plan.operating_point.get('reference_mode')=='rigid':
                            from .rigid_grasp_reference import RigidGraspReferences
                            previous=self.grasp
                            self.grasp=RigidGraspReferences(p.pin,p.meta,q,previous.reference,plan.surfaces,plan.points,plan.normals,samples=plan.samples,shape=plan.shape)
                            self.grasp.last_velocity=previous.last_velocity.copy();self.grasp.preload=previous.preload.copy()
                            self.grasp.configure_operating_point(plan.operating_point)
        if self.grasp and not self.grasp.last_diagnostics.get('release_direction_feasible',True):self.reason='release_motion_not_found'
        timeout=self.release_timeout if phase=='UNLOAD' and self.release_timeout is not None else self.timeouts[phase]
        if phase=='RELEASE' and self.release_exit_timeout is not None:timeout=self.release_exit_timeout
        if phase=='RELEASE' and self.supported_release is not None:
            # The motion deadline is derived from the computed path. The old
            # fixed 14 s motion deadline must not expire while no path exists.
            # Physical monitoring and the outer execution horizon still apply.
            timeout=self.supported_release.opening_timeout
        if phase=='APPROACH':timeout=max(timeout,sum(plan.fit.get('approach_durations_s',[5.,5.]))+10.)
        if timeout is not None and elapsed>timeout:self.reason='phase_timeout:'+phase
        return cmd,f
