"""MuJoCo dynamics with source inertias, movable joints and actual rotor sites."""
from __future__ import annotations
from dataclasses import dataclass, replace
import hashlib
from functools import lru_cache
from copy import deepcopy
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import pinocchio as pin
import mujoco
from .model import ROOT, build_model, transform
from .planning_memo import evaluate as memoized

# Source PhysX Dock decomposition; other parts retain the CAD component hulls.
# This is a reusable part cache, not a catalogue of assembled structures.
CONTACT_CACHE=ROOT/'build/collision_physx_docks'


def contact_dynamics_model(plant,selected_links, *, compact_frames=True):
    """Omit prohibited collision pairs; callers MUST audit full geometry.

    This is equivalent only until the full clearance boundary is crossed.
    Body inertias, joints, actuators and all permitted task contacts stay.
    """
    doc=ET.fromstring(plant.xml)
    for body in doc.find('worldbody').iter():
        for geom in list(body.findall('geom')):
            name=geom.get('name','')
            if name not in ['object','support','destination_support','floor'] and not any(name.startswith(link+'/') for link in selected_links):
                body.remove(geom)
    if compact_frames:
        world=doc.find('worldbody')
        used_sites={value for element in doc if element.tag!='worldbody' for node in element.iter()
                    for key,value in node.attrib.items() if key in ['site','site1','site2','refsite']}
        for parent in world.iter():
            for site in list(parent.findall('site')):
                if site.get('name') not in used_sites:parent.remove(site)
        removed=set()
        for parent in list(world.iter())[::-1]:
            for body in list(parent.findall('body')):
                if len(body)==0:
                    original=plant.mj.body(body.get('name')).id
                    if plant.mj.body_mass[original] or np.any(plant.mj.body_inertia[original]):
                        raise ValueError('cannot remove a physical body')
                    removed.add(body.get('name'));parent.remove(body)
        for contact in doc.findall('contact'):
            for exclude in list(contact.findall('exclude')):
                if exclude.get('body1') in removed or exclude.get('body2') in removed:contact.remove(exclude)
    xml=ET.tostring(doc,encoding='unicode');model=mujoco.MjModel.from_xml_string(xml)
    model.opt.timestep=plant.mj.opt.timestep
    for name in ['nq','nv','nu','na','neq','nmocap','nuserdata']:
        if getattr(model,name)!=getattr(plant.mj,name):raise ValueError('collision pruning changed '+name)
    original_bodies=[plant.mj.body(model.body(i).name).id for i in range(model.nbody)]
    for name in ['body_mass','body_inertia','body_ipos','body_iquat','body_pos','body_quat']:
        if not np.array_equal(getattr(model,name),getattr(plant.mj,name)[original_bodies]):
            raise ValueError('frame pruning changed '+name)
    for name in ['jnt_type','jnt_qposadr','jnt_dofadr','jnt_axis','jnt_range','dof_armature',
                 'actuator_gear','actuator_gainprm','actuator_biasprm','actuator_forcerange']:
        if not np.array_equal(getattr(model,name),getattr(plant.mj,name)):
            raise ValueError('collision pruning changed '+name)
    joints=[(j,model.joint(plant.pin.names[j]).id) for j in range(1,plant.pin.njoints)]
    return replace(plant,mj=model,xml=xml,joints=joints)


def numbers(x):
    return ' '.join(format(float(v), '.17g') for v in np.asarray(x).ravel())


def pose(T):
    xyzw = pin.Quaternion(T.rotation).coeffs()
    return dict(pos=numbers(T.translation), quat=numbers(xyzw[[3, 0, 1, 2]]))


@dataclass
class PlantModel:
    pin: object
    meta: dict
    mj: object
    xml: str
    joints: list

    @property
    def coordinate_maps(self):
        """Cache the fixed correspondence between the two model conventions."""
        if getattr(self, '_coordinate_models', None) != (id(self.pin), id(self.mj)):
            pq, mq, pv, mv, bases = [], [], [], [], []
            for jid, mjid in self.joints:
                joint = self.pin.joints[jid]
                dest, vel = self.mj.jnt_qposadr[mjid], self.mj.jnt_dofadr[mjid]
                if joint.nq == 7:
                    pq.extend(joint.idx_q + np.array([0, 1, 2, 6, 3, 4, 5]))
                    mq.extend(range(dest, dest + 7))
                    pv.extend(range(joint.idx_v, joint.idx_v + 6))
                    mv.extend(range(vel, vel + 6))
                    bases.append((joint.idx_q, joint.idx_v, vel))
                else:
                    pq.append(joint.idx_q); mq.append(dest)
                    pv.append(joint.idx_v); mv.append(vel)
            self._coordinate_maps = tuple(np.asarray(x, dtype=int) for x in [pq, mq, pv, mv]) + (bases,)
            self._coordinate_models = (id(self.pin), id(self.mj))
        return self._coordinate_maps

    def configuration(self, q):
        """Convert coordinates independently of compiler joint order."""
        pq, mq, _, _, _ = self.coordinate_maps
        result = self.mj.qpos0.copy()
        result[mq] = np.asarray(q)[pq]
        return result

    def observation(self, data):
        pq, mq, pv, mv, bases = self.coordinate_maps
        q = pin.neutral(self.pin)
        v = np.zeros(self.pin.nv)
        q[pq] = data.qpos[mq]
        v[pv] = data.qvel[mv]
        for pos, vel, source in bases:
            R = pin.Quaternion(q[pos+3:pos+7]).matrix()
            v[vel:vel+3] = R.T @ data.qvel[source:source+3]
        return q, v


def make_plant(n=1, edges=(), payload=None, *, timestep=.001,
               meshes=True, mesh_cache=None, floor=False):
    # Different postures of the same generated structure share this immutable
    # construction within one planning call. Loading returns an independent
    # model: later option, support and actuator changes must not leak.
    inputs=dict(operation='make_plant',modules=n,edges=edges,payload=payload,
                timestep=timestep,meshes=meshes,
                mesh_cache=None if mesh_cache is None else str(Path(mesh_cache).resolve()),floor=floor)
    return memoized(inputs,lambda:_make_plant(n,edges,payload,timestep=timestep,
                    meshes=meshes,mesh_cache=mesh_cache,floor=floor))


def _make_plant(n,edges,payload,*,timestep,meshes,mesh_cache,floor):
    model, meta = build_model(n, edges, payload)
    meta['material_mesh_cache']=str(Path(mesh_cache or ROOT/'build/collision_meshes').resolve()) if meshes else None
    doc = ET.Element('mujoco', model='amsrr_det')
    ET.SubElement(doc, 'compiler', angle='radian', inertiafromgeom='false',
                  autolimits='true', fusestatic='false')
    option = ET.SubElement(doc, 'option', timestep=str(timestep),
                          gravity=numbers(model.gravity.linear), integrator='implicitfast',
                          cone='elliptic', iterations='60', tolerance='1e-10')
    ET.SubElement(option, 'flag', filterparent='disable', autoreset='disable')
    default = ET.SubElement(doc, 'default')
    ET.SubElement(default, 'geom', friction='.6 0 0', condim='3',
                  solref='.01 1', solimp='.95 .95 .001')
    assets = ET.SubElement(doc, 'asset')
    world = ET.SubElement(doc, 'worldbody')
    ET.SubElement(world, 'light', pos='0 0 4', dir='0 0 -1')
    bodies = {0: world}
    actuators = ET.SubElement(doc, 'actuator')
    contact = ET.SubElement(doc, 'contact')
    data = model.createData()
    neutral = pin.neutral(model)
    byjoint = {a['joint']: a for a in meta['actuators']}
    for jid in range(1, model.njoints):
        j = model.joints[jid]
        b = ET.SubElement(bodies[model.parents[jid]], 'body', name=f'joint{jid}',
                          **pose(model.jointPlacements[jid]))
        bodies[jid] = b
        inertia = model.inertias[jid]
        I = inertia.inertia
        ET.SubElement(b, 'inertial', mass=str(inertia.mass), pos=numbers(inertia.lever),
                      fullinertia=numbers([I[0,0],I[1,1],I[2,2],I[0,1],I[0,2],I[1,2]]))
        if j.nq == 7:
            ET.SubElement(b, 'freejoint', name=model.names[jid])
        else:
            a = byjoint[jid]
            axis = pin.computeJointJacobian(model, data, neutral, jid)[3:, j.idx_v]
            ET.SubElement(b, 'joint', name=a['name'], type='hinge', axis=numbers(axis),
                          range=numbers([a['lower'], a['upper']]),
                          armature=str(model.armature[j.idx_v]))
    frame_bodies = {}
    for name, fid in meta['frames'].items():
        frame = model.frames[fid]
        b = ET.SubElement(bodies[frame.parentJoint], 'body', name=name,
                          **pose(frame.placement))
        ET.SubElement(b, 'site', name=name, size='.002', rgba='0 0 0 0')
        frame_bodies[name] = b
    for a in meta['actuators']:
        ET.SubElement(actuators, 'general', name=a['name'], joint=a['name'],
                      gaintype='fixed', gainprm=str(a['kp']), biastype='affine',
                      biasprm=numbers([0, -a['kp'], -a['kd']]),
                      forcerange=numbers([-a['effort'], a['effort']]))
    for i, r in enumerate(meta['rotors']):
        ET.SubElement(actuators, 'general', name=f'rotor{i}',
                      site=model.frames[r['frame']].name, gear='0 0 1 0 0 0',
                      ctrlrange=numbers([r['minimum'],r['maximum']]))
    if meshes:
        add_meshes(assets, frame_bodies, contact, meta, mesh_cache)
    if payload is not None:
        family=payload.get('shape','box');kind='box' if family=='elongated_box' else family
        if family in ['triangular_prism','tapered_box']:
            ET.SubElement(assets,'mesh',name='payload_shape',vertex=numbers(payload['vertices']),
                          face=' '.join(str(int(i)) for i in np.asarray(payload['faces']).ravel()))
            geometry=dict(type='mesh',mesh='payload_shape')
        elif family=='cylinder':
            geometry=dict(type=kind,size=numbers(np.asarray(payload['size'])[[0,2]]/2))
        else:
            geometry=dict(type=kind,size=numbers(np.asarray(payload['size'])/2))
        ET.SubElement(frame_bodies['object'],'geom',name='object',rgba='.8 .4 .15 1',**geometry)
    if floor:
        ET.SubElement(world, 'geom', name='floor', type='plane', size='10 10 .1',
                      friction='.8 0 0', rgba='.8 .8 .8 1')
    xml = ET.tostring(doc, encoding='unicode')
    mj = mujoco.MjModel.from_xml_string(xml)
    joints = [(i, mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_JOINT, model.names[i]))
              for i in range(1, model.njoints)]
    return PlantModel(model, meta, mj, xml, joints)


@lru_cache(maxsize=4)
def _part_templates(cache):
    """Verified component data, shared for one frozen-source planning run.

    This caches individual CAD parts only, never assembled structures. Source
    identity is checked when loading; callers freeze assets during a run.
    """
    cache=Path(cache)
    source=ROOT/'assets/robots/holon/holon.urdf'
    tree=ET.parse(source);assets=ET.Element('asset')
    unique,links,geometries={},[],[]
    for link in tree.findall('link'):
        for collider in link.findall('collision'):
            mesh=collider.find('geometry/mesh');path=source.parent/mesh.get('filename')
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            scale=np.fromstring(mesh.get('scale','1 1 1'),sep=' ')
            if digest not in unique:
                manifest=json.loads((cache/digest/'manifest.json').read_text())
                if manifest['source_sha256']!=digest:raise ValueError('stale collision geometry')
                names=[]
                for k,part in enumerate(manifest['parts']):
                    file=(cache/digest/part['file']).resolve()
                    if hashlib.sha256(file.read_bytes()).hexdigest()!=part['sha256']:
                        raise ValueError('collision geometry hash mismatch')
                    name=f'mesh{len(unique)}_{k}'
                    ET.SubElement(assets,'mesh',name=name,file=str(file),scale=numbers(scale))
                    names.append(name)
                unique[digest]=names
            name=link.get('name');links.append(name)
            geometries.append((name,[ET.Element('geom',name=f'{name}/{k}',type='mesh',mesh=asset,
                **pose(transform(collider.find('origin'))),rgba='0.25 0.5 0.75 1')
                for k,asset in enumerate(unique[digest])]))
    manifest=json.loads((ROOT/'configs/robot/holon_nominal_collision_pairs_v1.json').read_text())
    if hashlib.sha256(source.read_bytes()).hexdigest()!=manifest['urdf_sha256']:
        raise ValueError('stale exclusion manifest')
    adjacent={frozenset([j.find('parent').get('link'),j.find('child').get('link')])for j in tree.findall('joint')}
    exclusions=[(a,b)for i,a in enumerate(links)for j,b in enumerate(links[i+1:],i+1)
        if frozenset([a,b])in adjacent or [i,j]in manifest['authored_overlap_geometry_pairs']]
    return assets,geometries,exclusions


def add_meshes(assets,frame_bodies,contact,meta,mesh_cache):
    parts,geometries,exclusions=_part_templates(str(Path(mesh_cache or ROOT/'build/collision_meshes').resolve()))
    assets.extend(deepcopy(list(parts)))
    # Preserve the previous geometry ordering: source collider, module, hull.
    for link,geoms in geometries:
        for module in range(meta['modules']):
            name=f'm{module}/{link}'
            for prototype in geoms:
                geom=deepcopy(prototype);geom.set('name',f'm{module}/'+prototype.get('name'))
                frame_bodies[name].append(geom)
    for module in range(meta['modules']):
        for a,b in exclusions:ET.SubElement(contact,'exclude',body1=f'm{module}/{a}',body2=f'm{module}/{b}')
    for e in meta['edges']:
        mechanism=lambda p:('pitch_dock_mech'if p.startswith('pitch')else'yaw_dock_mech')+p[-1]
        ET.SubElement(contact,'exclude',body1=f"m{e['a']}/{mechanism(e['port_a'])}",
                      body2=f"m{e['b']}/{mechanism(e['port_b'])}")


def coat_grippers(xml, prefixes, *, stiffness=7500., damping=75., friction=4.5,
                  impedance=.01, reference_model=None):
    """Map the declared per-contact coating K/C to MuJoCo soft constraints.

    With constant impedance d, R=(1-d)/d * diagA, k=K*R, b=C*R.
    On the quadratic, unsaturated friction branch, the normal response is
    F=K*penetration-C*velocity-a/R (and F=K*penetration at rest).
    The acceleration term is MuJoCo regularization, not physical coating;
    low d reduces it. On the elliptic cone boundary, normal and tangential
    responses are coupled, so that independent Kelvin law is not valid.
    Verification must include branch, timestep and model sensitivity.
    Coefficients use compiled geometry/body masses, never contact-force truth.
    This conversion is for the selected gripper against the task object.
    A supplied reference model must have the same moving bodies, inertias and
    reference configuration. Adding world-fixed support geometry is allowed.
    """
    doc=ET.fromstring(xml)
    m=mujoco.MjModel.from_xml_string(xml) if reference_model is None else reference_model
    obj=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,'object')
    for geom in doc.findall('.//geom'):
        name=geom.get('name','')
        if not any(name.startswith(prefix+'/') for prefix in prefixes):continue
        gid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,name)
        diag=sum(m.body_invweight0[m.geom_bodyid[g],0] for g in [gid,obj])
        geom.set('solref',numbers([-stiffness*(1-impedance)*diag,
                                  -damping*(1-impedance)*diag]))
        geom.set('solimp',numbers([impedance,impedance,.001]))
        geom.set('friction',numbers([friction,0,0]));geom.set('priority','1')
    return ET.tostring(doc,encoding='unicode')
