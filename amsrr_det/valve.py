"""Passive hinged valve and its observation adapter for shared manipulation.

Only the physical model is integrated. The free-body control representation
retains existing robot kinematics; its object wrench is projected onto the
hinge's allowed twist. No force sensor or simulated reaction is an input.
"""
from dataclasses import dataclass, replace
import copy
import numpy as np
import mujoco
from lxml import etree as ET
from .mujoco_model import make_plant, numbers, CONTACT_CACHE
from .objects import ObjectShape


@dataclass(frozen=True)
class ValveSpec:
    center: tuple = (0., 0., 1.5)
    axis: tuple = (0., 0., 1.)
    initial_angle: float = 0.
    target_angle: float = np.pi
    torque: float = 1.5
    damping: float = .05
    length: float = .9
    width: float = .05
    mass: float = 1.2

    def __post_init__(self):
        if (not np.isfinite([*self.center,*self.axis,self.initial_angle,self.target_angle,
                             self.torque,self.damping,self.length,self.width,self.mass]).all()
                or not np.isclose(np.linalg.norm(self.axis),1.)
                or min(self.torque,self.length,self.width,self.mass)<=0 or self.damping<0):
            raise ValueError('invalid passive valve specification')

    @property
    def rotation(self):
        z=np.asarray(self.axis,dtype=float)
        x=np.eye(3)[np.argmin(abs(z))];x=x-z*(x@z);x/=np.linalg.norm(x)
        y=np.cross(z,x);a=self.initial_angle
        return np.column_stack([np.cos(a)*x+np.sin(a)*y,-np.sin(a)*x+np.cos(a)*y,z])

    def pose(self,angle=0.):
        c,s=np.cos(angle),np.sin(angle)
        R=self.rotation@np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
        quat=np.empty(4);mujoco.mju_mat2Quat(quat,R.ravel())
        return np.r_[self.center,quat]

    @property
    def shape(self):
        return ObjectShape('box',[self.length,self.width,self.width],self.mass)

    @property
    def payload(self):
        # A centred bar plus a balanced wheel. The COM is on the bearing axis,
        # so gravity contributes no torque about it at any mounting angle.
        bar_mass=.65*self.mass;wheel_mass=.35*self.mass;radius=.15
        inertia=bar_mass/12*np.array([2*self.width**2,self.length**2+self.width**2,
                                     self.length**2+self.width**2])
        inertia+=wheel_mass*np.array([radius**2/2,radius**2/2,radius**2])
        com_z=-.055*wheel_mass/self.mass
        inertia[:2]+=bar_mass*com_z**2+wheel_mass*(-.055-com_z)**2
        return dict(shape='box',size=[self.length,self.width,self.width],mass=self.mass,
                    com=[0.,0.,com_z],inertia=np.diag(inertia).tolist())


def _environment(doc,spec):
    """Rigid balanced handle and fixed housing, with collision geometry."""
    obj=doc.find(".//body[@name='object']")
    for i in range(16):
        a,b=2*np.pi*np.array([i,i+1])/16
        endpoints=np.r_[.15*np.cos(a),.15*np.sin(a),-.055,
                        .15*np.cos(b),.15*np.sin(b),-.055]
        ET.SubElement(obj,'geom',name=f'valve_wheel/{i}',type='capsule',
                      fromto=numbers(endpoints),size='.009',rgba='.3 .3 .3 1')
    world=doc.find('worldbody');pose=spec.pose()
    housing=ET.SubElement(world,'body',name='valve_mount',pos=numbers(pose[:3]),quat=numbers(pose[3:]))
    ET.SubElement(housing,'geom',name='valve_shaft',type='cylinder',pos='0 0 -.17',size='.025 .10',rgba='.2 .2 .2 1')
    ET.SubElement(housing,'geom',name='valve_housing',type='cylinder',pos='0 0 -.31',size='.065 .10',rgba='.2 .2 .2 1')


class ValvePlant:
    """Actual passive hinge plus a pose-only adapter for the common controller."""
    def __init__(self,modules,edges,spec,*,meshes=True,timestep=.000125):
        from .planning_memo import evaluate
        def create():
            self._initialize(modules,edges,spec,meshes=meshes,timestep=timestep)
            return self.__dict__.copy()
        self.__dict__.update(evaluate(dict(operation='valve_models',modules=modules,edges=edges,
            specification=spec.__dict__,meshes=meshes,timestep=timestep),create))

    def _initialize(self,modules,edges,spec,*,meshes=True,timestep=.000125):
        self.spec=spec
        free=make_plant(modules,edges,spec.payload,meshes=meshes,
                        mesh_cache=CONTACT_CACHE if meshes else None,timestep=timestep,floor=True,compile_model=False)
        doc=ET.fromstring(free.xml);_environment(doc,spec)
        xml=ET.tostring(doc,encoding='unicode')
        control=mujoco.MjModel.from_xml_string(xml)
        joints=[(i,control.joint(free.pin.names[i]).id) for i in range(1,free.pin.njoints)]
        self.control=replace(free,mj=control,xml=xml,joints=joints)
        node=doc.find(".//freejoint[@name='object']");body=node.getparent()
        body.remove(node);pose=spec.pose()
        body.set('pos',numbers(pose[:3]));body.set('quat',numbers(pose[3:]))
        ET.SubElement(body,'joint',name='object',type='hinge',axis='0 0 1',
                      limited='false',frictionloss=str(spec.torque),damping=str(spec.damping),
                      solreffriction='.001 1',solimpfriction='.9999 .9999 .001')
        self.xml=ET.tostring(doc,encoding='unicode')
        self.m=mujoco.MjModel.from_xml_string(self.xml)
        self.oj=self.m.joint('object').id
        self.oi=int(self.m.jnt_qposadr[self.oj]);self.ov=int(self.m.jnt_dofadr[self.oj])
        cm=self.control.mj;cj=cm.joint('object').id
        self.coi=int(cm.jnt_qposadr[cj]);self.cov=int(cm.jnt_dofadr[cj])
        pairs=[];speeds=[]
        for j in range(self.m.njnt):
            if j==self.oj:continue
            cj=cm.joint(self.m.joint(j).name).id
            nq,nv=(7,6) if self.m.jnt_type[j]==mujoco.mjtJoint.mjJNT_FREE else (1,1)
            pairs.extend(zip(range(self.m.jnt_qposadr[j],self.m.jnt_qposadr[j]+nq),
                             range(cm.jnt_qposadr[cj],cm.jnt_qposadr[cj]+nq)))
            speeds.extend(zip(range(self.m.jnt_dofadr[j],self.m.jnt_dofadr[j]+nv),
                              range(cm.jnt_dofadr[cj],cm.jnt_dofadr[cj]+nv)))
        self.pq,self.cq=np.asarray(pairs,dtype=int).T
        self.pv,self.cv=np.asarray(speeds,dtype=int).T

    def configure_contacts(self,links):
        from .planning_memo import evaluate
        def configure():
            self._configure_contacts(links)
            return self.__dict__.copy()
        self.__dict__.update(evaluate(dict(operation='valve_contact_models',xml=self.xml,links=links),configure))

    def _configure_contacts(self,links):
        """Use the same authored MIT drives and pad coating as grasping."""
        from .mujoco_model import coat_grippers
        from .collision_policy import configure
        from build.mit_drive import dock_mit_model
        control=dock_mit_model(self.control,kp=200.)
        physical=ET.fromstring(self.xml)
        for a in control.meta['actuators']:
            if a['role']!='dock':continue
            node=physical.find(f"actuator/general[@name='{a['name']}']")
            node.set('gainprm','1');node.set('biasprm',numbers([0.,-a['kp'],-a['kd']]))
        physical=ET.fromstring(coat_grippers(ET.tostring(physical,encoding='unicode'),links,reference_model=self.m))
        shared=ET.fromstring(control.xml)
        shared_geoms={node.get('name'):node for node in shared.findall('.//geom')}
        for node in physical.findall('.//geom'):
            name=node.get('name','')
            if not any(name.startswith(link+'/') for link in links):continue
            other=shared_geoms[name]
            for field in ['solref','solimp','friction','priority']:other.set(field,node.get(field))
        # Coating and affine motor coefficients do not change geometry,
        # inertia or joint ordering. Update independent compiled copies;
        # retain the same XML for independent reconstruction and auditing.
        self.xml=ET.tostring(physical,encoding='unicode');self.m=copy.copy(self.m)
        for actuator in control.meta['actuators']:
            if actuator['role']!='dock':continue
            i=self.m.actuator(actuator['name']).id
            self.m.actuator_gainprm[i]=0.;self.m.actuator_gainprm[i,0]=1.
            self.m.actuator_biasprm[i]=0.;self.m.actuator_biasprm[i,:3]=[0.,-actuator['kp'],-actuator['kd']]
        xml=ET.tostring(shared,encoding='unicode')
        self.control=replace(control,xml=xml)
        for node in physical.findall('.//geom'):
            name=node.get('name','')
            if not any(name.startswith(link+'/') for link in links):continue
            for m in [self.m,self.control.mj]:
                i=m.geom(name).id
                for field in ['solref','solimp','friction']:
                    values=np.fromstring(node.get(field),sep=' ')
                    getattr(m,'geom_'+field)[i,:len(values)]=values
                m.geom_priority[i]=int(node.get('priority'))
        for m in [self.m,self.control.mj]:
            configure(m);m.opt.impratio=10000.;m.opt.ccd_tolerance=1e-12;m.opt.ccd_iterations=1000

    def observation(self,data):
        q=self.control.mj.qpos0.copy();v=np.zeros(self.control.mj.nv)
        q[self.cq]=data.qpos[self.pq];v[self.cv]=data.qvel[self.pv]
        q[self.coi:self.coi+7]=self.spec.pose(data.qpos[self.oi])
        v[self.cov+5]=data.qvel[self.ov]
        return q,v

    def initialize(self,free_q):
        data=mujoco.MjData(self.m);data.qpos[self.pq]=free_q[self.cq]
        data.qpos[self.oi]=0.
        mujoco.mj_forward(self.m,data)
        return data
