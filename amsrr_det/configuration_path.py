"""Smooth configuration paths and their own velocity/acceleration maps."""
import numpy as np
import mujoco
from scipy.interpolate import PchipInterpolator
from scipy.spatial.transform import Rotation,RotationSpline


class ConfigurationPath:
    def __init__(self,model,parameters,configurations):
        self.m=model;self.parameters=np.asarray(parameters);q=np.asarray(configurations)
        self.scalar=PchipInterpolator(self.parameters,q,axis=0)
        self.scalar_joints=[];self.free=[]
        for j in range(model.njnt):
            qi=int(model.jnt_qposadr[j]);vi=int(model.jnt_dofadr[j])
            kind=int(model.jnt_type[j])
            if kind==int(mujoco.mjtJoint.mjJNT_FREE):
                rotation=Rotation.from_quat(q[:,[qi+4,qi+5,qi+6,qi+3]])
                self.free.append((qi,vi,RotationSpline(self.parameters,rotation)))
            elif kind in (int(mujoco.mjtJoint.mjJNT_HINGE),int(mujoco.mjtJoint.mjJNT_SLIDE)):
                self.scalar_joints.append((qi,vi))
            else:raise ValueError('configuration path needs free or scalar joints')

    def pose(self,parameter):
        s=float(np.clip(parameter,self.parameters[0],self.parameters[-1]))
        q=self.scalar(s)
        for qi,_,spline in self.free:
            q[qi+3:qi+7]=spline(s).as_quat()[[3,0,1,2]]
        return q

    def at(self,parameter):
        s=float(np.clip(parameter,self.parameters[0],self.parameters[-1]))
        q=self.scalar(s);first=self.scalar(s,1);second=self.scalar(s,2)
        v=np.zeros(self.m.nv);a=np.zeros(self.m.nv)
        for qi,vi in self.scalar_joints:v[vi]=first[qi];a[vi]=second[qi]
        for qi,vi,spline in self.free:
            quat=spline(s).as_quat();q[qi+3:qi+7]=quat[[3,0,1,2]]
            v[vi:vi+3]=first[qi:qi+3];a[vi:vi+3]=second[qi:qi+3]
            v[vi+3:vi+6]=spline(s,1);a[vi+3:vi+6]=spline(s,2)
        return q,v,a


class ContactConfigurationPath:
    """Keep a single material frame aligned through configuration interpolation.

    For multiple contacts, retain the fitted whole-robot curve. Its geometric
    and force screens still decide acceptance. No future physics is integrated.
    """
    def __init__(self,plant,parameters,configurations,surfaces,normals,object_shape=None):
        import pinocchio as pin
        from .grasp_geometry import grasp_frame
        self.plant=plant;self.m=plant.mj;self.parameters=np.asarray(parameters)
        self.curve=ConfigurationPath(self.m,parameters,configurations)
        self.single=len(surfaces)==1
        if not self.single:return
        self.shape=object_shape
        if object_shape is not None:
            from .contact_geometry import material_triangles
            self.triangles=material_triangles(surfaces[0][1],plant.meta.get('material_mesh_cache'))
        self.pin=pin;self.data=plant.pin.createData();self.pq,self.mq,*_=plant.coordinate_maps
        self.frame=plant.meta['frames'][grasp_frame(*surfaces[0])]
        self.of=plant.meta['object']['frame'];self.normal=np.asarray(normals[0]);local=[]
        for q in configurations:
            qp=pin.neutral(plant.pin);qp[self.pq]=np.asarray(q)[self.mq]
            pin.framesForwardKinematics(plant.pin,self.data,qp)
            T=self.data.oMf[self.of].inverse()*self.data.oMf[self.frame]
            local.append(np.r_[T.translation,pin.Quaternion(T.rotation).coeffs()])
        local=np.asarray(local)
        self.position=PchipInterpolator(parameters,local[:,:3],axis=0)
        self.rotation=RotationSpline(parameters,Rotation.from_quat(local[:,3:]))
        self.material=pin.SE3(Rotation.from_quat(local[len(local)//2,3:]).as_matrix(),local[len(local)//2,:3])

    def pose(self,s):
        q=self.curve.pose(s)
        if not self.single:return q
        pin=self.pin;p=self.plant;qp=pin.neutral(p.pin);qp[self.pq]=q[self.mq]
        pin.framesForwardKinematics(p.pin,self.data,qp)
        R=self.material.rotation.copy();x=R[:,0];axis=np.cross(x,self.normal);length=np.linalg.norm(axis)
        if length>1e-10:R=pin.exp3(axis/length*np.arctan2(length,x@self.normal))@R
        local=pin.SE3(R,self.material.translation.copy())
        if self.shape is not None:
            from .contact_geometry import close_material
            for _ in range(2):
                gap=float(close_material(self.shape,local,self.triangles)[2].min())
                local.translation+=self.normal*(gap+.0002)
        target=self.data.oMf[self.of]*local
        base=pin.SE3(pin.Quaternion(qp[3:7]).matrix(),qp[:3])
        base=target*self.data.oMf[self.frame].inverse()*base
        q[:3]=base.translation;q[3:7]=pin.Quaternion(base.rotation).coeffs()[[3,0,1,2]]
        return q

    def at(self,parameter):
        if not self.single:return self.curve.at(parameter)
        lo,hi=self.parameters[[0,-1]];s=float(np.clip(parameter,lo,hi));q=self.pose(s)
        h=min(1e-4,(s-lo)/2,(hi-s)/2)
        if h<1e-7:return q,np.zeros(self.m.nv),np.zeros(self.m.nv)
        qm,qp=self.pose(s-h),self.pose(s+h);vm=np.zeros(self.m.nv);vp=np.zeros(self.m.nv)
        mujoco.mj_differentiatePos(self.m,vm,h,qm,q)
        mujoco.mj_differentiatePos(self.m,vp,h,q,qp)
        for qi,vi,_ in self.curve.free:
            R=Rotation.from_quat(q[[qi+4,qi+5,qi+6,qi+3]]).as_matrix()
            Rm=Rotation.from_quat(qm[[qi+4,qi+5,qi+6,qi+3]]).as_matrix()
            vm[vi+3:vi+6]=R.T@Rm@vm[vi+3:vi+6]
        return q,(vm+vp)/2,(vp-vm)/h


def _quat_product(a,b):
    return np.r_[a[0]*b[0]-a[1:]@b[1:],a[0]*b[1:]+b[0]*a[1:]+np.cross(a[1:],b[1:])]


class AxisRotationPath:
    """Exact rigid assembly rotation; its derivatives share the same curve.

    Static load transfer along this curve is valid only when the task's
    gravity and load maps have the corresponding rotational symmetry.
    This class itself makes no feasibility or collision claim.
    """
    def __init__(self,plant,parameters,configuration,axis,center,origin=0.):
        self.m=plant.mj;self.parameters=np.asarray(parameters);self.q=np.asarray(configuration).copy()
        self.axis=np.asarray(axis,dtype=float);self.center=np.asarray(center,dtype=float);self.origin=float(origin)
        if abs(np.linalg.norm(self.axis)-1)>1e-12:raise ValueError('rotation axis must be unit length')
        self.free=[]
        for j in range(self.m.njnt):
            if self.m.jnt_type[j]==mujoco.mjtJoint.mjJNT_FREE:
                qi=int(self.m.jnt_qposadr[j]);vi=int(self.m.jnt_dofadr[j])
                R=Rotation.from_quat(self.q[[qi+4,qi+5,qi+6,qi+3]]).as_matrix()
                self.free.append((qi,vi,R.T@self.axis))
        vectoring=[a for a in plant.meta['actuators'] if a['role']=='vectoring']
        self.gq=np.array([int(self.m.jnt_qposadr[self.m.joint(a['name']).id]) for a in vectoring])
        self.gv=np.array([int(self.m.jnt_dofadr[self.m.joint(a['name']).id]) for a in vectoring])
        self.directions=None

    def pose(self,parameter):
        s=float(np.clip(parameter,*self.parameters[[0,-1]]));angle=s-self.origin
        rotation=Rotation.from_rotvec(self.axis*angle).as_matrix()
        quat=np.r_[np.cos(angle/2),self.axis*np.sin(angle/2)];q=self.q.copy()
        for qi,_,_ in self.free:
            q[qi:qi+3]=self.center+rotation@(self.q[qi:qi+3]-self.center)
            q[qi+3:qi+7]=_quat_product(quat,self.q[qi+3:qi+7])
        if self.directions is not None:q[self.gq]=self.directions(s)
        return q

    def at(self,parameter):
        s=float(np.clip(parameter,*self.parameters[[0,-1]]));q=self.pose(s)
        v=np.zeros(self.m.nv);a=np.zeros(self.m.nv)
        for qi,vi,local_axis in self.free:
            v[vi:vi+3]=np.cross(self.axis,q[qi:qi+3]-self.center)
            a[vi:vi+3]=np.cross(self.axis,v[vi:vi+3]);v[vi+3:vi+6]=local_axis
        if self.directions is not None:
            v[self.gv]=self.directions(s,1);a[self.gv]=self.directions(s,2)
        return q,v,a

    def velocities(self,parameters):
        s=np.clip(np.asarray(parameters),*self.parameters[[0,-1]])
        rotation=Rotation.from_rotvec((s-self.origin)[:,None]*self.axis).as_matrix()
        v=np.zeros((len(s),self.m.nv))
        for qi,vi,local_axis in self.free:
            arm=rotation@(self.q[qi:qi+3]-self.center)
            v[:,vi:vi+3]=np.cross(self.axis,arm);v[:,vi+3:vi+6]=local_axis
        if self.directions is not None:v[:,self.gv]=self.directions(s,1)
        return v

    def set_directions(self,parameters,configurations):
        self.directions=PchipInterpolator(parameters,np.asarray(configurations)[:,self.gq],axis=0)

    def record(self):
        return dict(kind='rigid_axis',parameters=self.parameters,configuration=self.q,axis=self.axis,
            center=self.center,origin=self.origin,direction_knots=None if self.directions is None else self.directions.x,
            direction_coefficients=None if self.directions is None else self.directions.c)

    @classmethod
    def from_record(cls,plant,record):
        from scipy.interpolate import PPoly
        self=cls(plant,record['parameters'],record['configuration'],record['axis'],record['center'],record['origin'])
        if record['direction_knots'] is not None:
            self.directions=PPoly(np.asarray(record['direction_coefficients']),np.asarray(record['direction_knots']))
        return self


class TabulatedContactPath:
    """Prepare a differentiable real-time reference; no online frame fitting.

    The final interpolant, including normalized quaternions and their own
    derivatives, is the curve used by both force screening and execution.
    """
    def __init__(self,plant,parameters,configurations,surfaces,normals,*,geometry_parameters=None,geometry_configurations=None,object_shape=None):
        from scipy.interpolate import CubicSpline
        direction_parameters=parameters;direction_configurations=configurations
        if geometry_parameters is not None:
            parameters=geometry_parameters;configurations=geometry_configurations
        direct=ContactConfigurationPath(plant,parameters,configurations,surfaces,normals,object_shape)
        self.m=plant.mj;self.free=direct.curve.free;self.scalar=direct.curve.scalar_joints
        self.parameters=np.asarray(parameters);lo,hi=self.parameters[[0,-1]]
        nodes=np.sort(np.r_[self.parameters,np.linspace(lo,hi,int(np.ceil((hi-lo)/np.deg2rad(1)))+1)])
        # The coarse and dense linspaces represent the same angles with
        # slightly different roundoff. Near-duplicate knots destabilize a
        # cubic even when the position samples differ only at roundoff.
        nodes=nodes[np.r_[True,np.diff(nodes)>1e-10*max(1.,hi-lo)]]
        poses=[]
        for s in nodes:
            q=direct.pose(float(s))
            if poses:
                for qi,_,_ in self.free:
                    if q[qi+3:qi+7]@poses[-1][qi+3:qi+7]<0:q[qi+3:qi+7]*=-1
            poses.append(q)
        # Fit positions once. Runtime derivatives are those of this final
        # normalized cubic itself, not a second finite-difference path.
        self.spline=CubicSpline(nodes,np.asarray(poses),axis=0)
        self.gq=np.array([int(self.m.jnt_qposadr[self.m.joint(a['name']).id]) for a in plant.meta['actuators'] if a['role']=='vectoring'])
        self.gv=np.array([int(self.m.jnt_dofadr[self.m.joint(a['name']).id]) for a in plant.meta['actuators'] if a['role']=='vectoring'])
        self.directions=None
        if geometry_parameters is not None:self.set_directions(direction_parameters,direction_configurations)

    def record(self):
        return dict(parameters=self.parameters,knots=self.spline.x,coefficients=self.spline.c,
            direction_knots=None if self.directions is None else self.directions.x,
            direction_coefficients=None if self.directions is None else self.directions.c)

    @classmethod
    def from_record(cls,plant,record):
        if record.get('kind')=='rigid_axis':return AxisRotationPath.from_record(plant,record)
        from scipy.interpolate import PPoly
        self=cls.__new__(cls);self.m=plant.mj;self.free=[];self.scalar=[]
        self.parameters=np.asarray(record['parameters'])
        for j in range(self.m.njnt):
            qi=int(self.m.jnt_qposadr[j]);vi=int(self.m.jnt_dofadr[j])
            if self.m.jnt_type[j]==mujoco.mjtJoint.mjJNT_FREE:self.free.append((qi,vi,None))
            else:self.scalar.append((qi,vi))
        self.gq=np.array([int(self.m.jnt_qposadr[self.m.joint(a['name']).id]) for a in plant.meta['actuators'] if a['role']=='vectoring'])
        self.gv=np.array([int(self.m.jnt_dofadr[self.m.joint(a['name']).id]) for a in plant.meta['actuators'] if a['role']=='vectoring'])
        self.spline=PPoly(np.asarray(record['coefficients']),np.asarray(record['knots']))
        self.directions=None if record['direction_knots'] is None else PPoly(np.asarray(record['direction_coefficients']),np.asarray(record['direction_knots']))
        return self

    def set_directions(self,parameters,configurations):
        """Change only rotor angles; keep the fitted body/contact curve fixed."""
        self.directions=PchipInterpolator(parameters,np.asarray(configurations)[:,self.gq],axis=0)

    def at(self,parameter):
        s=float(np.clip(parameter,*self.parameters[[0,-1]]));q=self.spline(s);qd=self.spline(s,1);qdd=self.spline(s,2)
        v=np.zeros(self.m.nv);a=np.zeros(self.m.nv)
        for qi,vi in self.scalar:v[vi]=qd[qi];a[vi]=qdd[qi]
        for qi,vi,_ in self.free:
            v[vi:vi+3]=qd[qi:qi+3];a[vi:vi+3]=qdd[qi:qi+3]
            sl=slice(qi+3,qi+7);x=q[sl].copy();dx=qd[sl];ddx=qdd[sl];norm=np.linalg.norm(x);u=x/norm
            dn=float(u@dx);du=(dx-u*dn)/norm;ddn=float(du@dx+u@ddx);ddu=(ddx-u*ddn-2*du*dn)/norm
            conjugate=u*np.array([1.,-1.,-1.,-1.]);q[sl]=u
            v[vi+3:vi+6]=2*_quat_product(conjugate,du)[1:]
            a[vi+3:vi+6]=2*_quat_product(conjugate,ddu)[1:]
        if self.directions is not None:
            q[self.gq]=self.directions(s);v[self.gv]=self.directions(s,1);a[self.gv]=self.directions(s,2)
        return q,v,a

    def pose(self,parameter):
        s=float(np.clip(parameter,*self.parameters[[0,-1]]));q=self.spline(s)
        for qi,_,_ in self.free:q[qi+3:qi+7]/=np.linalg.norm(q[qi+3:qi+7])
        if self.directions is not None:q[self.gq]=self.directions(s)
        return q

    def velocities(self,parameters):
        """Batch the same normalized-curve derivative used by at()."""
        s=np.clip(np.asarray(parameters),*self.parameters[[0,-1]])
        q=self.spline(s);qd=self.spline(s,1);v=np.zeros((len(s),self.m.nv))
        for qi,vi in self.scalar:v[:,vi]=qd[:,qi]
        for qi,vi,_ in self.free:
            v[:,vi:vi+3]=qd[:,qi:qi+3]
            x=q[:,qi+3:qi+7];dx=qd[:,qi+3:qi+7]
            norm=np.linalg.norm(x,axis=1,keepdims=True);u=x/norm
            du=(dx-u*np.sum(u*dx,axis=1,keepdims=True))/norm
            v[:,vi+3:vi+6]=2*(u[:,:1]*du[:,1:]-du[:,:1]*u[:,1:]-np.cross(u[:,1:],du[:,1:]))
        if self.directions is not None:v[:,self.gv]=self.directions(s,1)
        return v


class PathClock:
    """Local velocity-based time allocation, followed by a smooth start/stop.

    The dense path derivatives define a sampled kinematic speed margin. This
    does not claim actuator acceleration or dynamic feasibility guarantees.
    """
    def __init__(self,plant,curve,minimum_duration=10.):
        model=plant.mj;lo,hi=curve.parameters[[0,-1]]
        nodes=np.linspace(lo,hi,max(3,int(np.ceil((hi-lo)/np.deg2rad(.25)))+1))
        actuators=plant.meta['actuators']
        indices=[int(model.jnt_dofadr[model.joint(a['name']).id]) for a in actuators]
        limits=.5*np.array([a['speed'] for a in actuators])
        velocity=(curve.velocities(nodes) if hasattr(curve,'velocities') else
                  np.asarray([curve.at(parameter)[1] for parameter in nodes]))
        rates=np.maximum.reduce([np.full(len(nodes),minimum_duration/(hi-lo)),
            np.max(abs(velocity[:,indices])/limits,axis=1),
            np.linalg.norm(velocity[:,:3],axis=1)/.3,np.linalg.norm(velocity[:,3:6],axis=1)/.8])
        # 1.875 is the peak derivative of the outer quintic clock. Another
        # 1.5 covers the scalar shape-preserving cubic's local slope growth.
        times=np.r_[0.,np.cumsum(np.diff(nodes)*np.maximum(rates[1:],rates[:-1])*1.875*1.5)]
        self.duration=float(times[-1]);self.curve=PchipInterpolator(times,nodes)

    def at(self,time):
        p=float(np.clip(time/self.duration,0,1))
        s=p**3*(10-15*p+6*p*p);ds=30*p*p*(1-p)**2
        dds=60*p*(1-p)*(1-2*p)/self.duration
        clock=self.duration*s
        return (float(self.curve(clock)),float(self.curve(clock,1)*ds),
                float(self.curve(clock,2)*ds**2+self.curve(clock,1)*dds))
