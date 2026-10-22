"""Equivalent evaluator reduction in native code; no control inputs added."""
import ctypes,hashlib,os,subprocess
from pathlib import Path
import numpy as np
import mujoco
from amsrr_det.task_audit import TaskAudit
from .model import ROOT

class FastTaskAudit(TaskAudit):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        root=Path(mujoco.__file__).parent;source=Path(__file__).with_suffix('.cc')
        version=mujoco.mj_versionString();digest=hashlib.sha256(source.read_bytes()+version.encode()).hexdigest()[:16]
        binary=ROOT/'build'/('fast_task_audit_'+digest+'.so')
        binary.parent.mkdir(exist_ok=True)
        if not binary.exists():
            temporary=binary.with_suffix('.'+str(os.getpid())+'.so')
            subprocess.run(['c++','-O3','-std=c++17','-shared','-fPIC','-Wall','-Wextra','-I'+str(root/'include'),str(source),str(root/('libmujoco.so.'+version)),'-Wl,-rpath,'+str(root),'-o',str(temporary)],check=True)
            os.replace(temporary,binary)
        self.library=ctypes.CDLL(str(binary))
        if self.library.audit_version()!=mujoco.mj_version():raise RuntimeError('MuJoCo header/library version mismatch')
        ptr=ctypes.c_void_p;integer=ctypes.c_int
        self.call=self.library.audit_step;self.call.restype=integer
        self.call.argtypes=[ptr,ptr,integer,integer,integer,integer,ptr,ptr,ptr,integer,ptr,ptr,ptr,ptr,ptr,integer,ptr,integer,ptr,ctypes.c_double,ptr,ptr,ptr,ptr,ptr,ptr]
        self._native_meta=[np.array(x,dtype=np.int32)for x in [list(self.link_bodies.values()),self.groups,self.allowed_support,self.ai,self.qadr,self.vadr]]
        self.size=np.ascontiguousarray(self.size,dtype=float)
        if min(self.object_geom,self.object_body,*self.link_bodies.values(),*self.ai,*self.qadr,*self.vadr)<0:raise ValueError('missing audit body, joint or actuator')
        self._stats=np.zeros(14);self._stats[7]=np.inf;self._stats[13]=np.nan
        self._anchors=np.zeros((len(self.links),10))
        self._vertices=np.ascontiguousarray(self.shape.vertices if self.shape is not None and self.shape.vertices is not None else np.empty((0,3)),dtype=float)
        self._shape=0 if self.shape is None or self.shape.family in ['box','elongated_box'] else {'ellipsoid':1,'cylinder':2}.get(self.shape.family,3)
        # These arrays are owned by this evaluator and never replaced. Cache
        # their addresses instead of rebuilding ctypes wrappers every step.
        b,g,s,ai,qi,vi=self._native_meta
        address=lambda x:x.ctypes.data
        self._arguments=(self.object_geom,len(self.links),address(b),address(g),address(s),len(ai),
            address(ai),address(qi),address(vi),address(self.limits),address(self.effort),self._shape,
            address(self.size),len(self._vertices),address(self._vertices),self.support_top,
            address(self._stats),address(self.saturated_time),address(self.gap_time),
            address(self._anchors),address(self.last_normal_loads))
        self._unexpected=np.empty((16,2),dtype=np.int32)
        if self.proximity.joint_motion is not None:
            self.proximity.joint_motion=_NativeMotionBound(self.proximity.joint_motion,self.library)

    def update(self,data,*,carried=False,grasp_contact_allowed=True):
        m=self.plant.mj
        if data.model._address!=m._address:raise ValueError('data/model mismatch')
        if data.ncon>len(self._unexpected):self._unexpected=np.empty((data.ncon,2),dtype=np.int32)
        count=self.call(m._address,data._address,carried,grasp_contact_allowed,*self._arguments,self._unexpected.ctypes.data)
        if count<0:raise ValueError('audit requires 3D point contacts')
        self.other_contacts.update(tuple(sorted([self.names[a],self.names[b]]))for a,b in self._unexpected[:count])
        (self.max_force,self.max_moment,self.max_penetration,self.max_joint_speed,self.max_effort_ratio,self.max_joint_limit_violation,self.max_clearance,self.min_clearance_carried,self.max_raw_effort_ratio,self.max_servo_error,self.max_slip,self.max_material_displacement,self.longest_gap,last)=self._stats
        self.last_contact_time=None if np.isnan(last) else float(last);self.time=float(data.time)
        violations,part,environment=self.proximity.check(data.qpos,grasp_contact_allowed)
        self.minimum_part_gap=min(self.minimum_part_gap,part);self.minimum_environment_gap=min(self.minimum_environment_gap,environment);self.clearance_violations.update(violations)


class _NativeMotionBound:
    def __init__(self,bound,library):
        self.original=bound
        self.library=library
        self.call=self.library.motion_maximum;p=ctypes.c_void_p;i=ctypes.c_int
        self.call.argtypes=[i,i,i,i,p,p,p,p,p,p,p];self.call.restype=None
        self.qi=np.asarray(bound.qi,dtype=np.int32);self.levers=bound.levers
        self.roots=np.full(len(self.levers),-1,np.int32);self.reaches=np.zeros(len(self.roots))
        for k,(rows,reach)in bound.roots.items():self.roots[rows]=k;self.reaches[rows]=reach
        self._arguments=(len(self.roots),len(self.qi),self.qi.ctypes.data,self.levers.ctypes.data,self.roots.ctypes.data,self.reaches.ctypes.data)
    def maximum(self,states,anchor):
        states=np.atleast_2d(states)
        if len(states)!=1:return self.original.maximum(states,anchor)
        states=np.ascontiguousarray(states,dtype=float);anchor=np.ascontiguousarray(anchor,dtype=float);output=np.zeros(1)
        if states.shape[1]!=len(anchor):raise ValueError("state/anchor shape mismatch")
        address=lambda x:x.ctypes.data
        self.call(1,states.shape[1],*self._arguments,address(states),address(anchor),address(output))
        return output
