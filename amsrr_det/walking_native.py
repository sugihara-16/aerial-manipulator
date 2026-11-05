"""Compiled evaluation of the unchanged periodic geometric residual/Jacobian."""
import ctypes as ct,hashlib,os,subprocess,tempfile,copy
from pathlib import Path
from functools import lru_cache
import mujoco,numpy as np

@lru_cache(maxsize=1)
def library():
 root=Path(__file__).resolve().parents[1];source=root/'native/walking_geometry.cpp';package=Path(mujoco.__file__).parent
 eigen=next((root/'.venv/lib').glob('python*/site-packages/cmeel.prefix/include/eigen3'))
 version=mujoco.mj_versionString()
 # The binary is built for this host. Include its target in the cache key so
 # a checkout copied to another processor cannot reuse incompatible code.
 info=Path('/proc/cpuinfo').read_text().splitlines()
 target='\n'.join(next((line for line in info if line.startswith(key)),key) for key in ['model name','flags'])
 flags=['-O3','-march=native','-DNDEBUG','-std=c++17','-shared','-fPIC']
 digest=hashlib.sha256(source.read_bytes()+version.encode()+target.encode()+' '.join(flags).encode()).hexdigest()[:16]
 binary=root/'build'/f'walking_geometry_{digest}.so'
 if not binary.exists():
  fd,name=tempfile.mkstemp(suffix='.so',dir=binary.parent);os.close(fd)
  try:
   subprocess.run(['c++',*flags,'-I'+str(package/'include'),'-I'+str(eigen),str(source),str(package/f'libmujoco.so.{version}'),'-Wl,-rpath,'+str(package),'-o',name],check=True)
   os.replace(name,binary)
  finally:Path(name).unlink(missing_ok=True)
 lib=ct.CDLL(str(binary));p=ct.c_void_p;i=ct.c_int
 lib.walking_geometry.argtypes=[p,p,i,p,p,p,i,p,p,p,p,p,p,p,p,i,p,p,p,p,p];lib.walking_geometry.restype=None
 lib.walking_seed.argtypes=[p,p,i,p,p,p,p,p,i,p,p,p,p,p,p,i,p,p,p,i];lib.walking_seed.restype=ct.c_double
 return lib

class CycleGeometry:
 def __init__(self,kin):
  self.kin=kin;self.call=library().walking_geometry;self.indices=np.array(kin.indices,np.int32)
  # These residuals discard plane contacts and already use exact material
  # height/part bounds. Omit only that unused query; final CAD and physics
  # retain the original model with every floor contact enabled.
  self.geometry_model=copy.copy(kin.m);floor=self.geometry_model.geom('floor').id
  self.geometry_model.geom_contype[floor]=0;self.geometry_model.geom_conaffinity[floor]=0
  self.feet=np.array([b for b,_ in kin.controller.material],np.int32);self.offsets=np.r_[0,np.cumsum([len(v) for _,v in kin.controller.material])].astype(np.int32)
  self.vertices=np.vstack([v for _,v in kin.controller.material]);self.centers=np.ascontiguousarray(kin.controller.foot_points)
  self.bodies=np.array(kin.bids,np.int32);self.bound_centers=np.ascontiguousarray(kin.centers);self.halves=np.ascontiguousarray(kin.halves)
  self.nd=len(self.indices);self.rows=12*len(self.feet)+self.nd+2+len(self.bodies)+1
 def evaluate(self,x,base,targets,rotations,support,com):
  # Retain conversion storage until the native call returns.
  args=[self.indices,np.ascontiguousarray(x),np.ascontiguousarray(base),self.feet,self.offsets,self.vertices,self.centers,np.ascontiguousarray(targets),np.ascontiguousarray(rotations),np.array(support,np.int32),np.ascontiguousarray(com),self.bodies,self.bound_centers,self.halves]
  r=np.empty(self.rows);J=np.empty((self.rows,self.nd));a=[v.ctypes.data for v in args];k=self.kin
  self.call(self.geometry_model._address,k.d._address,self.nd,*a[:3],len(self.feet),*a[3:11],len(self.bodies),*a[11:],r.ctypes.data,J.ctypes.data)
  return r,J

 def fit(self,seed,body,feet,iterations,rotations=None):
  k=self.kin;q=np.array(seed,copy=True);ref=k.nominal.copy();ref[:7]=body
  arrays=[self.indices,np.array(k.qi,np.int32),np.ascontiguousarray(k.limits),q,ref,self.feet,self.offsets,self.vertices,self.centers,
          np.ascontiguousarray(feet),np.ascontiguousarray(k.rotations if rotations is None else rotations),self.bodies,self.bound_centers,self.halves]
  a=[v.ctypes.data for v in arrays]
  error=library().walking_seed(self.geometry_model._address,k.d._address,self.nd,*a[:5],len(self.feet),*a[5:11],len(self.bodies),*a[11:],iterations)
  return q,error
