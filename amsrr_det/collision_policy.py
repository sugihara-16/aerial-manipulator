"""Native face contacts with reliable single witnesses on curved cylinder sides.

The public MuJoCo collision table is installed once per process before starting
workers. Shapes, materials and force laws are unchanged. On a cylinder cap,
multiple points are accepted only when their normals agree with its witness.
"""
import ctypes,hashlib,os,subprocess
from pathlib import Path
import mujoco
p=Path(__file__).resolve().parents[1]/'build';s=Path(__file__).with_suffix('.cc');pkg=Path(mujoco.__file__).parent
h=hashlib.sha256(s.read_bytes()+mujoco.mj_versionString().encode()).hexdigest()[:12]
so=p/f'collision_policy_{h}.so'
if not so.exists():
 temp=so.with_suffix(f'.{os.getpid()}.so')
 subprocess.run(['c++','-O3','-std=c++17','-shared','-fPIC','-I'+str(pkg/'include'),str(s),str(pkg/f'libmujoco.so.{mujoco.mj_versionString()}'),'-Wl,-rpath,'+str(pkg),'-o',str(temp)],check=True)
 os.replace(temp,so)
lib=ctypes.CDLL(str(so));lib.install_policy.restype=ctypes.c_int
if lib.install_policy()!=0:raise RuntimeError('MuJoCo collision header/library mismatch')
def configure(model):
 model.opt.disableflags&=~int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)
 model.opt.disableflags&=~int(mujoco.mjtDisableBit.mjDSBL_MULTICCD)
