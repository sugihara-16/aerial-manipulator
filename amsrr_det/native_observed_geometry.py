"""Batch exact MuJoCo queries; no new geometry model or distance tolerance."""
import ctypes,hashlib,os,subprocess,tempfile
from functools import lru_cache
from pathlib import Path
import numpy as np
import mujoco

@lru_cache(maxsize=1)
def library():
    root=Path(__file__).resolve().parents[1];source=root/'native/observed_geometry.cpp'
    package=Path(mujoco.__file__).parent;version=mujoco.mj_versionString()
    digest=hashlib.sha256(source.read_bytes()+version.encode()).hexdigest()[:16]
    binary=root/'build'/f'observed_geometry_{digest}.so'
    if not binary.exists():
        fd,name=tempfile.mkstemp(prefix='observed_geometry_',suffix='.so',dir=binary.parent);os.close(fd)
        try:
            subprocess.run(['c++','-O3','-std=c++17','-shared','-fPIC','-I'+str(package/'include'),str(source),
                str(package/f'libmujoco.so.{version}'),'-Wl,-rpath,'+str(package),'-o',name],check=True)
            os.replace(name,binary)
        finally:
            if Path(name).exists():Path(name).unlink()
    lib=ctypes.CDLL(str(binary))
    if lib.observed_geometry_version()!=mujoco.mj_version():raise RuntimeError('MuJoCo geometry library version mismatch')
    pointer=ctypes.c_void_p
    lib.pad_gap_rows.argtypes=[pointer,pointer,ctypes.c_int,ctypes.c_int,pointer,pointer,pointer,pointer]
    lib.pad_gap_rows.restype=ctypes.c_int
    lib.hold_acceleration.argtypes=[ctypes.c_int,*[pointer]*8]
    lib.hold_acceleration.restype=None
    lib.contact_mechanics.argtypes=[pointer,pointer,ctypes.c_int,pointer,ctypes.c_int,pointer,pointer,pointer]
    lib.contact_mechanics.restype=None
    return lib

@lru_cache(maxsize=128)
def packed_groups(groups):
    counts=[len(g)for g in groups]
    return np.r_[0,np.cumsum(counts)].astype(np.int32),np.array([g for group in groups for g in group],dtype=np.int32)

def pad_gap_rows(model,data,groups):
    starts,geoms=packed_groups(tuple(tuple(g)for g in groups));n=len(groups)
    gaps=np.empty(n);rows=np.empty((n,model.nv))
    error=library().pad_gap_rows(model._address,data._address,model.geom('object').id,n,
        starts.ctypes.data,geoms.ctypes.data,gaps.ctypes.data,rows.ctypes.data)
    if error or not np.isfinite(gaps).all() or not np.isfinite(rows).all():
        raise ValueError('pad surface is outside geometry query')
    return gaps,rows
