"""Goldfarb/Idnani QP, with batched BLAS operations and unchanged checks.

The upstream C implementation and its license are in third_party/quadprog.
Compilation occurs once per source/platform and never inside a control cycle.
"""
import ctypes
import hashlib
import os
from pathlib import Path
import platform
import subprocess
import tempfile

import numpy as np
import scipy


def _load():
    root=Path(__file__).resolve().parents[1];source=root/'third_party/quadprog'
    files=[source/f for f in ['solve.QP.c','linear-algebra.c','qr-update.c']]+[root/'native/contact_algebra.c']
    blas=next((Path(scipy.__file__).parent.parent/'scipy.libs').glob('libscipy_openblas-*.so'))
    digest=hashlib.sha256(b''.join(f.read_bytes() for f in files)+str(blas).encode()+platform.machine().encode()).hexdigest()[:16]
    binary=root/'build'/f'contact_qp_{digest}.so'
    if not binary.exists():
        fd,name=tempfile.mkstemp(prefix='contact_qp_',suffix='.so',dir=binary.parent);os.close(fd)
        try:
            subprocess.run(['cc','-O3','-fPIC','-shared','-ffp-contract=off','-Wl,-Bsymbolic',
                *map(str,files),str(blas),'-Wl,-rpath,'+str(blas.parent),'-lm','-o',name],check=True)
            os.replace(name,binary)
        finally:
            if Path(name).exists():Path(name).unlink()
    lib=ctypes.CDLL(str(binary));call=lib.qpgen2_;p=ctypes.c_void_p;i=ctypes.c_int
    call.argtypes=[p,p,i,p,p,p,p,p,i,i,p,p,p,p,i];call.restype=i
    lib.contact_project.argtypes=[i,p,p,ctypes.c_double,p,p,p,p,p];lib.contact_project.restype=None
    lib.contact_branch.argtypes=[i]*7+[p]*7+[ctypes.c_double,p,p,ctypes.c_double,p,p,p];lib.contact_branch.restype=None
    return lib,call


_library,_solve=_load()


def project(affine,rn,ratio,operators,dual,G):
    arrays=[np.ascontiguousarray(x,dtype=float) for x in [affine,rn,operators,dual,G]]
    modes=np.empty(len(rn),np.int32);forces=np.empty((len(rn),3))
    _library.contact_project(len(rn),arrays[0].ctypes.data,arrays[1].ctypes.data,float(ratio),
        *[a.ctypes.data for a in arrays[2:]],modes.ctypes.data,forces.ctypes.data)
    return modes,forces


def branch(A,b,nv,nu,ne,ni,pressure,Dn,p0,rn,alpha,ratio,H,vertices,C,modes):
    arrays=[np.ascontiguousarray(x,dtype=float) for x in [A,b,pressure,Dn,p0,rn,alpha,H,vertices]]
    modes=np.ascontiguousarray(modes,dtype=np.int32);count=len(modes);n=A.shape[1]
    multipliers=int(np.maximum((modes>0).astype(int)+(modes>8),0).sum());apex=int((modes<0).sum())
    total=n+count+multipliers;eq=ne+count+multipliers;ineq=ni+8*apex+multipliers
    matrix=np.empty((len(b)+count+2*multipliers+8*apex,total));bound=np.empty(len(matrix))
    _library.contact_branch(len(b),n,nv,nu,count,ne,ni,*[a.ctypes.data for a in arrays[:7]],float(ratio),
        arrays[7].ctypes.data,arrays[8].ctypes.data,float(C),modes.ctypes.data,matrix.ctypes.data,bound.ctypes.data)
    return matrix,bound,eq,ineq


def solve_qp(H,a,C,b,meq=0,factorized=False):
    n=len(a);count=len(b);r=min(n,count)
    G=np.array(H,dtype=float,order='F',copy=True);linear=np.array(a,dtype=float,copy=True)
    matrix=np.array(C,dtype=float,order='F',copy=True);bound=np.array(b,dtype=float,copy=True)
    if (G.shape!=(n,n) or matrix.shape!=(n,count) or linear.shape!=(n,) or bound.shape!=(count,)
        or n<1 or not 0<=meq<=count or any(not x.flags.f_contiguous for x in [G,matrix])
        or not all(np.isfinite(x).all() for x in [G,linear,matrix,bound])):
        raise ValueError('invalid dense QP input')
    x=np.empty(n);dual=np.empty(count);cost=np.empty(1)
    active=np.empty(count,dtype=np.int32);nact=np.empty(1,dtype=np.int32);iterations=np.empty(2,dtype=np.int32)
    work=np.empty(2*n+2*count+r*(r+5)//2)
    status=_solve(G.ctypes.data,linear.ctypes.data,n,x.ctypes.data,dual.ctypes.data,cost.ctypes.data,
        matrix.ctypes.data,bound.ctypes.data,count,meq,active.ctypes.data,nact.ctypes.data,
        iterations.ctypes.data,work.ctypes.data,int(factorized))
    if status:raise ValueError('constraints are inconsistent, no solution' if status==1 else 'H is not positive definite')
    return x,cost[0],linear,iterations,dual,active[:nact[0]]
