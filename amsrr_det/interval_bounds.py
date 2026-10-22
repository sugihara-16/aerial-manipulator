"""Conservative certificate for pairs absent from a proximity query.

The previous certificate uses twice the largest world displacement for every
pair. Here common ancestor motion cancels, and an enclosing box can establish
additional distance. Present contacts still undergo the original exact check.
"""
import ctypes,hashlib,os,subprocess
from functools import lru_cache
from pathlib import Path
from .model import ROOT
import numpy as np
import mujoco


class IntervalPairBounds:
    def __init__(self, check, body_rows, roots):
        m=check.model;self.check=check;self.motion=check.joint_motion
        if m.nflex or m.npair or self.motion is None:
            raise ValueError('pair bounds require rigid bodies without explicit geom pairs')
        entities=[];bounds=[];planes=[];types=[];affinities=[];selected=[];supports=[];objects=[]
        classifier=check.classifiers[True]
        for body in check.bodies:
            geoms=np.flatnonzero(m.geom_bodyid==body)
            groups=[[g]for g in geoms] if body==0 else [geoms]
            for group in groups:
                group=np.asarray(group);geom=int(group[0]) if body==0 else -1
                if geom<0:
                    if m.body_bvhadr[body]<0 or np.any(m.geom_type[group]==mujoco.mjtGeom.mjGEOM_PLANE):
                        raise ValueError('finite body bound unavailable')
                    bound=m.bvh_aabb[m.body_bvhadr[body]]
                else:bound=m.geom_aabb[geom]
                entities.append((int(body),geom));bounds.append(bound.copy())
                planes.append(geom>=0 and m.geom_type[geom]==mujoco.mjtGeom.mjGEOM_PLANE)
                types.append(int(np.bitwise_or.reduce(m.geom_contype[group],initial=0)))
                affinities.append(int(np.bitwise_or.reduce(m.geom_conaffinity[group],initial=0)))
                selected.append(bool(np.all(classifier.selected[group])))
                supports.append(bool(np.all(classifier.supports[group])))
                objects.append(len(group)==1 and group[0]==classifier.object)
        self.entities=np.array(entities);self.bounds=np.array(bounds);self.planes=np.array(planes)
        pairs=[];limits=[];excluded=set(map(int,m.exclude_signature))
        parent_filter=not (m.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_FILTERPARENT)
        for a,(ba,_) in enumerate(entities):
            for b in range(a+1,len(entities)):
                bb=entities[b][0];wa,wb=m.body_weldid[[ba,bb]]
                if wa==wb or not (m.body_dofnum[wa] or m.body_dofnum[wb]):continue
                if parent_filter and wa and wb and (wa==m.body_weldid[m.body_parentid[wb]] or wb==m.body_weldid[m.body_parentid[wa]]):continue
                if (min(ba,bb)<<16)+max(ba,bb) in excluded:continue
                if not ((types[a]&affinities[b]) or (types[b]&affinities[a])):continue
                if ((objects[a] and (selected[b] or supports[b])) or
                    (objects[b] and (selected[a] or supports[a]))):continue
                pairs.append((a,b));limits.append(.03 if supports[a] or supports[b] else .001)
        self.pairs=np.array(pairs,dtype=int).reshape(-1,2);self.limits=np.array(limits)
        a,b=self.pairs.T;rows=body_rows[self.entities[:,0]];self.rows=rows
        self.a,self.b=a,b
        la,lb=self.motion.levers[rows[a]],self.motion.levers[rows[b]]
        self.levers=(la+lb)*np.logical_xor(la>0,lb>0)
        self.different_roots=roots[rows[a]]!=roots[rows[b]]
        self.tree=IntervalTree()

    def certifies(self, data, half, hinge_change, root_movement):
        a,b=self.a,self.b;rows=self.rows
        movement=half*(self.levers@hinge_change)
        movement+=np.where(self.different_roots,root_movement[rows[a]]+root_movement[rows[b]],0.)
        unresolved=np.flatnonzero(movement>self.check.halo+1e-6)
        if not len(unresolved):return True
        bodies,geoms=self.entities.T;world=geoms>=0
        rotation=data.ximat[bodies].reshape(-1,3,3).copy();position=data.xipos[bodies].copy()
        rotation[world]=data.geom_xmat[geoms[world]].reshape(-1,3,3)
        position[world]=data.geom_xpos[geoms[world]]
        center=position+np.einsum('nij,nj->ni',rotation,self.bounds[:,:3])
        radius=np.einsum('nij,nj->ni',abs(rotation),self.bounds[:,3:])
        # Widen the bounding boxes to cover floating-point transform rounding.
        radius+=1e-12*(1+abs(center)+radius)
        i,j=a[unresolved],b[unresolved]
        distance=np.linalg.norm(np.maximum(abs(center[i]-center[j])-radius[i]-radius[j],0.),axis=1)
        for p,other in [(i,j),(j,i)]:
            mask=self.planes[p]
            if np.any(mask):
                normal=rotation[p[mask],:,2]
                local=np.einsum('nij,ni->nj',rotation[other[mask]],normal)
                support=np.sum(abs(local)*self.bounds[other[mask],3:],axis=1)
                distance[mask]=np.sum(normal*(center[other[mask]]-position[p[mask]]),axis=1)-support-1e-11
        valid=distance-movement[unresolved]>=self.limits[unresolved]-1e-6
        remaining=unresolved[~valid]
        if not len(remaining):return True
        entities=self.entities[self.pairs[remaining]].reshape(-1,4)
        required=self.limits[remaining]+movement[remaining]-1e-6+1e-8
        return self.tree.certifies(self.check.model,data,entities,required)



@lru_cache(maxsize=1)
def _library():
    source=Path(__file__).with_suffix('.cc');package=Path(mujoco.__file__).parent
    library=package/('libmujoco.so.'+mujoco.mj_versionString())
    digest=hashlib.sha256(source.read_bytes()+library.read_bytes()).hexdigest()[:16]
    binary=ROOT/'build'/('interval_bounds_'+digest+'.so')
    if not binary.exists():
        temporary=binary.with_suffix('.'+str(os.getpid())+'.so')
        subprocess.run(['c++','-O3','-std=c++17','-fPIC','-shared','-Wall','-Wextra',
            '-I'+str(package/'include'),str(source),str(library),'-Wl,-rpath,'+str(package),'-o',str(temporary)],check=True)
        os.replace(temporary,binary)
    return ctypes.CDLL(str(binary))

class IntervalTree:
    def __init__(self):
        self.library=_library();self.call=self.library.interval_leaves
        ptr=ctypes.c_void_p;integer=ctypes.c_int
        self.call.argtypes=[ptr,ptr,integer,ptr,ptr,integer,ptr];self.call.restype=integer
        self.separation=self.library.interval_separation
        self.separation.argtypes=[ptr,ptr,integer,integer,ptr];self.separation.restype=ctypes.c_double
        self.buffer=np.empty((4096,3),dtype=np.int32);self.queries=0;self.overflows=0
        self.witness=np.zeros(6)

    def certifies(self,model,data,entities,required):
        entities=np.ascontiguousarray(entities,dtype=np.int32);required=np.ascontiguousarray(required,dtype=float)
        count=self.call(model._address,data._address,len(required),entities.ctypes.data,required.ctypes.data,len(self.buffer),self.buffer.ctypes.data)
        if count<0:self.overflows+=1;return False
        for a,b,row in self.buffer[:count]:
            self.queries+=1
            # GJK supplies a useful direction, not a certified distance.
            cutoff=required[row]+np.linalg.norm(data.geom_xpos[a]-data.geom_xpos[b])+model.geom_rbound[a]+model.geom_rbound[b]
            mujoco.mj_geomDistance(model,data,int(a),int(b),float(cutoff),self.witness)
            lower=self.separation(model._address,data._address,int(a),int(b),self.witness.ctypes.data)
            if not np.isfinite(lower) or lower<required[row]:return False
        return True
