"""Preserve combinations of joint angles that carry the planned floor load.

The feature is the local change of required joint torque at fixed world ground
forces and fixed rotor thrust. It is not measured contact force, nor a claim
that every perturbed contact state is executable. Full physics remains audit.
"""
import time,numpy as np,pinocchio as pin
from amsrr_det.walking_load import motion_load,ground_contacts,support_switch_times
from amsrr_det.equilibrium import static_balance

import os,multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor


def _sample_rows(plant,control,reference,strength,times):
    values=[];failed=0
    model,meta=plant.pin,plant.meta;data=model.createData();vi=np.array([a['v']for a in meta['actuators']]);dock=np.array([a['role']=='dock'for a in meta['actuators']]);limits=np.array([a['effort']for a in meta['actuators']]);dr=control.drive
    for t in times:
     q,v,acc=reference.rates(t);qp,load=motion_load(plant,dr,q,v,acc);contacts=ground_contacts(control,reference.feet(t)[3]);force=static_balance(model,meta,qp,contacts,generalized_load=load,fast_lp=True,minimize_burden=True,peak_effort_bound=.99)
     K=np.zeros((len(vi),plant.mj.nv))
     if force.get('minimum_peak_effort_ratio',999)>.99:failed+=1;values.append(K);continue
     # Tangent-space transport agrees with motion_load: only free translation
     # differs between the Pinocchio and MuJoCo velocity conventions here.
     _,pv=plant.observation(dr.d);_,_,pvi,mvi,bases=plant.coordinate_maps;pa=np.empty(model.nv);pa[pvi]=acc[mvi]
     for pos,vel,source in bases:
      R=pin.Quaternion(qp[pos+3:pos+7]).matrix();pa[vel:vel+3]=R.T@acc[source:source+3]-np.cross(pv[vel+3:vel+6],pv[vel:vel+3])
     pin.computeJointJacobians(model,data,qp);pin.updateFramePlacements(model,data)
     external=[pin.Force.Zero()for _ in range(model.njoints)];extra=np.zeros((model.nv,model.nv))
     for r,T in zip(meta['rotors'],force['thrust']):
      frame=model.frames[r['frame']];external[frame.parentJoint]+=frame.placement.act(pin.Force(np.array([0.,0.,T]),np.zeros(3)))
     for contact,value in zip(contacts,force['contacts']):
      f=meta['frames'][contact['frame']];frame=model.frames[f];world=np.array(value['force_world']);point=np.array(contact['point']);local=data.oMf[f].rotation.T@world
      external[frame.parentJoint]+=frame.placement.act(pin.Force(local,np.cross(point,local)))
      J=pin.getFrameJacobian(model,data,f,pin.LOCAL_WORLD_ALIGNED);Jp=J[:3]-pin.skew(data.oMf[f].rotation@point)@J[3:];extra-=Jp.T@pin.skew(world)@J[3:]
     derivative=pin.computeRNEADerivatives(model,data,qp,pv,pa,external)[0].copy()+extra
     # Only internal Dock combinations are restored; the task controls base,
     # gimbals and feet. Keep true singular values so weak directions stay free.
     K[:,dr.vi[dock]]=derivative[np.ix_(vi,vi[dock])]/limits[:,None]*strength
     values.append(K)
    return np.array(values),failed


_worker_context=None

def _initialize_rows(plant,control,reference,strength,cpus):
    global _worker_context
    os.sched_setaffinity(0,cpus)
    _worker_context=(plant,control,reference,strength)


def _row_batch(times):
    return _sample_rows(*_worker_context,times)


class LoadPathReference:
    def __init__(self,plant,control,reference,*,strength=.5,cpus=None):
        started=time.perf_counter();self.period=reference.period;g=reference.gait
        self.times=np.unique(np.r_[0.,self.period,np.arange(64)*self.period/64,
            support_switch_times(self.period,len(control.links),g['duty'],g['phases'])])
        workers=min(4,len(cpus)) if cpus else 1
        if workers>1:
            # Independent reference rows use identical equations. Fork only
            # during initial preparation, before background control starts.
            with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('fork'),
                    initializer=_initialize_rows,initargs=(plant,control,reference,strength,cpus)) as pool:
                rows=list(pool.map(_row_batch,np.array_split(self.times,workers)))
            self.values=np.concatenate([r[0] for r in rows]);failed=sum(r[1] for r in rows)
            # Match the serial calculation's final controller observation.
            motion_load(plant,control.drive,*reference.rates(self.times[-1]))
        else:
            self.values,failed=_sample_rows(plant,control,reference,strength,self.times)
        self.report=dict(seconds=time.perf_counter()-started,samples=len(self.times),failed=failed,
            strength=strength,max_row_norm=float(np.linalg.norm(self.values,axis=2).max()),workers=workers)

    def __call__(self,time):
        t=time%self.period;k=min(np.searchsorted(self.times,t,side='right')-1,len(self.times)-2)
        blend=(t-self.times[k])/(self.times[k+1]-self.times[k])
        return (1-blend)*self.values[k]+blend*self.values[k+1]
