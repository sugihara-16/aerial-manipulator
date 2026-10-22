"""Development-only batching of the unchanged DrivePrediction interval."""
import ctypes
import json
import tempfile
from pathlib import Path
import numpy as np
from amsrr_det.drive_prediction import DrivePrediction

_lib=ctypes.CDLL(str(Path(__file__).parent/'prediction_native_avx_fixed/libdrive_interval.so'))
_pointer=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
_lib.prediction_new.argtypes=[ctypes.c_char_p,ctypes.c_char_p]
_lib.prediction_new.restype=ctypes.c_void_p
_lib.prediction_delete.argtypes=[ctypes.c_void_p]
_lib.prediction_error.restype=ctypes.c_char_p
_lib.prediction_interval.argtypes=[ctypes.c_void_p,*[_pointer]*5,ctypes.c_double,ctypes.c_int,ctypes.c_int,_pointer]
_lib.prediction_interval.restype=ctypes.c_int


class NativeDriveInterval(DrivePrediction):
    def __init__(self,model,meta):
        super().__init__(model,meta)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            path=Path(temporary)/'model.bin';model.saveToBinary(str(path))
            self.handle=_lib.prediction_new(str(path).encode(),json.dumps(meta).encode())
        if not self.handle:raise ValueError(_lib.prediction_error().decode())
        self.fallbacks=0;self.calls=0

    def __del__(self):
        if getattr(self,'handle',None):_lib.prediction_delete(self.handle)

    def interval(self,q,v,command,force,period=.01,step=.001,*,linearize=False,history=False,generalized_load=None):
        self.calls+=1
        count=round(period/step)
        if count<1 or abs(count*step-period)>1e-12:raise ValueError('prediction interval must contain whole steps')
        nq,nv,na,nr=self.model.nq,self.model.nv,len(command),len(force);nu=na+nr
        arrays=[np.ascontiguousarray(a,dtype=np.float64) for a in [q,v,command,force,np.zeros(nv) if generalized_load is None else generalized_load]]
        if any(a.shape!=(size,) or not np.isfinite(a).all() for a,size in zip(arrays,[nq,nv,len(self.qi),len(self.meta['rotors']),nv])):
            raise ValueError('invalid predictor input')
        width=nq+2*nv+na+(2*nv*nu if linearize else 0)
        output=np.empty((count,width))
        iterations=_lib.prediction_interval(self.handle,*arrays,step,count,int(linearize),output)
        if iterations<0:
            error=_lib.prediction_error().decode()
            if error!='active_set_fallback_required':raise ValueError(error)
            self.fallbacks+=1
            return super().interval(q,v,command,force,period,step,linearize=linearize,history=history,generalized_load=generalized_load)
        samples=[];torque_start=nq+nv;acc_start=torque_start+na;jac_start=acc_start+nv
        for row in output if history else output[-1:]:
            sample=dict(q=row[:nq].copy(),v=row[nq:nq+nv].copy())
            if linearize:
                sample.update(position_wrt_input=row[jac_start:jac_start+nv*nu].reshape(nv,nu).copy(),
                              velocity_wrt_input=row[jac_start+nv*nu:].reshape(nv,nu).copy())
            samples.append(sample)
        last=samples[-1]
        info=dict(torque=output[-1,torque_start:acc_start].copy(),acceleration=output[-1,acc_start:jac_start].copy(),
                  iterations=iterations,mean_torque=output[:,torque_start:acc_start].mean(axis=0))
        if linearize:info.update(position_wrt_input=last['position_wrt_input'],velocity_wrt_input=last['velocity_wrt_input'])
        if history:info['samples']=samples
        return last['q'],last['v'],info
