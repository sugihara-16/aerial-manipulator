"""Separated wake-characterisation experiment, not an assembly evaluation."""
import json,os
from pathlib import Path
import numpy as np
import mujoco,pinocchio as pin
import scripts.run_assembly_task as runner
from amsrr_det.assembly import make_assembly_plant

class Hover:
    def __init__(self,plan):
        self.plan=plan;self.stage='HOVER_PROBE';self.clock=0.;self.success=False;self.failure=None;self.retries=0;self.errors={};self.events=[];self.control_ready=False;self.hover_observations=[]
    def update(self,d):
        q=self.plan['initial'];qi=self.plan['qi'];vi=self.plan['vi']
        self.errors=dict(position_max_m=max(float(np.linalg.norm(d.qpos[k:k+3]-q[k:k+3]))for k in qi),
                         linear_speed_max_m_s=max(float(np.linalg.norm(d.qvel[j:j+3]))for j in vi))
        if d.time>=5.:
            self.success=self.errors['position_max_m']<.04 and self.errors['linear_speed_max_m_s']<.06
            if not self.success:self.failure='hover_error'
        return q,np.zeros(len(d.qvel)),np.zeros(len(d.qvel))

def prepare(case,timestep):
    p=make_assembly_plant(2,[],timestep=timestep);q=pin.neutral(p.pin)
    for b in p.meta['bases']:
        k=b['q'];i=b['module'];q[k:k+3]=[case['offset_m']*i,0.,1.5+.5*i]
    initial=p.configuration(q);m=p.mj
    qi=np.array([m.jnt_qposadr[m.joint(p.pin.names[b['joint']]).id]for b in p.meta['bases']]);vi=np.array([m.jnt_dofadr[m.joint(p.pin.names[b['joint']]).id]for b in p.meta['bases']])
    return dict(plant=p,edges=[],initial=initial,closed=initial,qi=qi,vi=vi,rounds=[],evidence=dict(scope='two-module hover airflow experiment'))
runner.prepare=prepare;runner.AssemblyFSM=Hover;os.sched_setaffinity(0,{12})
for label,offset,strength in [('stack_no_air',0.,0.),('stack_air',0.,1.),('offset_air',1.,1.)]:
    case=dict(id=label,modules=2,edges=[],initial_seed=873,offset_m=offset)
    result=runner.run(case,Path('artifacts/assembly_20261106/air_probes_central27')/label,worker_cpu=14,air_strength=strength)
    print(json.dumps(result),flush=True)
