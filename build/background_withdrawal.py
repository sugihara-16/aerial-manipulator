"""Prepare the existing withdrawal while unloaded control keeps running.

The prepared geometry still comes from an observed pose. Its later handoff is
checked by independent execution, not assumed exact after the planning wait.
The existing rotor-direction transition remains a separate legacy branch.
"""
import numpy as np
from build.hybrid_withdrawal import HybridWithdrawal
from build.background_plan import BackgroundPlan
from build.unloaded_hover import command as gravity_command
from build.algebraic_release import coupled


class BackgroundHybridWithdrawal(HybridWithdrawal):
    def __init__(self,*args,parallel_process=False,**kwargs):
        super().__init__(*args,**kwargs);self.job=None;self.wait=None;self.parallel_process=parallel_process

    def _prepare_copy(self,q,v,previous):
        prepared=HybridWithdrawal(self.plan,self.plant,q)
        prepared.prepare(q,v,previous)
        return prepared

    def _prepare_path(self,q,v,previous):
        prepared=self._prepare_copy(q,v,previous)
        return dict(q=q,**{name:getattr(prepared,name)for name in ['path_data','durations','ends','planning_s','info']})

    def _restore_path(self,payload,q):
        q=payload.pop('q')
        prepared=HybridWithdrawal(self.plan,self.plant,q)
        for name,value in payload.items():setattr(prepared,name,value)
        prepared.query.qpos[:]=q;qp,_=self.plant.observation(prepared.query)
        prepared.flight=coupled.CoupledFlightController(self.plant.pin,self.plant.meta,qp,track_vectoring=True,planned_input_reference=True)
        prepared.ready=True
        return prepared

    def update(self,q,v,previous):
        if self.transition is None and not self.ready and self.elapsed>=.3-1e-9:
            if self.job is None:
                if self.parallel_process:
                    from build.process_plan import ProcessPlan
                    self.job=ProcessPlan(self.elapsed,self._prepare_path,q.copy(),v.copy(),previous.copy())
                else:self.job=BackgroundPlan(self.elapsed,self._prepare_copy,q.copy(),v.copy(),previous.copy())
            if not self.job.ready(self.elapsed):
                command=gravity_command(self.local,q,previous)[0]
                self.elapsed+=.01;return command
            prepared,self.wait=self.job.result(self.elapsed);self.job=None
            if self.parallel_process:prepared=self._restore_path(prepared,q)
            prepared.elapsed=self.elapsed;prepared.entry=self.elapsed
            prepared.flight.command=np.where(self.local.dock,previous[:self.local.nj]/self.local.kp,previous[:self.local.nj])
            prepared.flight.force=previous[self.local.nj:].copy()
            self.transition=prepared
        command=super().update(q,v,previous)
        if self.wait is not None:self.info=dict(self.info,background_plan=self.wait)
        return command

    def close(self):
        if self.job is not None:self.job.close();self.job=None
