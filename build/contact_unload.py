"""Transfer the supported payload before switching to unloaded flight.

Keep the grasp controller during a continuous input transfer. Releasing pads
have no positive retention floor; the independent task audit still checks all
loads, collisions and final object motion. This is not a stability certificate.
"""
import numpy as np
from build.background_withdrawal_hold import HeldBackgroundWithdrawal
from build.unloaded_hover import command as gravity_command
from build.sticking_design import LocalGrasp
from build.constrained_momentum_feedback import ConstrainedMomentumFeedback
from build.planned_withdrawal import PlannedWithdrawal
from build.check_local_pickup import smooth


class ContactUnload:
    def __init__(self,plan,plant,q,**kwargs):
        self.plan,self.plant,self.q=plan,plant,q.copy();self.options=kwargs
        links=[f'm{i}/'+('yaw_dock_mech'if p.startswith('yaw')else'pitch_dock_mech')+p[-1]for i,p in plan.surfaces]
        self.local=LocalGrasp(plant,q,links);self.elapsed=0.;self.release=None
        self.ready=False;self.complete=False;self.info={};self.feedback=None

    def update(self,q,v,previous):
        if self.release is not None:
            result=self.release.update(q,v,previous)
            self.ready,self.complete,self.info=self.release.ready,self.release.complete,self.release.info
            return result
        if self.feedback is None:
            self.first=previous.copy()
            self.feedback=ConstrainedMomentumFeedback(self.local,q,previous,period=.01,contact_coupled=True)
            self.feedback.contact_response.minimum_normal_load=0.
        try:empty=gravity_command(self.local,q,previous)[0]
        except ValueError as error:
            if str(error)!='no unloaded gravity allocation inside motor bounds':raise
            self.release=PlannedWithdrawal(self.plan,self.plant,q)
            return self.update(q,v,previous)
        # The existing continuous-release design uses at least two seconds.
        # Preserve the last issued input at entry, then transfer its nominal
        # effort/thrust while retaining feedback and the finite contact model.
        ratio=smooth(self.elapsed/2.)
        command=self.feedback.update(q,v,self.q,input_reference=(1-ratio)*self.first+ratio*empty)
        self.elapsed+=.01
        self.info=dict(unload_ratio=ratio,allocation=self.feedback.diagnostics)
        if self.elapsed>=2.:
            self.release=HeldBackgroundWithdrawal(self.plan,self.plant,q,**self.options)
            self.release.elapsed=.3  # continuous transfer has replaced the instantaneous unload stage
        return command
