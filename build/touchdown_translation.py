"""Low-speed landing reference until observed geometric contact.

The endpoint of a zero-speed position reference can remain above the support
because of tracking error. Continue down at the acquisition approach speed;
the caller switches modes on actual geometric contact. The extra reference
travel is capped by the existing 2 mm penetration limit, not case identity.
"""
import numpy as np
from build.local_translation import PredictiveTranslation


class TouchdownTranslation(PredictiveTranslation):
    def reference(self,time):
        q,v,u,s=super().reference(time)
        elapsed=max(0.,time-2.-self.duration)
        depth=min(.002,.0002*elapsed)
        if elapsed:
            for j in self.free:
                q[self.design.m.jnt_qposadr[j]+2]-=depth
                v[self.design.m.jnt_dofadr[j]+2]=-.0002 if depth<.002 else 0.
        return q,v,u,s
