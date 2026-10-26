"""Low-speed landing reference until observed geometric contact.

The endpoint of a zero-speed position reference can remain above the support
because of tracking error. Continue down at the acquisition approach speed;
the caller switches modes on actual geometric contact. The extra reference
travel is capped by the 3 cm task position tolerance. Actual penetration
remains limited to 2 mm by independent physical checks.
"""
import numpy as np
from build.local_translation import PredictiveTranslation


class TouchdownTranslation(PredictiveTranslation):
    def reference(self,time):
        q,v,u,s=super().reference(time)
        elapsed=max(0.,time-2.-self.duration)
        depth=min(.03,.002*elapsed)
        if elapsed:
            for j in self.free:
                q[self.design.m.jnt_qposadr[j]+2]-=depth
                v[self.design.m.jnt_dofadr[j]+2]=-.002 if depth<.03 else 0.
        return q,v,u,s
