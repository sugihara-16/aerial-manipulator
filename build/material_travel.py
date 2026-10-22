"""Independent pose-only record of material-point displacement after contact.

Anchors are captured once per pad at its first geometric contact. They are
never reset during acquisition/refinement. This is displacement of the saved
material pairs, not a measurement of an individual contact force or stick mode.
"""
import numpy as np
from build.sticking_design import LocalGrasp


class MaterialTravel:
    def __init__(self,plant,links):
        self.plant=plant;self.links=links;self.local=None;self.captured=set()
        self.peak=np.zeros(len(links));self.path=None;self.previous=None

    def update(self,q):
        if len(self.captured)<len(self.links):
            current=LocalGrasp(self.plant,q,self.links)
            added=[e for e in current.elements if e[0] not in self.captured]
            if self.local is None:
                self.local=current;self.local.elements=[]
            self.local.elements.extend(added);self.captured.update(e[0]for e in added)
        self.local.at(q);values=self.local.contact()[1].reshape(-1,2)
        if self.previous is None:self.previous=np.zeros_like(values);self.path=np.zeros(len(values))
        if len(self.previous)<len(values):
            count=len(values)-len(self.previous)
            self.previous=np.vstack([self.previous,np.zeros((count,2))]);self.path=np.r_[self.path,np.zeros(count)]
        self.path+=np.linalg.norm(values-self.previous,axis=1);self.previous=values.copy()
        for i,e in enumerate(self.local.elements):self.peak[e[0]]=max(self.peak[e[0]],float(np.linalg.norm(values[i])))

    def result(self):
        return dict(anchors_captured_pads=sorted(self.captured),maximum_displacement_m=self.peak.tolist(),
                    maximum_sampled_path_m=float(self.path.max(initial=0.)) if self.path is not None else 0.)
