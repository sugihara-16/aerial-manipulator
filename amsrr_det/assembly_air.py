"""Uncalibrated rotor-resolved interference model for docking tests.

Commands mean still-air thrust. Foreign wakes reduce that thrust at the rotor
site; their off-centre application already creates moments. No duplicate
vehicle-level down-force or moment is added. Same-module interference belongs
to the isolated-vehicle calibration and is excluded here.

The spreading profile/induced-velocity scale follows momentum/jet models;
near-field coefficients are declared engineering assumptions, NOT identified
Holon aerodynamics. Fixed-frequency random phases make time-step comparisons
use the same continuous disturbance rather than different random draws.
"""
from dataclasses import dataclass
import numpy as np
import mujoco


@dataclass(frozen=True)
class AirParameters:
    density: float = 1.225
    radius: float = .0778225663  # swept radius of authored thrust_1.STL, metres
    spreading: float = .1
    inflow_loss: float = .22
    fluctuation: float = .25
    maximum_loss: float = .65
    strength: float = 1.
    response_s: float = .04  # engineering assumption for unsteady wake history


def interference(positions, axes, thrust, modules, phases, now, parameters):
    """Return fractional losses and mean incident-speed proxy, per rotor."""
    p=parameters;R=p.radius
    delta=positions[:,None,:]-positions[None,:,:]  # receiver minus source
    axial=-np.einsum('ijk,jk->ij',delta,axes)
    radial2=np.maximum(0.,np.sum(delta*delta,axis=2)-axial*axial)
    downstream=np.maximum(axial,0.)
    width=R+p.spreading*downstream
    radial=(1+(np.sqrt(2)-1)*radial2/width**2)**-2
    # Bounded individual jets in the near field, with momentum-conserving
    # axial decay/spreading downstream. Upstream inflow decays faster.
    disk_gain=1+axial/np.sqrt(axial*axial+R*R)
    gain=disk_gain*R/width
    same=modules[:,None]==modules[None,:]
    overlap=radial*gain;overlap[same]=0.
    vi=np.sqrt(np.maximum(thrust,0.)/(2*p.density*np.pi*R*R))
    velocity=-np.einsum('ij,j,jk->ik',overlap,vi,axes)
    incoming=np.maximum(0.,-np.sum(velocity*axes,axis=1))
    ratio=incoming/np.maximum(vi,1.)
    oscillation=(np.sin(2*np.pi*2.3*now+phases[:,0])+
                 .5*np.sin(2*np.pi*6.7*now+phases[:,1]))/1.5
    loss=p.strength*p.inflow_loss*ratio**2*(1+p.fluctuation*oscillation)
    return np.clip(loss,0.,p.maximum_loss),incoming


class AssemblyAir:
    def __init__(self,plant,seed,parameters=AirParameters()):
        self.m=plant.mj;self.parameters=parameters
        rotors=plant.meta['rotors'];self.nj=len(plant.meta['actuators'])
        self.sites=np.array([self.m.site(plant.pin.frames[r['frame']].name).id for r in rotors])
        self.bodies=self.m.site_bodyid[self.sites]
        self.modules=np.array([r['module'] for r in rotors])
        self.phases=np.random.default_rng(seed).uniform(-np.pi,np.pi,(len(rotors),2))
        self.loss=np.zeros(len(rotors));self.maximum=0.;self.integral=0.;self.next_update=-1.;self.updated=None
        self.step=.002

    def apply(self,data):
        if data.time+1e-10 < self.next_update: return
        self.next_update=data.time+self.step
        positions=data.site_xpos[self.sites]
        axes=data.site_xmat[self.sites].reshape(-1,3,3)[:,:,2]
        # The original positive thrust actuator still executes/clips the
        # still-air command. A single negative correction models its loss.
        thrust=np.maximum(0.,data.ctrl[self.nj:])
        target,incoming=interference(positions,axes,thrust,self.modules,
            self.phases,data.time,self.parameters)
        dt=self.step if self.updated is None else float(data.time-self.updated)
        self.updated=float(data.time)
        alpha=1. if self.parameters.response_s<=0 else -np.expm1(-dt/self.parameters.response_s)
        self.loss+=alpha*(target-self.loss)
        data.xfrc_applied[:]=0.
        force=-self.loss[:,None]*thrust[:,None]*axes
        for i,bid in enumerate(self.bodies):
            data.xfrc_applied[bid,:3]+=force[i]
            data.xfrc_applied[bid,3:]+=np.cross(positions[i]-data.xipos[bid],force[i])
        self.maximum=max(self.maximum,float(self.loss.max(initial=0.)))
        self.integral+=dt*float(np.sum(thrust*self.loss))
