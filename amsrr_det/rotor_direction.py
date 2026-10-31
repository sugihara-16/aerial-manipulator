"""Local rotor direction sensitivity for the shared input allocation.

This is an allocation approximation, not future contact simulation. A damped
motor response maps bounded present effort to a small direction change. The
actual simulator still applies the authored motor, inertia and effort limit.
"""
import numpy as np
import mujoco


class RotorDirection:
    def __init__(self, drive, horizon=.1):
        self.drive=drive;self.horizon=horizon
        self.gi=np.array([r['gimbal'] for r in drive.plant.meta['rotors']])
        self.data=mujoco.MjData(drive.m)
        self.jac=np.zeros((3,drive.m.nv))
        self.values=np.zeros((2,drive.m.nv,len(self.gi)))

    def linearize(self,q,v,thrust):
        dr=self.drive;m=dr.m;d=self.data;eps=1e-5
        # Each rotor belongs to its own vectoring hinge. Perturbing all such
        # hinges at once differentiates each rotor column with respect to its
        # own angle, including ancestor joint reactions and moving lever arm.
        for k,sgn in enumerate([-1.,1.]):
            d.qpos[:]=q;d.qpos[dr.qi[self.gi]]+=sgn*eps
            mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
            for j,site in enumerate(dr.sites):
                mujoco.mj_jacSite(m,d,self.jac,None,site)
                self.values[k,:,j]=self.jac.T@d.site_xmat[site].reshape(3,3)[:,2]
        derivative=(self.values[1]-self.values[0])/(2*eps)*np.asarray(thrust)
        inertia=np.diag(dr.mass)[dr.vi[self.gi]]
        kd=dr.kd[self.gi];kp=dr.kp[self.gi]
        # Linear damped second-order servo with fixed command over horizon.
        # Integrate its two real roots analytically; no numerical rollout.
        disc=np.sqrt((kd*kd-4*inertia*kp).astype(complex))
        r1=(-kd+disc)/(2*inertia);r2=(-kd-disc)/(2*inertia)
        den=r1-r2
        critical=abs(den)<1e-8;safe=np.where(critical,1.,den)
        position=((r1*np.exp(r2*self.horizon)-r2*np.exp(r1*self.horizon))/safe).real
        speed=((np.exp(r1*self.horizon)-np.exp(r2*self.horizon))/safe).real
        root=-kd/(2*inertia)
        position[critical]=((1-root*self.horizon)*np.exp(root*self.horizon))[critical]
        speed[critical]=(self.horizon*np.exp(root*self.horizon))[critical]
        gain=(1-position)/kp
        # Command already compensates measured velocity in the MIT mapping.
        # Actuator effort first balances gravity and the current rotor load.
        # A trim effort is not an acceleration: at a stationary supported
        # reference its predicted angle change must be exactly zero.
        holding=dr.d.qfrc_bias[dr.vi[self.gi]]-dr.B[np.ix_(dr.vi[self.gi],np.arange(dr.nj,m.nu))]@np.asarray(thrust)
        drift=(speed+gain*kd)*v[dr.vi[self.gi]]-gain*holding
        B=dr.B.copy();B[:,self.gi]+=derivative*gain
        return B,derivative@drift
