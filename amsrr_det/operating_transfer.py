"""Continuous introduction of a qualified planned grasp operating point.

The previous motor inputs remain actual issued values. No simulated contact
force, physical state reset, or online force estimate is used here.
"""
import numpy as np
import pinocchio as pin


def pose(q):
    return pin.SE3(pin.Quaternion(q[3:7]).matrix(),q[:3])


class OperatingTransfer:
    def __init__(self,parameters):
        self.parameters={k:np.asarray(v) for k,v in parameters.items()}
        offset=self.parameters['base_offset']
        self.end_offset=pin.SE3(pin.exp3(offset[3:]),offset[:3])
        self.previous_offset=pin.SE3.Identity()
        self.started=False

    def start(self,grasp,flight,q,reference_velocity):
        # Keep the reference used to issue the final acquisition command.
        nu=np.zeros(grasp.model.nv) if reference_velocity is None else reference_velocity.copy()
        grasp.last_velocity=np.r_[nu[:6],nu[grasp.vi]]
        self.initial_preload=grasp.geometry(q)[-1].copy()
        self.initial_bias=flight.kp*(flight.command-grasp.reference[flight.qi])-flight.kd*nu[flight.vi]
        self.initial_force=flight.force.copy()
        self.initial_gimbal=flight.command[flight.ai].copy()
        self.started=True

    def inputs(self,load):
        if not self.started:raise ValueError('operating-point handoff has not started')
        p=self.parameters
        return ((1-load)*self.initial_preload+load*p['preload'],
                (1-load)*self.initial_bias+load*p['joint_bias'],
                (1-load)*self.initial_force+load*p['thrust'],
                (1-load)*self.initial_gimbal+load*p['gimbal'])

    def compose(self,q,previous_reference,target,velocity,load,dt):
        offset=pin.exp6(pin.log6(self.end_offset).vector*load)
        before=pose(previous_reference)*self.previous_offset
        after=pose(target)*offset
        result=target.copy();result[:3]=after.translation;result[3:7]=pin.Quaternion(after.rotation).coeffs()
        rate=velocity.copy();rate[:3]=(after.translation-before.translation)/dt
        rate[3:6]=pin.log3(after.rotation@before.rotation.T)/dt
        if np.linalg.norm(result[:3]-q[:3])>.05+1e-8:
            raise ValueError('composed_base_reference_bound')
        self.previous_offset=offset
        return result,rate

    def release(self,grasp,target,velocity):
        # Bake the already issued reference into the existing release bridge.
        grasp.reference=target.copy()
        grasp.last_velocity=np.r_[velocity[:6],velocity[grasp.vi]]
