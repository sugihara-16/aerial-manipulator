"""Recheck a release command over its physical hold interval.

Only an invalid local response triggers a bounded input repair. This is not a
whole-task rollout or a structural candidate selector. Execution is audited
independently, including the final object pose and every physical substep.
"""
import numpy as np
import mujoco

from build.cached_response import CachedResponse
from build.response_contact_control import ResponseContactControl


def object_tracking_ratios(model, q, v, reference, velocity):
    """Separate object pose and velocity errors in the existing task units."""
    joint=model.joint('object').id
    qi,vi=int(model.jnt_qposadr[joint]),int(model.jnt_dofadr[joint])
    angle=np.empty(3);actual_rotation=np.empty(9);reference_rotation=np.empty(9)
    mujoco.mju_subQuat(angle,q[qi+3:qi+7],reference[qi+3:qi+7])
    mujoco.mju_quat2Mat(actual_rotation,q[qi+3:qi+7])
    mujoco.mju_quat2Mat(reference_rotation,reference[qi+3:qi+7])
    angular=actual_rotation.reshape(3,3)@v[vi+3:vi+6]-reference_rotation.reshape(3,3)@velocity[vi+3:vi+6]
    return np.array([np.linalg.norm(q[qi:qi+3]-reference[qi:qi+3])/.03,
        np.linalg.norm(angle)/.2,np.linalg.norm(v[vi:vi+3]-velocity[vi:vi+3])/.02,
        np.linalg.norm(angular)/.1])


def object_response_in_tracking_domain(model, q, v, reference, velocity, states):
    # Outside the domain, compare against the CURRENT excess, not an already
    # deteriorating nominal prediction. This is a local check, not invariance.
    allowed=np.maximum(1.,object_tracking_ratios(model,q,v,reference,velocity))
    for row in states:
        target=reference.copy();mujoco.mj_integratePos(model,target,velocity,float(row[0]))
        ratios=object_tracking_ratios(model,row[1:1+model.nq],row[1+model.nq:],target,velocity)
        if np.any(ratios>allowed+1e-7):return False
    return True


class HeldInputGuard:
    def __init__(self, plant, links, clearance_model, period=.0025, *, object_velocity_norm_limits=None, workers=4):
        steps = round(period / plant.mj.opt.timestep)
        self.control = ResponseContactControl(
            plant, links, period=period, iterations=2, prediction_steps=steps,
            objective='input', clearance_model=clearance_model,
            fast_solve=True, adaptive_damping=True)
        self.control.predictor = CachedResponse(plant, links, period=period, steps=steps, workers=workers)
        # Repair acceptance must be tighter than this guard and the audit.
        # A relative 1e-6 force excess would admit 30 micro-newtons above 30 N.
        self.control.feasibility_tolerance = 1e-8
        self.control.object_velocity_norm_limits=object_velocity_norm_limits
        self.repairs = 0

    def update(self, q, v, command, *, reference=None, velocity=None):
        self.last_request=dict(q=q.copy(),v=v.copy(),command=command.copy())
        self.last_request.update({key:value.copy() for key,value in
            [('reference',reference),('velocity',velocity)] if value is not None})
        g = self.control
        response = g.predictor.evaluate(q, v, command)
        m, states = g.m, response['states']
        qs, vs = states[:, 1:1+m.nq], states[:, 1+m.nq:]
        ratios = (np.linalg.norm(response['loads'][:, :, 1:4], axis=2)/30.,
                  np.linalg.norm(response['loads'][:, :, 4:7], axis=2)/5.,
                  abs(response['motors'][:, 0, :g.nj])/g.effort,
                  abs(vs[:, g.vi])/3., response['penetration']/.002)
        bad = max(float(np.max(x, initial=0.)) for x in ratios) > 1.+1e-7
        limits=getattr(g,'object_velocity_norm_limits',None)
        if limits is not None:
            start=int(m.jnt_dofadr[m.joint('object').id])
            bad=bad or any(np.max(np.linalg.norm(vs[:,start+offset:start+offset+3],axis=1))>limit*(1.+1e-8)
                for offset,limit in zip((0,3),limits))
        bad = bad or (np.max(g.qlo-qs[:, g.qi], initial=0.) > 1e-7
                or np.max(qs[:, g.qi]-g.qhi, initial=0.) > 1e-7
                or bool(g.clearance.check_many(np.vstack([q, qs]), True)[0]))
        g.object_tracking_domain=None
        if reference is not None:
            from build.object_tracking_domain import ObjectTrackingDomain
            velocity=np.zeros(m.nv) if velocity is None else velocity
            g.object_tracking_domain=ObjectTrackingDomain(m,q,v,reference,velocity)
            bad=bad or not object_response_in_tracking_domain(m,q,v,reference,velocity,states)
        if bad:
            command = g.update(q, v, q, input_reference=command,
                               normal_targets=np.zeros(len(g.links)))
            self.repairs += 1
            if reference is not None:response=g.response
        if reference is not None:
            velocity=np.zeros(m.nv) if velocity is None else velocity
            if not object_response_in_tracking_domain(m,q,v,reference,velocity,response['states']):
                raise ValueError('partial-contact input leaves object tracking domain')
        g.previous=command.copy()
        return command

    def close(self):
        self.control.predictor.close()
