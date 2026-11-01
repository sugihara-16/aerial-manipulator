"""Privileged valve evidence, isolated from the controller and selector."""
import numpy as np
import mujoco
from .task_geometry import box_face_mask


def side_face_evidence(rows):
    """Audit loaded phases without stopping execution; a sampled check.

    Forces below one micronewton are numerical zero, not a usable load.
    """
    loaded=[r for r in rows if r['phase'] in ('LOAD','TURN','SETTLE','UNLOAD')]
    violations=[r for r in loaded if r['non_side_normal_force']>1e-6]
    return dict(passes=bool(loaded) and not violations,samples=len(loaded),
        violating_samples=len(violations),maximum_non_side_normal_force_n=max(
            (r['non_side_normal_force'] for r in loaded),default=0.),
        maximum_non_side_axis_torque_nm=max((abs(r['non_side_axis_torque']) for r in loaded),default=0.),
        sample_period_s=.1,normal_force_zero_tolerance_n=1e-6,
        phases=['LOAD','TURN','SETTLE','UNLOAD'])


class ValveAudit:
    def __init__(self,system):
        self.system=system;m=system.m;plant=system.control
        self.shape=system.spec.shape
        self.object=m.body('object').id
        self.root=int(m.jnt_bodyid[next(i for i in range(m.njnt) if m.jnt_type[i]==mujoco.mjtJoint.mjJNT_FREE)])
        self.sites=[m.site(plant.pin.frames[r['frame']].name).id for r in plant.meta['rotors']]
        self.nj=len(plant.meta['actuators']);self.rows=[];self.states=[]
        self.probe=mujoco.MjData(m)

    def record(self,state,phase,applied_solution,applied_record,reference=None):
        m=self.system.m;axis=np.asarray(self.system.spec.axis);center=np.asarray(self.system.spec.center)
        # Integration buffers contain forces from the previous substep but
        # positions from its end. Re-evaluate a separate diagnostic state so
        # forces, geometry and the currently applied input share one instant.
        # This neither changes the plant nor supplies controller observations.
        d=self.probe;mujoco.mj_copyData(d,m,state);mujoco.mj_forward(m,d)
        root_center=d.subtree_com[self.root];on_object=np.zeros(6);on_robot=np.zeros(6)
        local=np.zeros(6)
        other_axis_torque=0.;other_normal_force=0.
        object_rotation=d.xmat[self.object].reshape(3,3)
        for index,c in enumerate(d.contact):
            bodies=m.geom_bodyid[np.asarray(c.geom)]
            if self.object not in bodies:continue
            other=int(bodies[1] if bodies[0]==self.object else bodies[0])
            if not (m.body(other).name or '').startswith('m'):continue
            mujoco.mj_contactForce(m,d,index,local)
            force=c.frame.reshape(3,3).T@local[:3]*(1. if bodies[1]==self.object else -1.)
            moment=c.frame.reshape(3,3).T@local[3:]*(1. if bodies[1]==self.object else -1.)
            on_object+=np.r_[force,np.cross(c.pos-center,force)+moment]
            on_robot+=np.r_[-force,np.cross(c.pos-root_center,-force)-moment]
            normal_to_object=c.frame.reshape(3,3)[0]*(1. if bodies[1]==self.object else -1.)
            object_point=(c.pos+.5*c.dist*normal_to_object-d.xpos[self.object])@object_rotation
            object_geom=int(c.geom[0] if bodies[0]==self.object else c.geom[1])
            if object_geom!=m.geom('object').id or not box_face_mask(
                    object_point,self.shape.size,1,normal_to_object@object_rotation,c.dist):
                other_axis_torque+=float(axis@(np.cross(c.pos-center,force)+moment))
                other_normal_force+=abs(float(local[0]))
        rotor_force=d.site_xmat[self.sites].reshape(-1,3,3)[:,:,2]*d.actuator_force[self.nj:,None]
        rotor=np.r_[rotor_force.sum(axis=0),np.cross(d.site_xpos[self.sites]-root_center,rotor_force).sum(axis=0)]
        mujoco.mj_subtreeVel(m,d)
        momentum=np.r_[d.subtree_linvel[self.root]*m.body_subtreemass[self.root],d.subtree_angmom[self.root]]
        last=applied_solution;requested=np.asarray(last.get('requested_object_wrench',np.zeros(6)))
        allocated=requested+np.asarray(last.get('wrench_error',np.zeros(6)))
        application=applied_record or {}
        self.rows.append(dict(time_s=float(d.time),force_time_s=float(d.time),phase=phase,
            force_scope='instantaneous physical-model reevaluation at saved state and applied input',
            angle=float(d.qpos[self.system.oi]),speed=float(d.qvel[self.system.ov]),
            request_axis_torque=float(axis@requested[3:]),allocation_axis_torque=float(axis@allocated[3:]),
            contact_axis_torque=float(axis@on_object[3:]),object_contact_wrench=on_object.tolist(),
            non_side_axis_torque=other_axis_torque,non_side_normal_force=other_normal_force,
            robot_contact_wrench=on_robot.tolist(),rotor_wrench=rotor.tolist(),
            gravity_wrench=np.r_[m.body_subtreemass[self.root]*m.opt.gravity,np.zeros(3)].tolist(),
            momentum=momentum.tolist(),observation_s=application.get('observation_s'),
            application_s=application.get('application_s',application.get('application_tick_s')),
            reference=None if reference is None else np.asarray(reference).tolist()))
        self.states.append((d.qpos.copy(),d.qvel.copy(),d.ctrl.copy(),d.actuator_force.copy()))

    def save(self,out):
        import json
        (out/'physical_trace.json').write_text(json.dumps(self.rows,indent=2))
        if self.states:
            q,v,u,effort=map(np.array,zip(*self.states))
            np.savez_compressed(out/'physical_states.npz',q=q,v=v,command=u,effort=effort)
