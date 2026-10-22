"""Minimal independent PhysX articulation diagnostic from the owned rigid model.

No robot, inertia, joint or drive is imported from the mainline workspace.
Collision geometry and a complete task runner are intentionally separate.
"""
import numpy as np
import pinocchio as pin
from pxr import Usd,UsdGeom,UsdPhysics,UsdUtils,PhysxSchema,Gf
import omni.usd
import carb.settings
from omni.physx import get_physx_simulation_interface,get_physx_interface
import omni.physics.tensors as tensors


def quat(R):
    q=pin.Quaternion(np.asarray(R)).coeffs()
    return Gf.Quatf(float(q[3]),Gf.Vec3f(*map(float,q[:3])))


def author_tree(model,meta,q,position_iterations=32,velocity_iterations=8,collision_system='PCM'):
    context=omni.usd.get_context();context.new_stage();stage=context.get_stage()
    UsdGeom.SetStageMetersPerUnit(stage,1.);UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z)
    scene=UsdPhysics.Scene.Define(stage,'/World/Physics');scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0,0,-1));scene.CreateGravityMagnitudeAttr().Set(9.80665)
    physics=PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim());physics.CreateEnableGPUDynamicsAttr().Set(False);physics.CreateBroadphaseTypeAttr().Set('MBP');physics.CreateSolverTypeAttr().Set('TGS')
    if collision_system not in ['PCM','SAT']:raise ValueError('unknown collision detection system')
    physics.CreateCollisionSystemAttr().Set(collision_system)
    data=model.createData();pin.framesForwardKinematics(model,data,q);paths={};actuators={a['joint']:a for a in meta['actuators']}
    for jid in range(1,model.njoints):
        path=f'/World/Robot/link_{jid}';paths[jid]=path;body=UsdGeom.Xform.Define(stage,path);pose=data.oMi[jid]
        body.AddTranslateOp().Set(Gf.Vec3d(*map(float,pose.translation)));body.AddOrientOp().Set(quat(pose.rotation));prim=body.GetPrim();UsdPhysics.RigidBodyAPI.Apply(prim)
        inertia=model.inertias[jid];diagonal,rotation=np.linalg.eigh(inertia.inertia)
        if np.linalg.det(rotation)<0:rotation[:,0]*=-1
        mass=UsdPhysics.MassAPI.Apply(prim);mass.CreateMassAttr().Set(float(inertia.mass));mass.CreateCenterOfMassAttr().Set(Gf.Vec3f(*map(float,inertia.lever)));mass.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(*map(float,diagonal)));mass.CreatePrincipalAxesAttr().Set(quat(rotation))
        api=PhysxSchema.PhysxRigidBodyAPI.Apply(prim);api.CreateLinearDampingAttr().Set(0.);api.CreateAngularDampingAttr().Set(0.);api.CreateSleepThresholdAttr().Set(0.);api.CreateSolverPositionIterationCountAttr().Set(position_iterations);api.CreateSolverVelocityIterationCountAttr().Set(velocity_iterations)
        if model.joints[jid].nq==7:
            if meta['object'] and jid==meta['object']['joint']:continue
            UsdPhysics.ArticulationRootAPI.Apply(prim);art=PhysxSchema.PhysxArticulationAPI.Apply(prim);art.CreateEnabledSelfCollisionsAttr().Set(True);art.CreateSolverPositionIterationCountAttr().Set(position_iterations);art.CreateSolverVelocityIterationCountAttr().Set(velocity_iterations);art.CreateSleepThresholdAttr().Set(0.);continue
        a=actuators[jid];A=model.jointPlacements[jid]
        axis=pin.computeJointJacobian(model,model.createData(),pin.neutral(model),jid)[3:,a['v']]
        x=np.array([1.,0.,0.]);v=np.cross(x,axis);xyzw=np.r_[v,1+x@axis]
        if np.linalg.norm(xyzw)<1e-8:xyzw=np.array([0.,0.,1.,0.])
        xyzw/=np.linalg.norm(xyzw);B=pin.Quaternion(xyzw).matrix()
        joint=UsdPhysics.RevoluteJoint.Define(stage,f'/World/Robot/joint_{jid}');joint.CreateAxisAttr().Set('X');joint.CreateBody0Rel().SetTargets([paths[model.parents[jid]]]);joint.CreateBody1Rel().SetTargets([path])
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*map(float,A.translation)));joint.CreateLocalRot0Attr().Set(quat(A.rotation@B));joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0));joint.CreateLocalRot1Attr().Set(quat(B));joint.CreateLowerLimitAttr().Set(float(np.rad2deg(a['lower'])));joint.CreateUpperLimitAttr().Set(float(np.rad2deg(a['upper'])))
        drive=UsdPhysics.DriveAPI.Apply(joint.GetPrim(),'angular');drive.CreateTypeAttr().Set('force');drive.CreateStiffnessAttr().Set(a['kp']*np.pi/180);drive.CreateDampingAttr().Set(a['kd']*np.pi/180);drive.CreateMaxForceAttr().Set(a['effort']);drive.CreateTargetPositionAttr().Set(float(np.rad2deg(q[a['q']])))
        PhysxSchema.PhysxJointAPI.Apply(joint.GetPrim()).CreateArmatureAttr().Set(float(model.armature[a['v']]))
    return stage,paths


class TreeSimulation:
    def __init__(self,model,meta,q,stage,paths,velocity=None,physics_threads=1):
        self.model,self.meta,self.stage,self.paths=model,meta,stage,paths
        get_physx_interface().set_thread_count(physics_threads)
        carb.settings.get_settings().set_bool("/physics/updateToUsd",False)
        carb.settings.get_settings().set_bool("/physics/updateVelocitiesToUsd",False)
        self.interface=get_physx_simulation_interface();self.stage_id=UsdUtils.StageCache.Get().GetId(stage).ToLongInt();assert self.interface.attach_stage(self.stage_id)
        get_physx_interface().force_load_physics_from_usd();get_physx_interface().start_simulation()
        self.sim=tensors.create_simulation_view('numpy',self.stage_id);self.views=[];self.time=0.;self.cached_commands=None;self.cached_thrust=None;self.forces=[]
        self.diagnostic_force_origins=None;self.diagnostic_world_wrenches=None
        for base in meta['bases']:
            view=self.sim.create_articulation_view(paths[base['joint']]);assert view.count==1
            jids=[int(path.rsplit('_',1)[-1]) for path in view.dof_paths[0]]
            links=[int(path.rsplit('_',1)[-1]) for path in view.link_paths[0]]
            ai=[next(i for i,a in enumerate(meta['actuators']) if a['joint']==jid) for jid in jids]
            view.set_dof_positions(np.array([[q[model.joints[j].idx_q] for j in jids]],dtype=np.float32),np.array([0],np.uint32))
            np.testing.assert_allclose(view.get_dof_stiffnesses()[0],[meta['actuators'][i]['kp'] for i in ai],rtol=1e-6)
            np.testing.assert_allclose(view.get_dof_dampings()[0],[meta['actuators'][i]['kd'] for i in ai],rtol=1e-6)
            np.testing.assert_allclose(view.get_dof_armatures()[0],[model.armature[model.joints[j].idx_v] for j in jids],rtol=1e-6,atol=1e-9)
            self.views.append((base,view,jids,links,ai))
        self.sim.update_articulations_kinematic()
        self.body_view=self.sim.create_rigid_body_view(paths[meta['object']['joint']]) if meta['object'] else None
        if velocity is not None:
            for base,view,jids,links,ai in self.views:
                R=pin.Quaternion(q[base['q']+3:base['q']+7]).matrix();local=velocity[base['v']:base['v']+6];omega=R@local[3:]
                world=np.r_[R@local[:3]+np.cross(omega,R@model.inertias[base['joint']].lever),omega]
                view.set_root_velocities(world[None].astype(np.float32),np.array([0],np.uint32))
                view.set_dof_velocities(np.array([[velocity[model.joints[j].idx_v] for j in jids]],dtype=np.float32),np.array([0],np.uint32))
            if self.body_view is not None:
                obj=meta['object'];R=pin.Quaternion(q[obj['q']+3:obj['q']+7]).matrix();local=velocity[obj['v']:obj['v']+6];omega=R@local[3:]
                world=np.r_[R@local[:3]+np.cross(omega,R@model.inertias[obj['joint']].lever),omega]
                self.body_view.set_velocities(world[None].astype(np.float32),np.array([0],np.uint32))

    def observation(self):
        q=pin.neutral(self.model);v=np.zeros(self.model.nv)
        for base,view,jids,links,ai in self.views:
            pose=view.get_root_transforms()[0].astype(float);vel=view.get_root_velocities()[0].astype(float);pose[3:]/=np.linalg.norm(pose[3:]);R=pin.Quaternion(pose[3:]).matrix()
            q[base['q']:base['q']+7]=pose
            origin_velocity=vel[:3]-np.cross(vel[3:],R@self.model.inertias[base['joint']].lever)
            v[base['v']:base['v']+6]=np.r_[R.T@origin_velocity,R.T@vel[3:]]
            positions=view.get_dof_positions()[0];velocities=view.get_dof_velocities()[0]
            for k,j in enumerate(jids):q[self.model.joints[j].idx_q]=positions[k];v[self.model.joints[j].idx_v]=velocities[k]
        if self.body_view is not None:
            obj=self.meta['object'];pose=self.body_view.get_transforms()[0].astype(float);vel=self.body_view.get_velocities()[0].astype(float);pose[3:]/=np.linalg.norm(pose[3:]);R=pin.Quaternion(pose[3:]).matrix();q[obj['q']:obj['q']+7]=pose;origin_velocity=vel[:3]-np.cross(vel[3:],R@self.model.inertias[obj['joint']].lever);v[obj['v']:obj['v']+6]=np.r_[R.T@origin_velocity,R.T@vel[3:]]
        return q,v

    def motor_effort(self,before_v,after_v,dt):
        """Motor-current equivalent over the just completed physics step.

        The backend projection excludes the configured armature inertia.
        Qualified only for this model's frictionless joints, away from stops;
        no contact-report value enters this reconstruction.
        """
        if dt<=0 or not np.isfinite(dt):raise ValueError('invalid motor sample interval')
        effort=np.zeros(len(self.meta['actuators']))
        for _,view,_,_,indices in self.views:
            effort[indices]=view.get_dof_projected_joint_forces()[0]
        vi=np.array([a['v'] for a in self.meta['actuators']])
        effort+=self.model.armature[vi]*(np.asarray(after_v)[vi]-np.asarray(before_v)[vi])/dt
        if not np.isfinite(effort).all():raise ValueError('nonfinite motor observation')
        return effort

    def step(self,command,thrust,dt):
        # The drive target is held over a control interval; external rotor
        # forces must be re-applied after every PhysX step.
        if self.cached_commands is None or not np.array_equal(self.cached_commands,command):
            for base,view,jids,links,ai in self.views:
                view.set_dof_position_targets(np.array([np.asarray(command)[ai]],dtype=np.float32),np.array([0],np.uint32))
            self.cached_commands=np.array(command).copy()
        if self.cached_thrust is None or not np.array_equal(self.cached_thrust,thrust):
            self.forces=[]
            for base,view,jids,links,ai in self.views:
                forces=np.zeros((1,len(links),3),np.float32);moments=forces.copy()
                for r,f in zip(self.meta['rotors'],thrust):
                    frame=self.model.frames[r['frame']]
                    if frame.parentJoint not in links:continue
                    k=links.index(frame.parentJoint);force=frame.placement.rotation[:,2]*f
                    forces[0,k]+=force;moments[0,k]+=np.cross(frame.placement.translation,force)
                self.forces.append((forces,moments,np.zeros_like(forces)))
            self.cached_thrust=np.array(thrust).copy()
        if self.diagnostic_force_origins is not None:
            from .physics_diagnostics import world_wrench
            self.diagnostic_world_wrenches=np.array([world_wrench(view.get_link_transforms()[0],forces[0],moments[0],origin)
                for (_,view,_,_,_),(forces,moments,_),origin in zip(self.views,self.forces,self.diagnostic_force_origins)])
        for (_,view,_,_,_), (forces,moments,positions) in zip(self.views,self.forces):
            view.apply_forces_and_torques_at_position(forces,moments,positions,np.array([0],np.uint32),False)
        self.interface.simulate(dt,self.time);self.interface.fetch_results();self.time+=dt

    def close(self):
        self.interface.detach_stage()
