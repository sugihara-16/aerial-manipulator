"""Bounded cooperative control using observed motion and task contact geometry.

Contact forces here are allocation targets, not measured or predicted forces.
The physical plant and measured command delay remain independent of this model.
"""
from amsrr_det.task_geometry import GraspFrame,box_face_mask
import time
import copy
import numpy as np
import mujoco
from amsrr_det.bounded_qp import bounded_qp
from build.generated_grasp_stage import align_reference_to_object
from build.momentum_feedback import rigid_motion_map

def skew(vector):
    x, y, z = vector
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


class DesiredContactLoad:
    """Distribute a desired object wrench over this task's contact geometry.

    These are feedforward targets. Neither solving this problem nor its
    residual establishes that the physical robot has realized those forces.
    """

    def __init__(self, plant, reference, links, friction=2.25, object_shape=None,material_friction_basis=False,object_face_axis=None):
        from build.sticking_design import LocalGrasp
        self.material_friction_basis=material_friction_basis
        if object_face_axis is not None and (object_shape is None or object_shape.family not in ('box','elongated_box') or object_face_axis not in (0,1,2)):
            raise ValueError('contact face restriction needs a box axis')
        self.object_face_axis=object_face_axis
        self.planar = object_shape is not None and object_shape.family in (
            'box', 'elongated_box', 'triangular_prism', 'tapered_box')
        local = LocalGrasp(plant, reference, links,
                           object_shape=None if self.planar else object_shape)
        if object_face_axis is not None:
            distances=local.material_values()[:,0]
            local.elements=[e for e,gap in zip(local.elements,distances)
                if box_face_mask(e[4],object_shape.size,object_face_axis,e[5][0],gap)]
        self.shape = object_shape
        if not local.elements:
            raise ValueError('task reference has no contact material points')
        # A pad's pressure distribution needs its footprint extremes, not
        # every duplicate point produced by the collision mesh. Keep actual
        # material points (an inner approximation, never an enlarged patch).
        self.elements = []
        for pad in range(len(links)):
            candidates = [e for e in local.elements if e[0] == pad]
            if len(candidates) > 4:
                positions = np.array([e[3] for e in candidates])
                _, _, axes = np.linalg.svd(positions-positions.mean(axis=0), full_matrices=False)
                uv = positions@axes[:2].T
                ids = sorted(set(int(np.argmax(uv@direction)) for direction in
                                 [[1., 1.], [1., -1.], [-1., 1.], [-1., -1.]]))
                candidates = [candidates[i] for i in ids]
            self.elements.extend(candidates)
        self.links = links
        if material_friction_basis:
            from amsrr_det.equilibrium import contact_basis
            self.elements=[(*e[:-1],contact_basis(dict(normal=e[-1][0])).T) for e in self.elements]
        self.m = plant.mj
        self.object_body = self.m.body('object').id
        self.mass_body = int(self.m.jnt_bodyid[self.m.joint('object').id])
        self.mass = float(self.m.body_mass[self.mass_body])
        self.n = len(self.elements)
        self.pad = np.array([e[0] for e in self.elements])
        self.count = np.bincount(self.pad, minlength=len(links))
        angles=np.arange(8)*np.pi/4+(np.pi/8 if material_friction_basis else 0.)
        directions = np.c_[np.cos(angles), np.sin(angles)]
        constraints = np.zeros((9*self.n, 3*self.n))
        for k in range(self.n):
            constraints[9*k, 3*k] = 1.
            constraints[9*k+1:9*k+9, 3*k] = friction*np.cos(np.pi/8)
            constraints[9*k+1:9*k+9, 3*k+1:3*k+3] = -directions
        self.constraints = constraints.T.copy()
        self.bounds = np.zeros(9*self.n)
        self.nominal_normal = max(1.5, 1.5*self.mass*9.80665/(friction*len(links)))
        self.previous = np.zeros(3*self.n)
        self.previous_material_force = None
        self.jac = np.zeros((3, self.m.nv))
        self.last = {}

    def prepare(self, drive, wrench, squeeze=1., active_only=False):
        d, m = drive.d, self.m
        rotation = d.xmat[self.object_body].reshape(3, 3)
        points = np.array([d.xpos[e[1]]+d.xmat[e[1]].reshape(3, 3)@e[3] for e in self.elements])
        if self.shape is not None:
            # Re-project observed pad material points onto known geometry.
            # A fixed plane is not a signed distance to a curved object.
            local = (points-d.xpos[self.object_body])@rotation
            surfaces, gaps, outward = self.shape.project_surface(local)
            if self.planar:
                # At an edge the closest-point vector can point sideways
                # although this pad presses on the top face. Keep the
                # generated supporting face normal; use true distance only
                # for availability, not to silently change the contact role.
                outward = -np.array([e[5][0] for e in self.elements])
        else:
            surfaces = np.array([e[4] for e in self.elements])
            outward = -np.array([e[5][0] for e in self.elements])
            gaps = np.einsum('ij,ij->i', -outward,
                            surfaces-(points-d.xpos[self.object_body])@rotation)
        G = np.zeros((6, 3*self.n))
        reaction = np.zeros((m.nv, 3*self.n))
        nominal = np.zeros(3*self.n)
        for k, e in enumerate(self.elements):
            normal = -outward[k]
            if self.material_friction_basis:
                from amsrr_det.equilibrium import contact_basis
                tangent = contact_basis(dict(normal=normal))[:, 1]
            else:
                tangent = e[5][1]-normal*(normal@e[5][1])
            if np.linalg.norm(tangent) < 1e-8:
                axis = np.eye(3)[np.argmin(abs(normal))]
                tangent = axis-normal*(normal@axis)
            tangent /= np.linalg.norm(tangent)
            axes = rotation@np.array([normal, tangent, np.cross(normal, tangent)]).T
            G[:3, 3*k:3*k+3] = axes
            G[3:, 3*k:3*k+3] = skew(rotation@surfaces[k])@axes
            mujoco.mj_jac(m, d, self.jac, None, points[k], int(e[1]))
            reaction[:, 3*k:3*k+3] = self.jac.T@axes
            nominal[3*k] = squeeze*self.nominal_normal/self.count[e[0]]
        radius = max(.05, max(np.linalg.norm(e[4]) for e in self.elements))
        scale = np.r_[np.ones(3), np.full(3, 1/radius)]
        active = np.ones(self.n, dtype=bool) if not active_only else np.asarray(gaps) <= .0001
        cols = np.flatnonzero(np.repeat(active, 3))
        constraints = np.flatnonzero(np.repeat(active, 9))
        self.current = dict(G=G, reaction=reaction, nominal=nominal, scale=scale,
                            active=active, cols=cols, constraints=constraints, gaps=np.asarray(gaps),
                            object_rotation=rotation.copy())
        return self.current

    def force_incumbent(self):
        """Re-express the numerical seed when contact coordinates change.

        This transports a warm start, not an observed contact force. Slot
        correspondence is only approximate when the collision points change.
        """
        if not self.material_friction_basis or self.previous_material_force is None:
            return self.previous
        c=self.current
        axes=(c['object_rotation'].T@c['G'][:3]).reshape(3,self.n,3).transpose(1,0,2)
        return np.einsum('nji,nj->ni',axes,self.previous_material_force).ravel()

    def record(self, force, wrench, fallback=False):
        c = self.current
        self.previous = force.copy()
        if self.material_friction_basis:
            axes=(c['object_rotation'].T@c['G'][:3]).reshape(3,self.n,3).transpose(1,0,2)
            self.previous_material_force=np.einsum('nij,nj->ni',axes,force.reshape(self.n,3))
        self.last = dict(wrench_error=(c['G']@force-wrench).tolist(),
                         normal_targets=np.bincount(self.pad, force[::3], minlength=len(self.links)).tolist(),
                         numerical_fallback=fallback, active_points=int(c['active'].sum()),
                         pad_gaps=[float(np.min(c['gaps'][self.pad == p])) for p in range(len(self.links))])



class RobotDrive:
    """Map desired robot motion and contact load to physical actuator commands."""

    def __init__(self, plant, period=.01):
        self.plant, self.m, self.h = plant, plant.mj, float(period)
        self.d = mujoco.MjData(self.m)
        self.actuators = plant.meta['actuators']
        self.nj = len(self.actuators)
        ids = [self.m.joint(a['name']).id for a in self.actuators]
        self.qi = self.m.jnt_qposadr[ids]
        self.vi = self.m.jnt_dofadr[ids]
        self.kp = np.array([a['kp'] for a in self.actuators])
        self.kd = np.array([a['kd'] for a in self.actuators])
        self.gain = self.m.actuator_gainprm[:, 0].copy()
        self.dock = np.array([a['role'] == 'dock' for a in self.actuators])
        self.limit = np.r_[[a['effort'] for a in self.actuators],
                          [r['maximum'] for r in plant.meta['rotors']]]
        self.sites = [self.m.site(plant.pin.frames[r['frame']].name).id
                      for r in plant.meta['rotors']]
        self.oi = int(self.m.jnt_qposadr[self.m.joint('object').id])
        self.ov = int(self.m.jnt_dofadr[self.m.joint('object').id])
        self.robot = np.setdiff1d(np.arange(self.m.nv), np.arange(self.ov, self.ov+6))
        self.mass = np.empty((self.m.nv, self.m.nv))
        self.B = np.zeros((self.m.nv, self.m.nu))
        self.B[self.vi, np.arange(self.nj)] = 1.
        self.jac = np.zeros((3, self.m.nv))
        self.constraints = np.c_[np.eye(self.m.nu), -np.eye(self.m.nu)]
        self.previous = np.zeros(self.m.nu)
        self.last = {}

    def observe(self, q, v):
        m, d = self.m, self.d
        d.qpos[:] = q
        d.qvel[:] = v
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        mujoco.mj_crb(m, d)
        mujoco.mj_comVel(m, d)
        mujoco.mj_rne(m, d, 0, d.qfrc_bias)
        mujoco.mj_fullM(m, d, self.mass)
        for k, site in enumerate(self.sites):
            mujoco.mj_jacSite(m, d, self.jac, None, site)
            self.B[:, self.nj+k] = self.jac.T @ d.site_xmat[site].reshape(3, 3)[:, 2]



class CurrentContactLoad(DesiredContactLoad):
    def __init__(self, plant, reference, links, friction=2.25, object_shape=None,material_friction_basis=False,object_face_axis=None):
        super().__init__(plant, reference, links, friction, object_shape,material_friction_basis,object_face_axis)
        self.reference_elements = list(self.elements)
        self.object_geom = self.m.geom('object').id
        self.geom_pad = {}
        for g in range(self.m.ngeom):
            name = self.m.geom(g).name or ''
            for i, link in enumerate(links):
                if name.startswith(link+'/'):
                    self.geom_pad[g] = i
        # Only task-pad/object pairs are relevant to the controller query.
        # The independent physical plant retains every authored collision.
        self.geometry_model = copy.copy(self.m)
        gm = self.geometry_model
        if gm.npair:
            raise ValueError('explicit collision pairs need a filtered query model')
        gm.geom_contype[:] = gm.geom_conaffinity[:] = 0
        gm.body_contype[:] = gm.body_conaffinity[:] = 0
        for geom in self.geom_pad:
            gm.geom_contype[geom], gm.geom_conaffinity[geom] = 1, 2
            body = gm.geom_bodyid[geom]
            gm.body_contype[body], gm.body_conaffinity[body] = 1, 2
        gm.geom_contype[self.object_geom], gm.geom_conaffinity[self.object_geom] = 2, 1
        body = gm.geom_bodyid[self.object_geom]
        gm.body_contype[body], gm.body_conaffinity[body] = 2, 1
        self.geometry = mujoco.MjData(gm)
        self.elements = []
        for pad in range(len(links)):
            values = [e for e in self.reference_elements if e[0] == pad]
            if not values:
                raise ValueError('generated contact role has no material point')
            self.elements.extend(values[k % len(values)] for k in range(4))
        self.reference_elements = list(self.elements)
        self.n = len(self.elements)
        self.pad = np.repeat(np.arange(len(links)), 4)
        self.count = np.full(len(links), 4)
        angles=np.arange(8)*np.pi/4+(np.pi/8 if material_friction_basis else 0.)
        directions = np.c_[np.cos(angles), np.sin(angles)]
        constraints = np.zeros((9*self.n, 3*self.n))
        for k in range(self.n):
            constraints[9*k, 3*k] = 1.
            constraints[9*k+1:9*k+9, 3*k] = friction*np.cos(np.pi/8)
            constraints[9*k+1:9*k+9, 3*k+1:3*k+3] = -directions
        self.constraints = constraints.T.copy()
        self.bounds = np.zeros(9*self.n)
        self.previous = np.zeros(3*self.n)

    def prepare(self, drive, wrench, squeeze=1., active_only=False):
        m, d = self.m, self.geometry
        d.qpos[:] = drive.d.qpos
        mujoco.mj_kinematics(self.geometry_model, d)
        mujoco.mj_collision(self.geometry_model, d)
        groups = [[] for _ in self.links]
        distances = [[] for _ in self.links]
        for contact in d.contact:
            ga, gb = map(int, contact.geom)
            if self.object_geom not in (ga, gb) or contact.dist > .0001:
                continue
            other = gb if ga == self.object_geom else ga
            pad = self.geom_pad.get(other)
            if pad is None:
                continue
            bp, bo = int(m.geom_bodyid[other]), self.object_body
            frame = contact.frame.reshape(3, 3).copy()*(1 if gb == self.object_geom else -1)
            normal = frame[0]
            pp = contact.pos-.5*contact.dist*normal
            po = contact.pos+.5*contact.dist*normal
            Rp, Ro = d.xmat[bp].reshape(3, 3), d.xmat[bo].reshape(3, 3)
            local_frame=frame@Ro
            object_point=Ro.T@(po-d.xpos[bo])
            if self.object_face_axis is not None and not box_face_mask(object_point,self.shape.size,self.object_face_axis,local_frame[0],contact.dist):
                continue
            if self.material_friction_basis:
                from amsrr_det.equilibrium import contact_basis
                local_frame=contact_basis(dict(normal=local_frame[0])).T
            groups[pad].append((pad, bp, bo, Rp.T@(pp-d.xpos[bp]), object_point, local_frame))
            distances[pad].append(float(contact.dist))
        self.elements = list(self.reference_elements)
        active = np.zeros(self.n, bool)
        actual_gaps = np.zeros(self.n)
        for pad, candidates in enumerate(groups):
            if not candidates:
                continue
            ids = list(range(len(candidates)))
            if len(candidates) > 4:
                points = np.array([e[3] for e in candidates])
                _, _, axes = np.linalg.svd(points-points.mean(axis=0), full_matrices=False)
                uv = points@axes[:2].T
                ids = sorted(set(int(np.argmax(uv@a)) for a in [[1,1],[1,-1],[-1,1],[-1,-1]]))
            for k in range(4):
                j = ids[k % len(ids)]
                self.elements[4*pad+k] = candidates[j]
                active[4*pad+k] = True
                actual_gaps[4*pad+k] = distances[pad][j]
        c = super().prepare(drive, wrench, squeeze, active_only)
        # The two surface material points differ under soft penetration.
        # Apply action and reaction at their shared contact point, otherwise
        # tangential force invents a small net internal moment.
        for k in np.flatnonzero(active):
            _, bp, bo, pp, po, frame = self.elements[k]
            Rp = d.xmat[bp].reshape(3, 3)
            Ro = d.xmat[bo].reshape(3, 3)
            point = .5*(d.xpos[bp]+Rp@pp+d.xpos[bo]+Ro@po)
            axes = Ro@frame.T
            block = slice(3*k, 3*k+3)
            c['G'][:3, block] = axes
            c['G'][3:, block] = skew(point-d.xpos[bo])@axes
            mujoco.mj_jac(m, drive.d, self.jac, None, point, int(bp))
            c['reaction'][:, block] = self.jac.T@axes
        if active_only:
            c['active'] = active
            c['cols'] = np.flatnonzero(np.repeat(active, 3))
            c['constraints'] = np.flatnonzero(np.repeat(active, 9))
            c['gaps'][active] = actual_gaps[active]
        return c

class ObjectRelativeControl:
    def __init__(self, plant, contact_reference, links, period=.01, object_shape=None, task_projection=None, vector_directions=False,material_friction_basis=False,free_flight_integral=False,object_face_axis=None):
        self.drive=RobotDrive(plant,period)
        self.load=CurrentContactLoad(plant,contact_reference,links,object_shape=object_shape,material_friction_basis=material_friction_basis,object_face_axis=object_face_axis)
        self.m=plant.mj;self.h=period;self.relative_feedback=True;self.last={}
        initial_rotation=np.empty((3,3));mujoco.mju_quat2Mat(initial_rotation.ravel(),contact_reference[self.drive.oi+3:self.drive.oi+7]);self.grasp_frame=GraspFrame(object_shape,initial_rotation)
        self.delta=np.zeros(self.m.nv);self.angle=np.zeros(3);self.integral=np.zeros(6);self.last_command=None
        self.free_flight_integral=free_flight_integral;self.flight_error_integral=np.zeros(6)
        self.flight_observation_time=None
        self.flight_integral_limit=np.array([.25,.25,.25,.15,.15,.15])
        self.flight_integral_gain=np.array([4.,4.,4.,8.,8.,8.])
        self.task_projection=None if task_projection is None else np.asarray(task_projection,dtype=float)
        self.direction=None
        if vector_directions:
            from amsrr_det.rotor_direction import RotorDirection
            self.direction=RotorDirection(self.drive)
        self.reference_geometry=mujoco.MjData(self.load.geometry_model)
        self.reference_groups=[tuple(g for g in self.load.geom_pad if self.load.geom_pad[g]==i) for i in range(len(links))]
        if self.task_projection is not None and (self.task_projection.ndim!=2 or self.task_projection.shape[1]!=6):
            raise ValueError('task projection maps six-dimensional wrenches to permitted motions')

    def update(self,q,v,shape_reference,object_reference,*,object_velocity=None,object_acceleration=None,shape_velocity=None,squeeze=1.,payload_fraction=1.,tracking_fraction=None,damping_scale=1.,active_only=True,deadline=None,track_object=False,wrench_bias=None,shape_contact_depth=None,motion_derivative=None,motion_curvature=None,observation_time=None):
        if deadline is None:
            deadline=time.perf_counter()+.008
        tracking_fraction=payload_fraction if tracking_fraction is None else tracking_fraction
        dr,load,m=self.drive,self.load,self.m;dr.observe(q,v);d=dr.d;oi,ov=dr.oi,dr.ov
        if shape_contact_depth is not None:
            # Correct interpolation error in a declared contact path using
            # surface geometry only. No physical force or future dynamics.
            from build.contact_step_control import pad_gap_jacobians
            shape_reference=shape_reference.copy();gm=load.geometry_model;gd=self.reference_geometry
            indices=np.r_[np.arange(3),dr.vi[dr.dock]];weights=np.r_[np.ones(3),np.full(np.count_nonzero(dr.dock),.2)]
            for _ in range(2):
                gd.qpos[:]=shape_reference;mujoco.mj_fwdPosition(gm,gd)
                gaps,J=pad_gap_jacobians(gm,gd,load.links,self.reference_groups)
                step=np.linalg.lstsq(J[:,indices]*weights,gaps+shape_contact_depth,rcond=1e-8)[0]
                delta=np.zeros(m.nv);delta[indices]=-weights*step
                delta*=min(1.,.003/max(np.max(abs(delta[:3])),1e-9),.03/max(np.max(abs(delta[dr.vi])),1e-9))
                mujoco.mj_integratePos(m,shape_reference,delta,1.)
        R=d.xmat[load.object_body].reshape(3,3);twist=np.r_[v[ov:ov+3],R@v[ov+3:ov+6]]
        tv=np.zeros(6) if object_velocity is None else object_velocity
        acc=np.zeros(6) if object_acceleration is None else object_acceleration.copy()
        mujoco.mju_subQuat(self.angle,object_reference[3:],q[oi+3:oi+7])
        objacc=acc+np.r_[4*(object_reference[:3]-q[oi:oi+3]),16*(R@self.angle)]+damping_scale*np.r_[[4.]*3,[8.]*3]*(tv-twist)
        error=np.r_[object_reference[:3]-q[oi:oi+3],R@self.angle]
        self.integral=np.clip(self.integral+self.h*tracking_fraction*error,-np.r_[[1.]*3,[.25]*3],np.r_[[1.]*3,[.25]*3])
        if tracking_fraction==0.:self.integral*=np.exp(-self.h/.25)
        objacc+=tracking_fraction*np.r_[[2.]*3,[8.]*3]*self.integral
        Ri=d.ximat[load.mass_body].reshape(3,3);Io=Ri@np.diag(m.body_inertia[load.mass_body])@Ri.T
        wrench=np.r_[load.mass*(tracking_fraction*objacc[:3]-payload_fraction*m.opt.gravity),tracking_fraction*(Io@objacc[3:]+np.cross(twist[3:],Io@twist[3:]))]
        if wrench_bias is not None:wrench+=wrench_bias
        c=load.prepare(dr,wrench,squeeze,active_only)
        if payload_fraction == 0. and squeeze == 0. and tracking_fraction == 0.:
            # Opening is free flight: do not allocate support to the object.
            c['cols'] = np.empty(0, dtype=int)
            c['constraints'] = np.empty(0, dtype=int)
        grasp_rotation=R;grasp_twist=twist;grasp_acc=objacc;grasp_observation=q
        symmetry=track_object and self.grasp_frame.active
        if symmetry:
            grasp_rotation,wg,ag=self.grasp_frame.update(R,np.ascontiguousarray(twist[3:]),np.ascontiguousarray(objacc[3:]))
            grasp_twist=twist.copy();grasp_twist[3:]=wg
            grasp_acc=objacc.copy();grasp_acc[3:]=ag
            grasp_observation=q.copy();mujoco.mju_mat2Quat(grasp_observation[oi+3:oi+7],grasp_rotation.ravel())
        aligned=align_reference_to_object(m,shape_reference,grasp_observation)
        mujoco.mj_differentiatePos(m,self.delta,1.,q,aligned)
        T=rigid_motion_map(m,q)
        shape_rate=np.zeros(m.nv) if shape_velocity is None else shape_velocity.copy()
        Rref=np.empty((3,3));mujoco.mju_quat2Mat(Rref.ravel(),shape_reference[oi+3:oi+7])
        turn=grasp_rotation@Rref.T
        shape_rate[:3]=turn@shape_rate[:3]
        Rd=np.empty((3,3));Ra=np.empty((3,3))
        mujoco.mju_quat2Mat(Rd.ravel(),aligned[3:7]);mujoco.mju_quat2Mat(Ra.ravel(),q[3:7])
        shape_rate[3:6]=Ra.T@Rd@shape_rate[3:6]
        reference_velocity=tracking_fraction*(T@grasp_twist)+shape_rate
        reference_velocity[:3]+=tracking_fraction*np.cross(grasp_twist[3:],aligned[:3]-q[:3])
        relative=16*self.delta+8*(reference_velocity-v)
        if not self.relative_feedback:relative[:6]=0
        relative[6:]=0
        target_acc=T@grasp_acc+relative
        # Capture uses relative position but damps common world motion.
        # Release follows a continuous world-fixed free-flight reference.
        target=shape_reference.copy();target[oi:oi+7]=object_reference
        target=align_reference_to_object(m,shape_reference,target)
        world_delta=np.zeros(m.nv);mujoco.mj_differentiatePos(m,world_delta,1.,q,target)
        target_acc[:6]=(T@(tracking_fraction*grasp_acc)+relative)[:6] if track_object else (T@acc+16*world_delta+8*(T@tv+shape_rate-v))[:6]
        if self.free_flight_integral:
            dt=self.h if observation_time is None or self.flight_observation_time is None else max(0.,observation_time-self.flight_observation_time)
            self.flight_observation_time=observation_time
            free=not track_object and payload_fraction==0. and tracking_fraction==0. and squeeze==0.
            if free:
                # Bounded integral action removes persistent free-flight pose
                # error left by constrained input allocation. Motor bounds
                # still apply to the complete acceleration request below.
                # Store angular error in world coordinates so an intervening
                # base rotation does not rotate the accumulated correction.
                error=world_delta[:6].copy();error[3:]=Ra@error[3:]
                self.flight_error_integral=np.clip(self.flight_error_integral+dt*error,
                    -self.flight_integral_limit,self.flight_integral_limit)
                correction=self.flight_integral_gain*self.flight_error_integral
                correction[3:]=Ra.T@correction[3:]
                target_acc[:6]+=correction
            else:self.flight_error_integral.fill(0.)
        if motion_derivative is not None:
            if self.task_projection is None or self.task_projection.shape!=(1,6):
                raise ValueError('path motion currently needs one permitted task coordinate')
            axis=self.task_projection[0];rate=float(axis@twist);acceleration=float(axis@objacc)
            first=np.array(motion_derivative,copy=True);second=np.array(motion_curvature,copy=True)
            first[3:6]=Ra.T@Rd@first[3:6];second[3:6]=Ra.T@Rd@second[3:6]
            # Chain rule along the generated deforming path. A rigid-motion
            # map would incorrectly rotate the base when Dock joints supply
            # part of the requested bar rotation.
            target_acc=tracking_fraction*(first*acceleration+second*rate**2)+relative
        elif symmetry or (track_object and self.task_projection is not None):
            arm=aligned[:3]-q[oi:oi+3]
            target_acc[:3]+=tracking_fraction*(np.cross(grasp_acc[3:],aligned[:3]-q[:3])+np.cross(grasp_twist[3:],np.cross(grasp_twist[3:],arm)))
        force=dr.mass@target_acc+d.qfrc_bias
        jv=np.zeros(dr.nj) if shape_velocity is None else shape_velocity[dr.vi]
        force[dr.vi]+=dr.kp*self.delta[dr.vi]+dr.kd*(jv-v[dr.vi])
        approach=np.zeros(3*load.n)
        approach[::3]=(~c['active'])*np.clip((.02-c['gaps'])/.01,0,1)*squeeze*load.nominal_normal/load.count[load.pad]
        force+=c['reaction']@approach
        cols,ci=c['cols'],c['constraints'];nc=len(cols);nu=m.nu;rows=dr.robot
        scale=np.zeros(m.nv);scale[:6]=4/np.maximum(np.diag(dr.mass)[:6],.01);scale[dr.vi]=1/dr.limit[:dr.nj]
        B=dr.B
        current_force=force.copy()
        if self.direction is not None:
            thrust=dr.previous[dr.nj:]*dr.limit[dr.nj:]
            if self.last_command is not None:thrust=self.last_command[dr.nj:]
            B,drift=self.direction.linearize(q,v,thrust)
            force-=drift
            scale[dr.vi[~dr.dock]]*=.03
        A=np.c_[B[rows]*dr.limit,-c['reaction'][np.ix_(rows,cols)]]*scale[rows,None]
        b=force[rows]*scale[rows]
        if self.direction is not None:
            # Present support uses observed rotor angles. Anticipated direction
            # action above is a separate frozen-geometry target over the local
            # servo horizon, never force already available at this instant.
            current=np.c_[dr.B[:6]*dr.limit,-c['reaction'][np.ix_(np.arange(6),cols)]]*scale[:6,None]
            A=np.vstack([A,current]);b=np.r_[b,current_force[:6]*scale[:6]]
        oscale=np.r_[np.full(3,4/max(load.mass,.01)),4/np.maximum(np.diag(Io),.01)]
        task_G=c['G'][:,cols];task_wrench=wrench
        if self.task_projection is not None:
            projection=self.task_projection
            task_G=projection@task_G;task_wrench=projection@wrench
            inertia=np.zeros((6,6));inertia[:3,:3]=load.mass*np.eye(3);inertia[3:,3:]=Io
            oscale=4/np.maximum(np.diag(projection@inertia@projection.T),.01)
        A=np.vstack([A,np.c_[np.zeros((len(task_wrench),nu)),task_G]*oscale[:,None]])
        b=np.r_[b,task_wrench*oscale]
        lower=-np.ones(nu);lower[dr.nj:]=0;upper=np.ones(nu)
        gi=np.flatnonzero(~dr.dock);ranges=m.jnt_range[[m.joint(dr.actuators[i]['name']).id for i in gi]]
        offset=dr.kp[gi]*q[dr.qi[gi]]+dr.kd[gi]*v[dr.vi[gi]]
        lower[gi]=np.maximum(lower[gi],(dr.kp[gi]*ranges[:,0]-offset)/dr.limit[gi]);upper[gi]=np.minimum(upper[gi],(dr.kp[gi]*ranges[:,1]-offset)/dr.limit[gi])
        bad=lower>upper;lower[bad]=upper[bad]=np.clip((lower[bad]+upper[bad])/2,-1,1)
        constraints=np.zeros((nu+nc,2*nu+len(ci)))
        constraints[:nu,:2*nu]=dr.constraints
        constraints[nu:,2*nu:]=load.constraints[np.ix_(cols,ci)]
        regularization=np.r_[np.full(nu,1e-4),np.full(nc,.01)]
        nominal=np.r_[dr.previous,c['nominal'][cols]]
        held=dr.previous.copy()
        if self.last_command is not None:
            effort=self.last_command*dr.gain
            effort[:dr.nj]-=dr.kp*q[dr.qi]+dr.kd*v[dr.vi]
            held=effort/dr.limit
        incumbent=np.r_[held,load.force_incumbent()[cols]]
        if nc and self.task_projection is None:
            G=c['G'][:,cols]*c['scale'][:,None]
            _,sigma,Vt=np.linalg.svd(G,full_matrices=False)
            rank=int(np.count_nonzero(sigma>max(sigma[0],1e-20)*1e-10))
            P=np.eye(nc)-Vt[:rank].T@Vt[:rank]
            A=np.vstack([A,np.c_[np.zeros((nc,nu)),P]])
            b=np.r_[b,P@c['nominal'][cols]]
        # With environmental support, forces outside the object's permitted
        # motion can unload the robot's joints. They are not internal squeeze
        # and must not be assigned a strong zero-force target.
        x,fallback,iterations=bounded_qp(A.T@A+np.diag(regularization),A.T@b+regularization*nominal,constraints,np.r_[lower,-upper,load.bounds[ci]],lower,upper,incumbent,deadline=deadline)
        x[:nu]=np.clip(x[:nu],lower,upper);dr.previous=x[:nu].copy()
        f=np.zeros(3*load.n);f[cols]=x[nu:];load.record(f,wrench,fallback)
        u=x[:nu]*dr.limit;u[:dr.nj]=(u[:dr.nj]+dr.kp*q[dr.qi]+dr.kd*v[dr.vi])/dr.gain[:dr.nj];u[gi]=np.clip(u[gi],ranges[:,0],ranges[:,1])
        self.last=dict(object_position_error_m=float(np.linalg.norm(object_reference[:3]-q[oi:oi+3])),relative_position_error_m=float(np.linalg.norm(self.delta[:3])),relative_angle_error_rad=float(np.linalg.norm(self.delta[3:6])),joint_error_rad=float(np.max(abs(self.delta[dr.vi]))),object_twist=twist.tolist(),requested_object_wrench=wrench.tolist(),**load.last,allocation=dict(velocity_error=float(np.linalg.norm(A@x-b)),saturated_inputs=int(np.count_nonzero(abs(x[:nu])>.999)),numerical_fallback=fallback,qp_iterations=iterations))
        self.last_command=u.copy()
        if self.free_flight_integral:self.last['flight_error_integral_world']=self.flight_error_integral.tolist()
        return u
