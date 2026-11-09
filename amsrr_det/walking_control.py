"""Ground support extension of the common bounded actuator allocation."""
from .walking_response import response as held_body_response, minimum_support_qp
import time
import numpy as np
import mujoco
from build.relative_task_control import RobotDrive
from amsrr_det.bounded_qp import bounded_qp
from amsrr_det.walking import foot_material,ground_support,requires_face_alignment


def energy_regularizer(previous,limits,joint_count,weight):
    """Quadratic local cost for the planner's dimensionless effort proxy.

    A QP uses half x' H x. Reweighting its rotor diagonal at the last command
    approximates sum(T**1.5)/weight**1.5; joint cost is .05*mean((tau/limit)**2).
    The 2% floor bounds curvature near zero. This is one local allocation,
    not a claim of converged power optimization or calibrated electrical cost.
    """
    nj=joint_count
    rotor=1.5*(limits[nj:]/weight)**1.5/np.sqrt(np.maximum(previous[nj:],.02))
    return 30000.0*np.r_[np.full(nj,0.1/nj),rotor]


class GroundControl:
    def __init__(self,plant,reference,links,*,foot_points=None):
        self.drive=RobotDrive(plant);self.m=plant.mj;self.reference=reference.copy()
        self.material=foot_material(plant,links);self.links=links
        self.face_alignment=np.array([requires_face_alignment(link) for link in links])
        self.foot_points=[];self.last={};self.last_command=None
        dr=self.drive;dr.observe(reference,np.zeros(self.m.nv))
        if foot_points is None:
            for bid,vertices in self.material:
                world=vertices@dr.d.xmat[bid].reshape(3,3).T+dr.d.xpos[bid]
                selected=vertices[world[:,2]<=world[:,2].min()+.0005]
                self.foot_points.append(selected.mean(axis=0))
        else:
            # Plan and execution must follow the same material point in x/y.
            # Selecting a new lowest vertex after smoothing shifts the target.
            self.foot_points=foot_points
        self.foot_points=np.array(self.foot_points, dtype=float, copy=True)
        if self.foot_points.shape!=(len(links),3) or not np.isfinite(self.foot_points).all():
            raise ValueError('invalid foot tracking points')
        self.contact_bias=np.zeros(len(links));self.last_observation=None
        self.lift_bias=np.zeros(len(links));self.path_bias=np.zeros(2)
        self.attitude_bias=np.zeros(3)
        self.joint_bias=np.zeros(self.drive.nj)
        self.attitude_weight=max(20.,2*float(plant.meta['modules']))
        self.jp=np.empty((3,self.m.nv));self.jr=self.jp.copy();self.jd=self.jp.copy()
        self.delta=np.zeros(self.m.nv);self.previous=np.zeros(self.m.nu+12*len(links))
        self.reference_data=mujoco.MjData(self.m)
        from amsrr_det.rotor_direction import RotorDirection
        self.direction=RotorDirection(dr)
        from amsrr_det.grasp_geometry import part_bounds
        bounds=[(self.m.body(f'm{i}/{name}').id,center,half)
                for i in range(plant.meta['modules']) for name,center,half in part_bounds()
                if f'm{i}/{name}' not in links]
        self.floor_bids=np.array([x[0] for x in bounds])
        self.floor_centers=np.array([x[1] for x in bounds]);self.floor_halves=np.array([x[2] for x in bounds])

    def positions(self,data):
        return np.array([data.xpos[b]+data.xmat[b].reshape(3,3)@p
                         for (b,_),p in zip(self.material,self.foot_points)])

    def update(self,q,v,base_reference,base_velocity,base_acceleration,feet,foot_velocity,foot_acceleration,stance,*,observation_time=None,deadline=None,shape_reference=None,shape_velocity=None,shape_acceleration=None,com_reference=None,com_velocity=None,com_acceleration=None,ground_fraction=1.,walk_time=None,flight_mode=False):
        if deadline is None:deadline=time.perf_counter()+.008
        dr=self.drive;m=self.m;dr.observe(q,v);d=dr.d;nu=m.nu;nf=len(self.material)
        dt=.01 if observation_time is None or self.last_observation is None else np.clip(observation_time-self.last_observation,0.,.05)
        ref=(self.reference if shape_reference is None else shape_reference).copy();ref[:7]=base_reference
        mujoco.mj_differentiatePos(m,self.delta,1.,q,ref)
        # Compensate persistent attitude error in the effort/tracking
        # compromise. Height keeps proportional/velocity feedback: integrating
        # it can fight the feet's load-bearing contact configuration.
        self.attitude_bias=np.clip(self.attitude_bias+.5*dt*self.delta[3:6],-.15,.15)
        desired=np.zeros(m.nv);desired[:6]=base_acceleration+25*self.delta[:6]+10*(base_velocity-v[:6])
        desired[3:6]+=25*self.attitude_bias
        sv=np.zeros(m.nv) if shape_velocity is None else shape_velocity
        sa=np.zeros(m.nv) if shape_acceleration is None else shape_acceleration
        # Bounded integral feedback restores lasting Dock shape error. It is
        # frozen in flight; clipping bounds memory but is not anti-windup or
        # a stability guarantee under competing tasks and actuator saturation.
        self.joint_bias=np.clip(self.joint_bias+(0. if flight_mode else .5)*dt*
            self.delta[dr.vi]*dr.dock,-.1,.1)
        desired[dr.vi]=sa[dr.vi]+64.*(self.delta[dr.vi]+self.joint_bias)+16.*(sv[dr.vi]-v[dr.vi])
        rd=self.reference_data;rd.qpos[:]=ref;rd.qvel[:]=sv
        mujoco.mj_kinematics(m,rd);mujoco.mj_comPos(m,rd);mujoco.mj_comVel(m,rd)
        J=[];Jdot=[];angular=[];angular_target=[]
        contact_jacobians=[];contact_targets=[];support_groups=[]
        reaction=np.zeros((m.nv,12*nf));active=[];positions=self.positions(d);gaps=np.empty(nf)
        for i,((bid,vertices),lp) in enumerate(zip(self.material,self.foot_points)):
            point=positions[i]
            mujoco.mj_jac(m,d,self.jp,self.jr,point,bid);J.append(self.jp.copy());angular.append(self.jr.copy())
            mujoco.mj_jacDot(m,d,self.jd,None,point,bid);Jdot.append(self.jd@v)
            R=d.xmat[bid].reshape(3,3);Rref=rd.xmat[bid].reshape(3,3)
            error=.5*np.cross(R.T,Rref.T).sum(axis=0)
            omega=self.jr@v
            mujoco.mj_jac(m,rd,None,self.jr,rd.xpos[bid],bid)
            reference_omega=self.jr@sv
            reference_alpha=self.jr@sa
            mujoco.mj_jacDot(m,rd,None,self.jd,rd.xpos[bid],bid)
            reference_alpha+=self.jd@sv
            if self.face_alignment[i] and (not flight_mode or ground_fraction>0.):
                # Landing still uses flight tracking for the assembly, but
                # its feet need the same face-normal objective as walking
                # once the ground-transfer objectives begin to ramp in.
                normal=R[:,0];tilt=np.eye(3)-np.outer(normal,normal)
                # Follow the generated swing orientation; it reaches the
                # downward contact face before planned touchdown.
                measured_height=float(np.min((vertices@R.T+d.xpos[bid])[:,2]))
                blend=np.clip((measured_height-.003)/.012,0.,1.)
                blend=blend*blend*(3-2*blend)
                contact_normal=(1-blend)*np.array([0.,0.,-1.])+blend*Rref[:,0]
                contact_normal/=np.linalg.norm(contact_normal)
                # A delayed lift must not tilt a foot still touching ground.
                # Only measured geometric clearance releases this target.
                angle_goal=400*np.cross(normal,contact_normal)+tilt@(40*(blend*reference_omega-omega)+blend*reference_alpha)
                yaw=np.eye(3)-tilt
                weight=120*tilt+.5*yaw
                angular[-1]=weight@angular[-1]
                mujoco.mj_jacDot(m,d,None,self.jd,point,bid)
                angular_target.append(weight@(angle_goal+yaw@(64*error+16*(self.jr@sv-omega)+self.jr@sa)-self.jd@v))
            else:
                angular[-1]*=.5
                angular_target.append(.5*(64*error+16*(self.jr@sv-omega)+self.jr@sa))
            world=vertices@d.xmat[bid].reshape(3,3).T+d.xpos[bid]
            lowest=int(np.argmin(world[:,2]));gap=world[lowest,2]
            gaps[i]=gap
            # Vertical clearance is the lowest CAD point, not the centre of
            # a sole that can tilt. Use the same measured material point for
            # height, velocity and acceleration in the tracking objective.
            mujoco.mj_jac(m,d,self.jp,None,world[lowest],bid)
            mujoco.mj_jacDot(m,d,self.jd,None,world[lowest],bid)
            J[-1][2]=self.jp[2];Jdot[-1][2]=self.jd[2]@v;positions[i,2]=gap
            if gap<.0002 and (stance[i] or foot_velocity[i,2]<=0.):
                # Geometry from measured poses, never simulated reaction truth.
                basis,extreme=ground_support(R,world)
                for j,index in enumerate(extreme):
                    if index in extreme[:j]:continue
                    group=[4*i+k for k,z in enumerate(extreme) if z==index];support_groups.append(group);repeat=len(group)
                    slot=4*i+j;support=world[index]
                    mujoco.mj_jac(m,d,self.jp,None,support,bid)
                    force_basis=basis.copy()
                    reaction[:,3*slot:3*slot+3]=self.jp.T@force_basis;active.append(slot)
                    # Same material point for velocity and acceleration.
                    # Sticking is a soft objective under finite friction.
                    contact_jacobians.append(self.jp.copy()*np.sqrt(repeat))
                    mujoco.mj_jacDot(m,d,self.jd,None,support,bid)
                    # Brake slip at the measured material point. This is a
                    # soft feedback objective, not a rolling constraint or
                    # an assumed actual reaction from the physical engine.
                    settle=-self.jd@v-40*self.jp@v
                    settle[2]-=400*support[2]
                    contact_targets.append(settle*np.sqrt(repeat))
        J=np.vstack(J);Jdot=np.concatenate(Jdot)
        cols=np.concatenate([np.arange(3*i,3*i+3) for i in active]) if active else np.empty(0,dtype=int)
        thrust=self.previous[dr.nj:nu]*dr.limit[dr.nj:]
        B=dr.B;drift=np.zeros(m.nv)
        W=np.c_[B*dr.limit,reaction[:,cols]]
        held,hold_bias,_,_=held_body_response(dr.mass,dr.vi,dr.kp,dr.kd,v)
        response=held@np.c_[W,drift-d.qfrc_bias];D=response[:,:-1];bias=response[:,-1]+hold_bias
        # One acceleration objective for both support and swing. The mode
        # changes targets and the available ground reactions, not the solver.
        scale=np.full(m.nv,.12);scale[:3]=40. if flight_mode else 4.;scale[3:6]=self.attitude_weight;scale[dr.vi[~dr.dock]]=0.3
        # Preserve the articulated motion in acceleration space. A small
        # generalized-torque residual alone does not restore joint shape
        # when the root and ground reactions move together.
        scale[dr.vi[dr.dock]]=6.0
        A=D*scale[:,None];target=(desired-bias)*scale
        if com_reference is not None:
            # The route belongs to the whole assembly, not just its root.
            # The free-base translation rows give mass * COM Jacobian;
            # their bias gives the corresponding velocity-dependent term.
            mass=float(m.body_mass.sum());Jcom=dr.mass[:2]/mass
            # Accumulate persistent route error without raising the single
            # allocation's tracking weight. This bounded reference correction
            # is feedback, not a learned or structure-specific offset.
            com_error=com_reference-d.subtree_com[0,:2]
            self.path_bias=np.clip(self.path_bias+.5*dt*com_error,-.05,.05)
            desired_com=com_acceleration+49*(com_error+self.path_bias)+14*(com_velocity-Jcom@v)
            drift_com=d.qfrc_bias[:2]/mass+m.opt.gravity[:2]
            A=np.vstack([A,60.0*Jcom@D]);target=np.r_[target,60.0*(desired_com-Jcom@bias-drift_com)]
        if hasattr(self,"load_path") and observation_time is not None and not flight_mode:
            K=self.load_path(observation_time if walk_time is None else walk_time);A=np.vstack([A,K@D]);target=np.r_[target,K@(desired-bias)]
        if contact_jacobians:
            CJ=np.vstack(contact_jacobians);ct=np.concatenate(contact_targets)
            A=np.vstack([A,30.0*CJ@D]);target=np.r_[target,30.0*(ct-CJ@bias)]
        # Press only until contact is acquired. Once supporting, the contact
        # acceleration objective replaces a conflicting below-floor target.
        # A small pressing reference acquires real contact despite residual
        # pose error. Reactions remain unavailable until measured geometry
        # reaches the floor; this does not invent support before touchdown.
        # Integrate only observed geometric separation during intended stance.
        # Retain the bounded correction across steps; never invent a reaction
        # or use simulated contact force to acquire the next support.
        self.last_observation=observation_time
        self.contact_bias[stance]=np.clip(self.contact_bias[stance]+ground_fraction*20*dt*(gaps[stance]-.00005),0.,.03)
        goal=feet.copy();goal[stance,2]-=np.where(gaps[stance]<.0002, 0., .016+self.contact_bias[stance])
        position_error=goal-positions;velocity_error=foot_velocity-(J@v).reshape(nf,3)
        foot_target=foot_acceleration+100*position_error+20*velocity_error
        # Swing height needs enough restoration to lift the whole CAD foot,
        # with critical damping; the same bounded allocation limits actuation.
        foot_target[:,2]+=1500.*position_error[:,2]+60.*velocity_error[:,2]
        # Compensate a lasting lift deficit during the intended swing, including
        # a foot that has not left the floor yet. Fade before planned landing
        # and reset in stance so an upward correction cannot persist there.
        # Bounds limit accumulated demand; they are not an anti-windup proof.
        envelope=np.clip(feet[:,2]/.02,0.,1.)
        self.lift_bias[stance]=0.
        self.lift_bias[~stance]=np.clip(self.lift_bias[~stance]+(0. if flight_mode else 10.)*dt*
            (feet[~stance,2]-positions[~stance,2])*envelope[~stance],0.,.015)
        foot_target[:,2]+=1600.*self.lift_bias*envelope
        # Brake the real foot before touchdown. The old below-floor reference
        # produced a new impact at each stance transition. Use measured gap
        # and velocity; no virtual floor force is introduced before contact.
        descending=(foot_velocity[:,2]<0) & (gaps<.012) & ~stance
        vz=(J@v).reshape(nf,3)[:,2]
        foot_target[descending,2]=np.maximum(foot_target[descending,2],
            -400*np.maximum(gaps[descending],0.)-40*vz[descending])
        missing=stance & (gaps>=.0002)
        foot_target[missing,2]=400*(-.0005-gaps[missing])-40*vz[missing]
        foot_target=foot_target.ravel()
        A=np.vstack([A,ground_fraction*10*J@D]);target=np.r_[target,ground_fraction*10*(foot_target-Jdot-J@bias)]
        angular=np.vstack(angular);at=np.concatenate(angular_target)
        A=np.vstack([A,ground_fraction*angular@D]);target=np.r_[target,ground_fraction*(at-angular@bias)]
        # Walking support belongs to the chosen feet. Keep nearby batteries
        # and other body parts above the ground through the same allocation,
        # rather than stopping or letting those parts become unintended feet.
        rotations=d.xmat[self.floor_bids].reshape(-1,3,3)
        corners=self.floor_centers-np.sign(rotations[:,2])*self.floor_halves
        lowest=d.xpos[self.floor_bids]+np.einsum('bij,bj->bi',rotations,corners)
        for i in np.flatnonzero(lowest[:,2]<.015):
            bid=int(self.floor_bids[i]);point=lowest[i]
            mujoco.mj_jac(m,d,self.jp,None,point,bid)
            mujoco.mj_jacDot(m,d,self.jd,None,point,bid)
            row=self.jp[2].copy();velocity=row@v
            acceleration=max(0.,100*(.01-point[2])-20*velocity)
            A=np.vstack([A,10*row@D]);target=np.r_[target,10*(acceleration-self.jd[2]@v-row@bias)]
        # Present support must use observed rotor directions. Anticipated
        # gimbal motion may improve future allocation, not current gravity.
        current_scale=4/np.maximum(np.diag(dr.mass)[:6],.01)
        current=np.c_[dr.B[:6]*dr.limit,reaction[np.ix_(np.arange(6),cols)]]
        A=np.vstack([A,current*current_scale[:,None]])
        target=np.r_[target,(dr.mass@desired+d.qfrc_bias)[:6]*current_scale]
        # Preserve the generated articulated shape through task feedback.
        # This stiffness is independent of the local MIT hold gain, so reducing
        # stale-command resistance does not remove commanded shape restoration. Low inertia must not turn
        # a nominal acceleration target into a nearly unweighted soft limb.
        holding=dr.mass@desired+d.qfrc_bias
        holding[dr.vi]+=np.where(dr.dock,30.0,dr.kp)*(self.delta[dr.vi]+self.joint_bias)+np.where(dr.dock,2.,dr.kd)*(sv[dr.vi]-v[dr.vi])
        js=1/dr.limit[:dr.nj];js[~dr.dock]*=.3
        A=np.vstack([A,W[dr.vi]*js[:,None]])
        target=np.r_[target,(holding[dr.vi]-drift[dr.vi])*js]
        lower=-np.ones(nu);lower[dr.nj:]=0;upper=np.ones(nu)
        gi=np.flatnonzero(~dr.dock);ranges=m.jnt_range[[m.joint(dr.actuators[i]['name']).id for i in gi]]
        offset=dr.kp[gi]*q[dr.qi[gi]]+dr.kd[gi]*v[dr.vi[gi]]
        lower[gi]=np.maximum(-1,(dr.kp[gi]*ranges[:,0]-offset)/dr.limit[gi]);upper[gi]=np.minimum(1,(dr.kp[gi]*ranges[:,1]-offset)/dr.limit[gi])
        bad=lower>upper;lower[bad]=upper[bad]=np.clip((lower[bad]+upper[bad])/2,-1,1)
        # Gimbal position servos receive this prescribed command. Allocate
        # other actuators against the same saturated effort, instead of
        # choosing a different gimbal effort and replacing it afterwards.
        target_velocity=np.clip(sv[dr.vi[gi]]+4*self.delta[dr.vi[gi]],-2.2,2.2)
        gimbal_command=np.clip(q[dr.qi[gi]]+(dr.kd[gi]/dr.kp[gi]+.005)*target_velocity,ranges[:,0],ranges[:,1])
        fixed_effort=np.clip((dr.kp[gi]*gimbal_command-offset)/dr.limit[gi],-1.,1.)

        nc=len(cols);C=np.zeros((nu+nc,2*nu+9*len(active)));C[:nu,:2*nu]=dr.constraints
        theta=np.arange(8)*np.pi/4+np.pi/8
        for j in range(len(active)):
            C[nu+3*j,2*nu+9*j]=1.
            C[nu+3*j,2*nu+9*j+1:2*nu+9*j+9]=.8*np.cos(np.pi/8)
            C[nu+3*j+1:nu+3*j+3,2*nu+9*j+1:2*nu+9*j+9]=-np.array([np.cos(theta),np.sin(theta)])
        contact_bounds=np.zeros(9*len(active))
        if active:
            # Ground-load preference uses the allocation's reaction variables,
            # never the simulator's measured reaction. Real support is audited
            # separately; this soft target cannot override motor/friction bounds.
            row=np.zeros(nu+nc);row[nu::3]=100/(self.m.body_mass.sum()*9.80665)
            A=np.vstack([A,row]);target=np.r_[target,100*0.75]
        # Consumption matters in execution as well as candidate selection.
        # Keep a support preference so effort reduction does not unload feet.
        # Tracking/restoration terms and all physical bounds remain active.
        energy=energy_regularizer(self.previous[:nu],dr.limit,dr.nj,self.m.body_mass.sum()*9.80665)
        # Execution prioritizes rotor consumption more strongly than the joint
        # term. The bounded task corrections above preserve sustained tracking.
        energy[dr.nj:]*=2.
        energy*=.001 if flight_mode else 1.
        reg=np.r_[energy+np.r_[np.full(dr.nj,.00005),np.full(nu-dr.nj,.05)],np.repeat([1e-5/len(g) for g in support_groups],3)]
        previous=np.r_[self.previous[:nu],np.array([self.previous[nu:].reshape(nf*4,3)[g].sum(axis=0) for g in support_groups]).ravel()]
        # Eliminate fixed inputs; duplicate lower/upper active constraints
        # make this small bound solver degenerate. Contact variables retain
        # their original order and physical bounds.
        H=A.T@A+np.diag(reg);linear=A.T@target
        free_act=np.setdiff1d(np.arange(nu),gi);free=np.r_[free_act,np.arange(nu,len(previous))]
        columns=np.ones(C.shape[1],bool);columns[gi]=False;columns[nu+gi]=False
        bounds=np.r_[lower,-upper,contact_bounds]-C[gi].T@fixed_effort
        z,fallback,iterations=minimum_support_qp(H[np.ix_(free,free)],linear[free]-H[np.ix_(free,gi)]@fixed_effort,C[np.ix_(free,np.flatnonzero(columns))],bounds[columns],lower[free_act],upper[free_act],previous[free],support_row=(row[free] if active else None),support_target=75.0,deadline=deadline,friction=.8)
        x=np.empty(len(previous));x[free]=z;x[gi]=fixed_effort
        self.previous[:nu]=x[:nu];self.previous[nu:]=0
        for group,force in zip(support_groups,x[nu:].reshape(-1,3)):self.previous[nu:].reshape(nf*4,3)[group]=force/len(group)
        u=np.clip(x[:nu],lower,upper)*dr.limit
        u[:dr.nj]=(u[:dr.nj]+dr.kp*q[dr.qi]+dr.kd*v[dr.vi])/dr.gain[:dr.nj]
        u[gi]=gimbal_command;self.last_command=u.copy()
        self.last=dict(base_error=float(np.linalg.norm(self.delta[:3])),foot_error=float(np.max(np.linalg.norm(feet-positions,axis=1))),active_feet=sorted({i//4 for i in active}),forces=self.previous[nu:].reshape(nf,4,3).sum(axis=1).tolist(),contact_bias=self.contact_bias.tolist(),lift_bias=self.lift_bias.tolist(),path_bias=self.path_bias.tolist(),joint_bias=self.joint_bias.tolist(),attitude_bias=self.attitude_bias.tolist(),fallback=bool(fallback),iterations=iterations,allocation_error=float(np.linalg.norm(A@x-target)))
        return u
