"""Kinematic one-cycle synthesis under stance and swing-foot constraints."""
import numpy as np
import mujoco
from .walking import rotation_z


def foot_targets(time,period,speed,nominal,*,height=.02,curvature=0.,origin=None,duty=.62,phases=None):
    count=len(nominal)
    offset=np.arange(count)/count if phases is None else np.asarray(phases)
    p=time/period+offset;cycle=np.floor(p);p-=cycle
    stance=p<duty;s=np.clip((p-duty)/(1-duty),0,1);duration=(1-duty)*period
    blend=s**3*(10-15*s+6*s*s);rate=30*s*s*(1-s)**2/duration
    accel=60*s*(1-s)*(1-2*s)/duration**2
    anchor=period*(cycle+.5*duty-offset)
    origin=np.zeros(3) if origin is None else np.asarray(origin)
    def footholds(t):
        theta=curvature*speed*t;offset=nominal-origin
        result=nominal.copy()
        if abs(curvature)<1e-10:result[:,0]+=speed*t
        else:
            result[:,0]=origin[0]+np.sin(theta)/curvature+np.cos(theta)*offset[:,0]-np.sin(theta)*offset[:,1]
            result[:,1]=origin[1]+(1-np.cos(theta))/curvature+np.sin(theta)*offset[:,0]+np.cos(theta)*offset[:,1]
        return result
    start,end=footholds(anchor),footholds(anchor+period);difference=end-start
    positions=start+blend[:,None]*difference;velocity=rate[:,None]*difference;acceleration=accel[:,None]*difference
    # Zero position, velocity and acceleration at both support transitions.
    positions[:,2]+=64*height*s**3*(1-s)**3
    velocity[:,2]=192*height*s**2*(1-s)**2*(1-2*s)/duration
    acceleration[:,2]=384*height*s*(1-s)*(1-5*s+5*s*s)/duration**2
    return positions,velocity,acceleration,stance


def foot_headings(time,period,speed,count,curvature,*,duty=.62,phases=None):
    offset=np.arange(count)/count if phases is None else np.asarray(phases)
    p=time/period+offset
    cycle=np.floor(p);p-=cycle;s=np.clip((p-duty)/(1-duty),0,1)
    blend=s**3*(10-15*s+6*s*s)
    return curvature*speed*period*(cycle+.5*duty-offset+blend)


class CycleKinematics:
    def __init__(self,plant,controller,nominal):
        self.plant=plant;self.m=plant.mj;self.d=mujoco.MjData(self.m);self.controller=controller
        self.nominal=nominal.copy();dr=controller.drive
        self.indices=np.r_[np.arange(6),dr.vi[dr.dock]];self.jac=np.zeros((3,self.m.nv));self.jr=self.jac.copy()
        self.qi=dr.qi[dr.dock];self.limits=np.array([[a['lower']+.02,a['upper']-.02] for a in dr.actuators if a['role']=='dock'])
        self.delta=np.zeros(self.m.nv)
        self.d.qpos[:]=nominal;mujoco.mj_kinematics(self.m,self.d)
        self.rotations=[self.d.xmat[b].reshape(3,3).copy() for b,_ in controller.material]
        from .grasp_geometry import part_bounds
        self.bounds=[(self.m.body(f'm{i}/{link}').id,center,half)
                     for i in range(plant.meta['modules']) for link,center,half in part_bounds()
                     if f'm{i}/{link}' not in controller.links]
        self.bids=np.array([a[0] for a in self.bounds]);self.centers=np.array([a[1] for a in self.bounds]);self.halves=np.array([a[2] for a in self.bounds])
        self.jb=np.empty((3,self.m.nv));self.floor=self.m.geom('floor').id

    def points(self):
        """Track the lowest physical material in z, including foot tilt."""
        points=self.controller.positions(self.d)
        material=[]
        for i,(bid,vertices) in enumerate(self.controller.material):
            world=vertices@self.d.xmat[bid].reshape(3,3).T+self.d.xpos[bid]
            lowest=world[np.argmin(world[:,2])];points[i,2]=lowest[2];material.append(lowest)
        return points,material

    def fit(self,seed,body,feet,iterations=35,rotations=None):
        from .walking_native import CycleGeometry
        if not hasattr(self,'compiled_seed'):self.compiled_seed=CycleGeometry(self)
        return self.compiled_seed.fit(seed,body,feet,iterations,rotations)

    def generate(self,spec,period=1.,samples=33,height=.03,iterations=35,duty=.62,capacity_timing=True,fixed_duty=None,fixed_phases=None):
        d=self.d;m=self.m;d.qpos[:]=self.nominal;mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
        origin=d.subtree_com[0].copy()
        feet=self.controller.positions(d);feet[:,2]=0.
        from .walking_load import support_timing,support_phases
        if fixed_phases is not None and fixed_duty is not None:
            phases=np.asarray(fixed_phases)
            phase_report=dict(method='generated_support_groups',duty=float(fixed_duty))
        elif capacity_timing:
            phases,phase_report=support_timing(self.plant,self.controller,self.nominal)
        else:
            phases,phase_report=support_phases(self.plant,self.controller,self.nominal)
            phase_report=dict(phase_report,duty=.7 if len(feet)==3 else duty)
        duty=phase_report['duty']
        if fixed_duty is not None:
            duty=float(fixed_duty);phase_report=dict(phase_report,duty=duty)
        if fixed_phases is not None:
            phases=np.asarray(fixed_phases);phase_report=dict(method='shared_body_support_groups',duty=duty)
        values=[];errors=[];seed=self.nominal.copy()
        root_rotation=np.empty((3,3));mujoco.mju_quat2Mat(root_rotation.ravel(),self.nominal[3:7])
        for now in np.linspace(0,period,samples):
            position,heading,*_=spec.path(now)
            target=self.nominal[:7].copy();target[:3]=origin+rotation_z(heading)@(target[:3]-origin)+np.r_[position,0.]
            mujoco.mju_mat2Quat(target[3:7],(rotation_z(heading)@root_rotation).ravel())
            fg,*_=foot_targets(now,period,spec.speed,feet,height=height,curvature=spec.curvature,origin=origin,duty=duty,phases=phases)
            rotations=[rotation_z(a)@R for a,R in zip(foot_headings(now,period,spec.speed,len(feet),spec.curvature,duty=duty,phases=phases),self.rotations)]
            seed,error=self.fit(seed,target,fg,iterations=max(35,iterations) if not values else iterations,rotations=rotations);values.append(seed.copy());errors.append(error)
        values=np.array(values)
        closure=values[-1].copy();position,heading,*_=spec.path(period);R=rotation_z(-heading)
        closure[:3]=origin+R@(closure[:3]-origin-np.r_[position,0.])
        end_rotation=np.empty((3,3));mujoco.mju_quat2Mat(end_rotation.ravel(),closure[3:7]);mujoco.mju_mat2Quat(closure[3:7],(R@end_rotation).ravel())
        delta=np.zeros(m.nv);mujoco.mj_differentiatePos(m,delta,1.,values[0],closure)
        return dict(period=period,duty=duty,phases=phases,phase_selection=phase_report,configurations=values,feet=feet,
                    foot_points=self.controller.foot_points.copy(),errors=errors,closure_error=float(np.max(abs(delta))),origin=origin,height=height)


def validate_cycle(plant,gait,*,spec,penetration_tolerance=.001):
    """Independent full-CAD/actuator check before any physical trial."""
    from .walking_cycle import CycleReference
    m=plant.mj;d=mujoco.MjData(m);ref=CycleReference(plant,spec,gait)
    ids=[m.joint(a['name']).id for a in plant.meta['actuators']]
    vi=m.jnt_dofadr[ids];qi=m.jnt_qposadr[ids];ranges=m.jnt_range[ids]
    limits=np.array([a['rated_speed'] if a['role']=='dock' and a.get('rated_speed') is not None else a['speed'] for a in plant.meta['actuators']])
    # Inspect the actual periodic interpolation, including points between the
    # fitted knots. Knot-to-knot secants miss spline speed/angle overshoot.
    samples=4*(len(gait['configurations'])-1)+1
    worst=0.;pair=None;velocity_ratio=0.;angle_excess=0.
    for t in np.linspace(0,ref.period,samples):
        q,v,_=ref.rates(t)
        if not np.all(np.isfinite(q)) or not np.all(np.isfinite(v)):
            return dict(passed=False,reason='nonfinite_periodic_motion')
        velocity_ratio=max(velocity_ratio,float(np.max(abs(v[vi])/limits)))
        angle_excess=max(angle_excess,float(np.max(ranges[:,0]-q[qi])),float(np.max(q[qi]-ranges[:,1])))
        d.qpos[:]=q;mujoco.mj_kinematics(m,d);mujoco.mj_collision(m,d)
        if d.ncon:
            contact=d.contact[int(np.argmin(d.contact.dist))]
            if contact.dist<worst:
                worst=float(contact.dist);pair=[m.geom(int(g)).name for g in contact.geom]
    result=dict(penetration=worst,pair=pair,velocity_ratio=velocity_ratio,angle_excess_rad=angle_excess,
                samples=samples,interpolated_motion=True,speed_basis="Dock manufacturer rated; gimbal existing configured speed",
                closure_error=gait['closure_error'],position_error=max(gait['errors']))
    result['passed']=bool(worst>=-penetration_tolerance and velocity_ratio<=1. and angle_excess<=1e-8 and gait['closure_error']<.002 and max(gait['errors'])<.005)
    return result
