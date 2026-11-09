"""Airborne prefix for a generated walking assembly; no physical state reset."""
import numpy as np
import mujoco
from .grasp_geometry import aim_rotors_up, robot_bottom
from .equilibrium import static_balance


def flight_directions(plant,configuration):
    """Fit rotor directions to whole-assembly balance when vertical aiming fails.

    Dock joints and base posture stay fixed. This is only an initial guess;
    the finite landing path and independent physical execution still decide
    whether that configuration is usable.
    """
    d=mujoco.MjData(plant.mj);d.qpos[:]=configuration
    q,_=plant.observation(d)
    q=aim_rotors_up(plant.pin,plant.meta,q)
    check=static_balance(plant.pin,plant.meta,q,fast_lp=True)
    if check['status']!='static_screen_pass':
        from .free_flight_trim import free_flight_trim
        q,_=free_flight_trim(plant.pin,plant.meta,q,physical_check_interval=8)
    return plant.configuration(q)


def smooth(x):
    x=np.clip(x,0.,1.)
    return x*x*x*(10+x*(-15+6*x))


class LandingReference:
    def __init__(self,plant,plan,walk):
        self.plant=plant;self.m=plant.mj;self.walk=walk
        self.d=mujoco.MjData(self.m);self.delta=np.empty(self.m.nv)
        self.walk_pose=walk.pose(0.)
        def upward(q):
            return flight_directions(plant,q)
        self.landing=upward(self.walk_pose)
        support=np.asarray(plan['support_pose'])
        # A short posture continuation repairs an unsupported all-feet seed.
        # These are poses of the generated graph, not stored structures.
        for fraction in [0.,.5,1.]:
            self.hover=upward(self.interpolate(support,self.walk_pose,fraction))
            check=self.balance(self.hover)
            if check['status']=='static_screen_pass' and check['minimum_peak_effort_ratio']<=.9:break
        self.flight_posture_fraction=fraction
        self.hold_end=.8;self.morph_seconds=float(plan.get('landing',{}).get('morph_seconds',3.))
        self.morph_end=self.hold_end+self.morph_seconds;self.touch_end=self.morph_end+4.5
        # A fixed 0.6 m whole-body clearance, including collision geometry.
        self.d.qpos[:]=self.hover;qp,_=plant.observation(self.d)
        self.height=.6-robot_bottom(plant.pin,plant.meta,qp)
        self.hover[2]+=self.height
        self.air_target=self.landing.copy();self.air_target[2]+=self.height
        self.duration=self.touch_end+3.
        self.feet0,self.fv,self.fa,self.stance=walk.feet(0.)
        self.initial_balance=self.balance(self.hover)
        if self.initial_balance['status']!='static_screen_pass':raise ValueError('airborne_hold_infeasible')

    def balance(self,q):
        self.d.qpos[:]=q;qp,_=self.plant.observation(self.d)
        return static_balance(self.plant.pin,self.plant.meta,qp,fast_lp=True)

    def interpolate(self,a,b,f):
        mujoco.mj_differentiatePos(self.m,self.delta,1.,a,b)
        q=a.copy();mujoco.mj_integratePos(self.m,q,self.delta,float(f));return q

    def pose(self,t):
        if t<self.hold_end:return self.hover.copy()
        if t<self.morph_end:return self.interpolate(self.hover,self.air_target,smooth((t-self.hold_end)/(self.morph_end-self.hold_end)))
        if t<self.touch_end:
            q=self.landing.copy();q[2]+=self.height*(1-smooth((t-self.morph_end)/(self.touch_end-self.morph_end)));return q
        return self.interpolate(self.landing,self.walk_pose,smooth((t-self.touch_end)/(self.duration-self.touch_end)))

    def rates(self,t):
        h=1e-4;q=self.pose(t);before=self.pose(t-h);after=self.pose(t+h)
        v=np.empty(self.m.nv);a=v.copy();b=v.copy()
        mujoco.mj_differentiatePos(self.m,v,2*h,before,after)
        mujoco.mj_differentiatePos(self.m,a,h,before,q);mujoco.mj_differentiatePos(self.m,b,h,q,after)
        return q,v,(b-a)/h

    def phase(self,t):
        return 'HOLD' if t<self.hold_end else 'MORPH' if t<self.morph_end else 'DESCEND' if t<self.touch_end else 'TRANSFER'

    def targets(self,t):
        q,v,a=self.rates(t)
        # Load-transfer objectives ramp in only near the planned floor.
        fraction=float(smooth((t-(self.touch_end-1.))/2.))
        phase=self.phase(t)
        return phase,q,v,a,self.feet0,np.zeros_like(self.feet0),np.zeros_like(self.feet0),self.stance if fraction>0 else np.zeros_like(self.stance),fraction

    def initial_command(self,control):
        dr=control.drive;force=self.initial_balance
        u=np.r_[[j['effort_nm'] for j in force['joints']],force['thrust']]
        u[:dr.nj]=(u[:dr.nj]+dr.kp*self.hover[dr.qi])/dr.gain[:dr.nj]
        control.last_command=u.copy()
        control.previous[:dr.nj]=np.array([j['effort_nm'] for j in force['joints']])/dr.limit[:dr.nj]
        control.previous[dr.nj:self.m.nu]=np.array(force['thrust'])/dr.limit[dr.nj:]
        return u


def screen_landing(plant,plan,spec):
    """Sample the finite path's actual inertial loads, never execute it.

    Contact is absent before the nominal touchdown. Afterwards only selected
    feet actually at the floor contribute. Passing does not certify feedback,
    impact transients or continuous geometry between the checked samples.
    """
    from .walking import foot_link
    from .walking_reference import KinematicReference
    from .walking_control import GroundControl
    from .walking_load import motion_load,ground_contacts
    gait={k:np.asarray(v) if isinstance(v,list) else v for k,v in plan['gait'].items()}
    links=[foot_link(*s) for s in plan['surfaces']];gait['foot_links']=links
    walk=KinematicReference(plant,spec,gait);ref=LandingReference(plant,plan,walk)
    controller=GroundControl(plant,ref.hover,links,foot_points=gait['foot_points'])
    dr=controller.drive;peak=0.;velocity=0.;penetration=0.;reports=[]
    limits=np.array([a.get('rated_speed') or a['speed'] for a in plant.meta['actuators']])
    ranges=np.array([[a['lower'],a['upper']] for a in plant.meta['actuators']])
    times=np.unique(np.r_[0.,np.linspace(ref.hold_end,ref.morph_end,9),np.linspace(ref.morph_end,ref.touch_end,13),np.linspace(ref.touch_end,ref.duration,13)])
    for t in times:
        q,v,a=ref.rates(float(t));qp,load=motion_load(plant,dr,q,v,a)
        velocity=max(velocity,float(np.max(abs(v[dr.vi])/limits)))
        if velocity>1.+1e-6 or np.max(np.maximum(ranges[:,0]-q[dr.qi],q[dr.qi]-ranges[:,1]))>1e-6:
            raise ValueError('landing_joint_motion')
        mujoco.mj_collision(plant.mj,dr.d)
        if dr.d.ncon:penetration=max(penetration,-float(min(dr.d.contact.dist)))
        if penetration>.001:raise ValueError('landing_cad_collision')
        contacts=[]
        if t>=ref.touch_end:
            acquired=[]
            for stance,(bid,vertices) in zip(ref.stance,controller.material):
                world=vertices@dr.d.xmat[bid].reshape(3,3).T+dr.d.xpos[bid]
                acquired.append(stance and world[:,2].min()<.0002)
            contacts=ground_contacts(controller,acquired)
        result=static_balance(plant.pin,plant.meta,qp,contacts,generalized_load=load,fast_lp=True)
        value=result.get('minimum_peak_effort_ratio',float('inf'));peak=max(peak,value)
        if result['status']!='static_screen_pass' or value>.9:
            if ref.hold_end<t<ref.morph_end and ref.morph_seconds<6.:
                return screen_landing(plant,dict(plan,landing=dict(morph_seconds=6.)),spec)
            raise ValueError('landing_dynamic_load')
        reports.append(dict(time_s=float(t),peak=value,contact_points=len(contacts)))
    return dict(passed=True,peak=peak,velocity_ratio=velocity,penetration_m=penetration,
                duration_s=ref.duration,morph_seconds=ref.morph_seconds,flight_posture_fraction=ref.flight_posture_fraction,initial_clearance_m=.6,samples=reports,
                scope='finite sampled nominal path; independent delayed physical execution required')
