"""Proximity detection without adding forces or changing physical geometry."""
import xml.etree.ElementTree as ET
import numpy as np
import copy
import mujoco

PART_CLEARANCE = .001
ENVIRONMENT_CLEARANCE = .03
SUPPORT_NAMES = ('floor', 'support', 'destination_support')


def forward_collision(model,data):
    """Refresh rigid geometry without constructing unused force constraints."""
    if not model.nflex and not (model.opt.enableflags & mujoco.mjtEnableBit.mjENBL_SLEEP):
        mujoco.mj_kinematics(model,data);mujoco.mj_collision(model,data)
    else:
        mujoco.mj_fwdPosition(model,data)


def proximity_model(model, *, omit_permitted_support=False):
    """Separate model: mesh CCD with a gap can change active-contact results."""
    result=copy.copy(model)
    # Buffered curved shapes need more EPA refinements than physical contact
    # geometry. Keep the same accuracy; allow convergence in this query copy.
    result.opt.ccd_iterations=max(500,result.opt.ccd_iterations)
    result.geom_gap[:]=PART_CLEARANCE/2
    supports=[];object_id=None
    for i in range(result.ngeom):
        name=result.geom(i).name
        if name in SUPPORT_NAMES:
            result.geom_gap[i]=ENVIRONMENT_CLEARANCE;supports.append(i)
        if name=='object':object_id=i
    # Object/support pairs are always permitted by ContactClassifier. They
    # still participate in physical contact/penetration audits, but their
    # buffered distance is not a clearance requirement. Remove only these
    # queries using an otherwise unused collision bit; preserve every other
    # pair and all original body exclusions.
    used=int(np.bitwise_or.reduce(result.geom_contype|result.geom_conaffinity,initial=0))
    bit=next((1<<i for i in range(31) if not used&(1<<i)),None)
    if omit_permitted_support and object_id is not None and supports and bit is not None:
        eligible=((result.geom_contype&result.geom_conaffinity[object_id])!=0)|((result.geom_conaffinity&result.geom_contype[object_id])!=0)
        eligible[object_id]=False;eligible[supports]=False
        result.geom_contype[object_id]=bit;result.geom_conaffinity[object_id]=0
        result.geom_conaffinity[eligible]|=bit
    # Broad/mid-phase collision bounds cache each body's detection margin.
    # Merely changing geom_gap leaves separated body pairs out of the search.
    mujoco.mj_setConst(result,mujoco.MjData(result))
    return result


def detection_buffers(xml):
    # MuJoCo 3.16 detects at margin + gap, acts only at margin. Do not inflate
    # margin: that would change the physical onset of contact and grasp forces.
    doc = ET.fromstring(xml)
    default = doc.find('default/geom')
    default.set('gap', str(PART_CLEARANCE / 2))
    for geom in doc.findall('.//worldbody/geom'):
        if geom.get('name') in SUPPORT_NAMES:
            geom.set('gap', str(ENVIRONMENT_CLEARANCE))
    return ET.tostring(doc, encoding='unicode')


class ContactClassifier:
    """Geometry identities are fixed; only contact indices change each step."""
    def __init__(self,model,names,selected_links):
        self.object=model.geom('object').id
        self.supports=np.array([name in SUPPORT_NAMES for name in names])
        self.selected=np.array([any(name.startswith(link+'/') for link in selected_links) for name in names])

    def __call__(self,data):
        g1,g2=np.asarray(data.contact.geom1),np.asarray(data.contact.geom2)
        first,second=g1==self.object,g2==self.object;other=np.where(first,g2,g1)
        intended=(first|second)&(self.supports[other]|self.selected[other])
        limits=np.where(self.supports[g1]|self.supports[g2],ENVIRONMENT_CLEARANCE,PART_CLEARANCE)
        return intended,limits


def classify_contacts(model,data,names,selected_links):
    return ContactClassifier(model,names,selected_links)(data)


class JointMotionBound:
    """Tree-length upper bound on movement, without transforming every mesh."""
    def __init__(self,model,bodies,radius):
        if np.any(model.body_jntnum>1) or np.any(~np.isin(model.jnt_type,[mujoco.mjtJoint.mjJNT_FREE,mujoco.mjtJoint.mjJNT_HINGE])):
            raise ValueError('motion bound requires free bases and single hinges')
        hinges=np.flatnonzero(model.jnt_type==mujoco.mjtJoint.mjJNT_HINGE)
        self.qi=model.jnt_qposadr[hinges].copy();columns={j:i for i,j in enumerate(hinges)}
        self.levers=np.zeros((len(bodies),len(hinges)));self.roots={}
        for row,body in enumerate(bodies):
            reach=float(radius[row])
            while body:
                if model.body_jntnum[body]:
                    joint=model.body_jntadr[body]
                    lever=reach+np.linalg.norm(model.jnt_pos[joint])
                    if model.jnt_type[joint]==mujoco.mjtJoint.mjJNT_FREE:
                        self.roots.setdefault(int(model.jnt_qposadr[joint]),[]).append((row,lever));break
                    self.levers[row,columns[joint]]=lever
                    # Rotation about an off-origin pivot also translates the
                    # body's origin, by at most twice the pivot offset.
                    reach+=2*np.linalg.norm(model.jnt_pos[joint])
                reach+=np.linalg.norm(model.body_pos[body]);body=model.body_parentid[body]
        self.roots={key:(np.array([x[0] for x in values]),np.array([x[1] for x in values])) for key,values in self.roots.items()}

    def maximum(self,states,anchor):
        states=np.atleast_2d(states)
        movement=abs(states[:,self.qi]-anchor[self.qi])@self.levers.T
        for k,(rows,reach) in self.roots.items():
            translation=np.linalg.norm(states[:,k:k+3]-anchor[k:k+3],axis=1)
            quat=states[:,k+3:k+7];initial=anchor[k+3:k+7]
            rotation=np.minimum(2.,2*np.minimum(np.linalg.norm(quat-initial,axis=1),np.linalg.norm(quat+initial,axis=1)))
            movement[:,rows]+=translation[:,None]+rotation[:,None]*reach
        return movement.max(axis=1,initial=0.)


class MotionBoundedClearance:
    """Reuse a distance query only while rigid-body motion proves it safe.

    A point's displacement is bounded by body translation plus body radius
    times the Frobenius norm of its rotation change. Twice the largest bound
    covers both bodies of every pair. The extra halo supplies known distance
    for pairs absent from the previous query; it changes no physical model.
    """
    def __init__(self,model,selected_links,halo=.003, *, omit_permitted_support=True):
        self.model=proximity_model(model,omit_permitted_support=omit_permitted_support);self.model.geom_gap+=halo/2
        mujoco.mj_setConst(self.model,mujoco.MjData(self.model))
        self.data=mujoco.MjData(self.model);self.halo=halo
        self.names=[model.geom(i).name for i in range(model.ngeom)]
        self.classifiers={allowed:ContactClassifier(self.model,self.names,selected_links if allowed else []) for allowed in [False,True]}
        radius=np.zeros(model.nbody)
        np.maximum.at(radius,model.geom_bodyid,np.linalg.norm(model.geom_pos,axis=1)+model.geom_rbound)
        self.bodies=np.unique(model.geom_bodyid);self.radius=radius[self.bodies]
        try:self.joint_motion=JointMotionBound(model,self.bodies,self.radius)
        except ValueError:self.joint_motion=None
        self.anchor=None;self.last_allowed=None;self.queries=0;self.reused=0

    def check_many(self,states,grasp_contact_allowed=True):
        """Check every supplied pose, batching the same conservative bound."""
        states=np.atleast_2d(states)
        if self.anchor is not None and self.joint_motion is not None and grasp_contact_allowed==self.last_allowed:
            oldq,_,_,part,environment=self.anchor
            bound=2*self.joint_motion.maximum(states,oldq)
            worst=float(bound.max(initial=0.))
            if min(part-PART_CLEARANCE,environment-ENVIRONMENT_CLEARANCE)-worst>=-1e-6:
                self.reused+=len(states)
                return [],part-worst,environment-worst
        part,environment=np.inf,np.inf
        for pose in states:
            bad,p,e=self.check(pose,grasp_contact_allowed)
            part,environment=min(part,p),min(environment,e)
            if bad:return bad,part,environment
        return [],part,environment

    def check(self,qpos,grasp_contact_allowed=True):
        m,d=self.model,self.data;d.qpos[:]=qpos
        if self.anchor is not None and grasp_contact_allowed==self.last_allowed:
            oldq,pos,mat,part,environment=self.anchor
            if self.joint_motion is not None:bound=2*float(self.joint_motion.maximum(qpos,oldq)[0])
            else:
                mujoco.mj_kinematics(m,d)
                movement=np.linalg.norm(d.xpos[self.bodies]-pos,axis=1)+self.radius*np.linalg.norm(d.xmat[self.bodies]-mat,axis=1)
                bound=2*movement.max(initial=0)
            if min(part-PART_CLEARANCE,environment-ENVIRONMENT_CLEARANCE)-bound>=-1e-6:
                self.reused+=1
                return [],part-bound,environment-bound
        # Rigid distance queries need poses and collision detection, not
        # inertia factorization or contact-force constraint construction.
        # Preserve the full pipeline for deformable/sleeping models.
        forward_collision(m,d)
        self.queries+=1
        intended,limits=self.classifiers[grasp_contact_allowed](d)
        gap=np.asarray(d.contact.dist);is_environment=limits==ENVIRONMENT_CLEARANCE
        part=float(gap[~intended & ~is_environment].min(initial=PART_CLEARANCE+self.halo))
        environment=float(gap[~intended & is_environment].min(initial=ENVIRONMENT_CLEARANCE+self.halo))
        self.anchor=(qpos.copy(),d.xpos[self.bodies].copy(),d.xmat[self.bodies].copy(),part,environment)
        self.last_allowed=grasp_contact_allowed
        violations=[tuple(sorted((self.names[c.geom1],self.names[c.geom2])))
                    for c,limit,allowed in zip(d.contact,limits,intended) if not allowed and c.dist<limit-1e-6]
        return violations,part,environment
