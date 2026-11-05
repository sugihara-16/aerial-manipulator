"""Local stance/step motion preference for poses with a body-mounted foot.

The same pose must support load and admit forward COM motion. These velocity
variables guide posture generation; they neither certify finite steps nor
replace the independent periodic and physical checks.
"""
import numpy as np
import pinocchio as pin


def residual(model,q,velocity,moving,frames,points,active,body_velocity,swing_velocity,*,data=None):
    if data is None:data=model.createData()
    pin.computeForwardKinematicsDerivatives(model,data,q,velocity,np.zeros(model.nv))
    pin.updateFramePlacements(model,data)
    pin.centerOfMass(model,data,q,velocity)
    com_velocity=np.asarray(body_velocity)
    values=[];poses=[];rates=[]
    for i,(f,point) in enumerate(zip(frames,points)):
        frame=model.frames[f];local=frame.placement*pin.SE3(np.eye(3),point)
        dq,dv=pin.getFrameVelocityDerivatives(model,data,frame.parentJoint,local,pin.LOCAL_WORLD_ALIGNED)
        value=dv@velocity;dq=dq.copy();dq[:3]+=pin.skew(value[3:])@dv[:3]
        target=np.zeros(3) if i in active else swing_velocity[i]
        weight=30. if i in active else 8.
        values.extend(weight*(value[:3]-target));poses.extend(weight*dq[:3]);rates.extend(weight*dv[:3,moving])
    dc=pin.getCenterOfMassVelocityDerivatives(model,data).copy()
    jc=pin.jacobianCenterOfMass(model,data,q).copy()
    values.extend(30*(jc@velocity-com_velocity));poses.extend(30*dc);rates.extend(30*jc[:,moving])
    return np.array(values),np.array(poses),np.array(rates)
