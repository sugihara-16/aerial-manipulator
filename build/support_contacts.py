"""Known, stationary support geometry for a conditional sticking mode.

Only geometric contact points are queried. Forces are decision variables with
nonnegative normal force and an inner Coulomb polygon; no simulator force is
read. This does not certify that the mode is actually active during execution.
"""
import numpy as np
import mujoco
from copy import copy


def support_contacts(model,q,tolerance=1e-4):
    # Expand the private geometric query only; equality with a support plane
    # can round to a tiny positive gap. This does not enable physical forces.
    query=copy(model)
    for geom in range(query.ngeom):
        if query.geom(geom).name in ['support','destination_support','floor']:
            query.geom_margin[geom]=max(query.geom_margin[geom],tolerance)
            body=int(query.geom_bodyid[geom])
            # The broad phase uses the separately compiled body margin.
            query.body_margin[body]=max(query.body_margin[body],tolerance)
    d=mujoco.MjData(query);d.qpos[:]=q;mujoco.mj_fwdPosition(query,d)
    obj=model.geom('object').id;records=[]
    for c in d.contact:
        a,b=map(int,c.geom)
        if obj not in (a,b) or c.dist>tolerance:continue
        other=b if a==obj else a
        if model.geom(other).name not in ['support','destination_support','floor']:continue
        body=int(model.geom_bodyid[obj]);R=d.xmat[body].reshape(3,3)
        frame=c.frame.reshape(3,3).copy()*(1 if b==obj else -1)
        # Sign yields an inward force on the object, with a positive normal.
        records.append(dict(body=body,point=R.T@(c.pos-d.xpos[body]),frame=frame,
                            friction=float(max(model.geom_friction[obj,0],model.geom_friction[other,0]))))
    return records


def support_jacobian(model,data,records):
    rows=[]
    for item in records:
        b=item['body'];point=data.xpos[b]+data.xmat[b].reshape(3,3)@item['point']
        J=np.zeros((3,model.nv));mujoco.mj_jac(model,data,J,None,point,b)
        rows.extend(item['frame']@J)
    return np.array(rows).reshape(-1,model.nv)
