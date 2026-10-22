"""Pad loads of an already generated finite-slip reference, without force truth."""
import numpy as np,mujoco


def planned_pad_loads(local,q,velocity):
    m=local.m;local.at(q);local.d.ctrl[:]=0.;mujoco.mj_forward(m,local.d)
    oid=m.geom('object').id;loads=np.zeros(len(local.links));rows=[]
    for contact in local.d.contact:
        ga,gb=map(int,contact.geom)
        if oid not in (ga,gb) or contact.dist>1e-7 or contact.efc_address<0:continue
        other=gb if ga==oid else ga
        pad=next((i for i,link in enumerate(local.links)if m.geom(other).name.startswith(link+'/')),None)
        if pad is None:continue
        adr=int(contact.efc_address);h=m.opt.timestep
        kbip=local.d.efc_KBIP[adr];spring=kbip[0]*kbip[2];damping=kbip[1]
        coefficient=local.d.efc_D[adr]/(1+h*damping+h*h*spring)
        kn=coefficient*spring;cn=coefficient*(damping+h*spring)
        normal=local.d.efc_J.reshape(local.d.nefc,m.nv)[adr]
        gap=local.d.efc_pos[adr]-local.d.efc_margin[adr]
        load=-kn*gap-cn*(normal@velocity);loads[pad]+=load
        rows.append(dict(pad=pad,normal_n=float(load)))
    if len(rows)!=len(local.elements) or np.min(loads)<=0 or not np.all(np.isfinite(loads)):
        raise ValueError('finite reference contact correspondence or load is invalid')
    return loads
