"""Prepared three-module grasp diagnostic. NOT task-driven structure generation."""
import argparse
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET
import numpy as np
import pinocchio as pin
import mujoco
from amsrr_det.mujoco_model import make_plant, numbers, coat_grippers
from amsrr_det.flight import FlightController
from amsrr_det.grasp_geometry import fit_contacts
from amsrr_det.grasp_control import GraspReferences, surface_points
from amsrr_det.equilibrium import static_balance
from amsrr_det.task_audit import TaskAudit


def smooth(t):
    u=np.clip(t,0,1)
    return u*u*u*(10+u*(-15+6*u))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path('artifacts/runs/grasp_smoke'))
    parser.add_argument('--duration',type=float,default=40.)
    parser.add_argument('--timestep',type=float,default=.001)
    parser.add_argument('--noslip',type=int,default=0)
    parser.add_argument('--mesh-cache',default='build/collision_fast')
    parser.add_argument('--coating',action='store_true')
    parser.add_argument('--feedback',action='store_true')
    parser.add_argument('--transport',action='store_true')
    parser.add_argument('--planned-trim',action='store_true')
    parser.add_argument('--impratio',type=float,default=1.)
    parser.add_argument('--integrator',default='implicitfast',choices=['implicitfast','discrete'])
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    edges=[dict(a=0,port_a='pitch1',b=1,port_b='yaw1'),dict(a=0,port_a='pitch2',b=2,port_b='yaw2')]
    p=make_plant(3,edges,payload={'size':[.30,.40,.15],'mass':1.},
                 mesh_cache=args.mesh_cache,floor=True,timestep=args.timestep)
    doc=ET.fromstring(p.xml);world=doc.find('worldbody')
    doc.find('option').set('noslip_iterations',str(args.noslip))
    doc.find('option').set('impratio',str(args.impratio))
    doc.find('option').set('integrator',args.integrator)
    ET.SubElement(world,'geom',name='support',type='box',pos='.745 0 .0875',size='.19 .24 .0875',rgba='.4 .4 .4 1')
    if args.transport:
        ET.SubElement(world,'geom',name='destination_support',type='box',pos='1.145 0 .0875',size='.19 .24 .0875',rgba='.4 .4 .4 1')
    for g in doc.findall('.//geom'):
        if any(g.get('name','').startswith(name+'/') for name in ['m1/yaw_dock_mech2','m2/yaw_dock_mech1']):
            g.set('friction','4.5 0 0');g.set('priority','1')
    xml=ET.tostring(doc,encoding='unicode')
    if args.coating:xml=coat_grippers(xml,['m1/yaw_dock_mech2','m2/yaw_dock_mech1'])
    (args.output/'scene.xml').write_text(xml)
    p.mj=mujoco.MjModel.from_xml_string(xml);d=mujoco.MjData(p.mj)
    q=pin.neutral(p.pin);q[2]=.23575
    oq=p.meta['object']['q'];q[oq:oq+3]=[.745,0,.25]
    surfaces=[(1,'yaw2'),(2,'yaw1')];normals=[[0,-1,0],[0,1,0]]
    closed,fit=fit_contacts(p.pin,p.meta,q,surfaces,[[.745,.199,.25],[.745,-.199,.25]],normals)
    q,openfit=fit_contacts(p.pin,p.meta,closed,surfaces,[[.745,.24,.25],[.745,-.24,.25]],normals)
    (args.output/'plan.json').write_text(json.dumps(dict(q_open=q.tolist(),q_closed=closed.tolist(),fit=fit,open_fit=openfit),default=float))
    d.qpos[:]=p.configuration(q);controller=FlightController(p.pin,p.meta,q)
    q0=q.copy();mujoco.mj_forward(p.mj,d)
    trim0=trim1=None
    if args.planned_trim:
        contacts=[];pd=p.pin.createData();pin.framesForwardKinematics(p.pin,pd,closed)
        obj=pd.oMf[p.meta['object']['frame']]
        for surface,normal in zip(surfaces,normals):
            frame,points=surface_points(p.pin,p.meta,closed,surface)
            for point in points:
                contacts.append(dict(frame=p.pin.frames[frame].name,point=point,target_frame='object',
                    target_point=obj.actInv(pd.oMf[frame].act(point)),normal=-np.asarray(normal),mu=2.25,max_force=30/len(points)))
        trim0=static_balance(p.pin,p.meta,q,payload_fraction=0.,balance_thrust=True)
        trim1=static_balance(p.pin,p.meta,closed,contacts,balance_thrust=True)
        (args.output/'trim.json').write_text(json.dumps(dict(unloaded=trim0,loaded=trim1)))
        if any(t['status']!='static_screen_pass' for t in [trim0,trim1]):raise RuntimeError('no load-feasible trim')
        f0,f1=(np.array(t['thrust']) for t in [trim0,trim1])
        j0,j1=(np.array([j['effort_nm'] for j in t['joints']]) for t in [trim0,trim1])
    samples=[surface_points(p.pin,p.meta,closed,s) for s in surfaces]
    audit=TaskAudit(p,['m1/yaw_dock_mech2','m2/yaw_dock_mech1'],[.3,.4,.15],.175)
    records=[];start=time.perf_counter();stop_reason=None;grasp=None;control_times=[]
    for step in range(round(args.duration/.01)):
        now=step*.01;q,v=p.observation(d);control_start=time.perf_counter()
        target=pin.interpolate(p.pin,q0,closed,float(smooth((now-2)/18)))
        target[2]+=.12*smooth((now-24)/8)
        payload=float(smooth((now-23)/2));tv=None
        if args.feedback and now>=22:
            if grasp is None:
                grasp=GraspReferences(p.pin,p.meta,q,target,surfaces,[[0,.2,0],[0,-.2,0]],normals,samples=samples)
            obj_target=q0[oq:oq+7].copy();obj_target[2]+=.12*smooth((now-24)/8)
            if args.transport:
                obj_target[0]+=.4*smooth((now-36)/12)
                obj_target[2]-=.12*smooth((now-50)/10)
                payload*=1-smooth((now-62)/2)
            release=args.transport and now>=64
            target,tv=grasp.update(q,v,obj_target,preload=np.full(2,-.03) if release else None,normal_speed=.01 if release else .001)
        cmd,f=controller.update(q,v,target,target_velocity=tv,payload_mass=payload,payload_com=q[oq:oq+3],
            joint_bias=(1-payload)*j0+payload*j1 if args.planned_trim else None,
            thrust_reference=(1-payload)*f0+payload*f1 if args.planned_trim else None)
        control_times.append(time.perf_counter()-control_start)
        d.ctrl[:]=np.r_[cmd,f]
        for _ in range(round(.01/args.timestep)):
            try:
                mujoco.mj_step(p.mj,d)
                audit.update(d,carried=36<=now<50 if args.transport else now>=34)
            except mujoco.FatalError as error:
                stop_reason='numerical_failure: '+str(error);break
            if any(d.warning[i].number for i in [mujoco.mjtWarning.mjWARN_BADQPOS,mujoco.mjtWarning.mjWARN_BADQVEL,mujoco.mjtWarning.mjWARN_BADQACC]):
                stop_reason='numerical_failure';break
        if stop_reason:break
        if not np.all(np.isfinite(d.qpos)) or not np.all(np.isfinite(d.qvel)) or any(d.warning[i].number for i in [mujoco.mjtWarning.mjWARN_BADQPOS,mujoco.mjtWarning.mjWARN_BADQVEL,mujoco.mjtWarning.mjWARN_BADQACC]):
            stop_reason='numerical_failure';break
        if step%10==0:
            contacts=[]
            for ci,c in enumerate(d.contact):
                a=mujoco.mj_id2name(p.mj,mujoco.mjtObj.mjOBJ_GEOM,c.geom1)
                b=mujoco.mj_id2name(p.mj,mujoco.mjtObj.mjOBJ_GEOM,c.geom2)
                cf=np.zeros(6);mujoco.mj_contactForce(p.mj,d,ci,cf)
                contacts.append(dict(a=a,b=b,gap=float(c.dist),force=cf.tolist(),position=c.pos.tolist(),frame=c.frame.tolist()))
            q,v=p.observation(d)
            records.append(dict(t=float(d.time),root=q[:7].tolist(),object=q[oq:oq+7].tolist(),
                configuration=q.tolist(),velocity=v.tolist(),contacts=contacts,command=cmd.tolist(),thrust=f.tolist(),
                actuator_force=d.actuator_force.tolist(),grasp=grasp.last_diagnostics if grasp else None))
    result=dict(scope='prepared_three_module_yaw_diagnostic_only',task_success=False,
                wall_s=time.perf_counter()-start,records=records,stop_reason=stop_reason,
                audit=audit.result(),control_time_s=dict(mean=float(np.mean(control_times)),p95=float(np.quantile(control_times,.95)),maximum=max(control_times)),
                contact_model='Coating K/C converted per contact; MuJoCo regularization and collision approximation retained' if args.coating else 'Uncalibrated MuJoCo soft constraints')
    (args.output/'trace.json').write_text(json.dumps(result))
    print(json.dumps(dict(wall_s=result['wall_s'],final_object=records[-1]['object'],
                         final_root=records[-1]['root'],contacts=len(records[-1]['contacts']))))


if __name__=='__main__':main()
