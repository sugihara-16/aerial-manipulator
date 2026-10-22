"""Offline reference integration of a declared drive model.

Independent Radau time integration, quaternion derivatives and force assembly.
This is a software oracle for the same nominal physics, NOT a real-robot oracle.
Contact reference below supports frictionless finite planes; friction-history
integration needs its own event-aware reference and is not silently approximated.
"""
import numpy as np
import pinocchio as pin
from scipy.integrate import solve_ivp


def replay_reference(model,meta,scenario,rows,end_time,**solver_options):
    """Replay a fixed recorded command schedule, including each hold boundary."""
    current=dict(scenario);trace=[];peak_speed=0.;calls=0
    peaks=np.zeros(len(meta['actuators']))
    indices=[a['v'] for a in meta['actuators']]
    for i,row in enumerate(rows):
        stop=rows[i+1]['time'] if i+1<len(rows) else end_time
        if stop<=row['time']:continue
        command=dict(thrust=row['thrust_command'],position=row['position_command'],velocity=row['velocity_command'])
        result=integrate_reference(model,meta,current,command,stop-row['time'],**solver_options)
        peaks=np.maximum(peaks,np.max(abs(result['trajectory'][model.nq+np.asarray(indices)]),axis=1))
        peak_speed=float(np.max(peaks))
        calls+=result['nfev']
        trace.append(dict(time=stop,q=result['q'],v=result['v']))
        current.update(q0=pin.normalize(model,result['q']).tolist(),v0=result['v'].tolist(),f0=result['f'].tolist(),tau0=result['tau'].tolist())
    return dict(trace=trace,peak_joint_speed=peak_speed,peak_joint_speeds=peaks,nfev=calls,final=current)


def integrate_reference(model,meta,scenario,command,duration,rtol=1e-9,atol=1e-11,
                        max_step=.001,output_step=.00001):
    if duration<=0 or min(rtol,atol,max_step,output_step)<=0:
        raise ValueError('positive duration, solver tolerances and sampling step required')
    nq,nv=model.nq,model.nv;nr=len(meta['rotors']);na=len(meta['actuators'])
    data=model.createData();q=np.array(scenario.get('q0',pin.neutral(model)),dtype=float)
    v=np.array(scenario.get('v0',np.zeros(nv)));f=np.array(scenario.get('f0',np.zeros(nr)));tau=np.array(scenario.get('tau0',np.zeros(na)))
    position=np.array(command.get('position',[q[a['q']] for a in meta['actuators']]))
    velocity=np.array(command.get('velocity',np.zeros(na)));thrust=np.array(command.get('thrust',np.zeros(nr)))
    kp=np.array([a['kp'] for a in meta['actuators']]);kd=np.array([a['kd'] for a in meta['actuators']]);limit=np.array([a['effort'] for a in meta['actuators']])
    qi=np.array([a['q'] for a in meta['actuators']]);vi=np.array([a['v'] for a in meta['actuators']])
    lag=np.array([scenario.get('gimbal_torque_lag_s' if a['role']=='vectoring' else 'dock_torque_lag_s',0.) for a in meta['actuators']]);fl= scenario.get('thrust_lag_s',.03)
    free=list(meta['bases'])+([meta['object']] if meta['object'] else [])
    patches=scenario.get('patches',[])
    if any(p['mu']!=0 for p in patches):raise ValueError('friction history reference not implemented')
    def fun(t,x):
        q=x[:nq];v=x[nq:nq+nv];f=x[nq+nv:nq+nv+nr];tau=x[-na:]
        pin.forwardKinematics(model,data,q,v);pin.updateFramePlacements(model,data)
        command_tau=np.clip(kp*(position-q[qi])+kd*(velocity-v[vi]),-limit,limit)
        actual=np.where(lag>0,tau,command_tau);effort=np.zeros(nv);effort[vi]=actual
        ext=[pin.Force.Zero() for _ in range(model.njoints)]
        for r,force in zip(meta['rotors'],f if fl>0 else thrust):
            frame=model.frames[r['frame']]
            ext[frame.parentJoint]+=frame.placement.act(pin.Force(np.array([0.,0.,force]),np.zeros(3)))
        for patch in patches:
            fid=meta['frames'][patch['frame']];target=meta['frames'].get(patch.get('target_frame',''))
            T=pin.SE3.Identity() if target is None else data.oMf[target]
            n=np.array(patch['normal'])
            for point in patch['points']:
                point=np.array(point);world=data.oMf[fid].act(point);rel=T.actInv(world)
                gap=n@rel-patch['offset']
                if gap>=0 or np.any(rel<patch['lower']) or np.any(rel>patch['upper']):continue
                def point_v(frame_id,local):
                    fr=model.frames[frame_id];vv=data.v[fr.parentJoint]
                    return data.oMi[fr.parentJoint].rotation@(vv.linear+np.cross(vv.angular,fr.placement.act(local)))
                relative_v=point_v(fid,point)-(np.zeros(3) if target is None else point_v(target,rel))
                vn=n@(T.rotation.T@relative_v)
                normal_force=max(0.,(-patch['kn']*gap-patch['cn']*vn)/4)
                force=T.rotation@n*normal_force
                for target_id,sign in [(fid,1)]+([] if target is None else [(target,-1)]):
                    joint=model.frames[target_id].parentJoint
                    localf=data.oMi[joint].rotation.T@(sign*force);p=data.oMi[joint].actInv(world)
                    ext[joint]+=pin.Force(localf,np.cross(p,localf))
        acceleration=pin.aba(model,data,q,v,effort,ext).copy()
        qdot=np.zeros(nq);qdot[qi]=v[vi]
        for base in free:
            iq,iv=base['q'],base['v'];xyz=q[iq+3:iq+6];w=q[iq+6];omega=v[iv+3:iv+6]
            qdot[iq:iq+3]=data.oMi[base['joint']].rotation@v[iv:iv+3]
            qdot[iq+3:iq+6]=.5*(w*omega+np.cross(xyz,omega));qdot[iq+6]=-.5*(xyz@omega)
        fdot=(thrust-f)/fl if fl>0 else np.zeros(nr)
        taudot=np.zeros(na);mask=lag>0;taudot[mask]=(command_tau[mask]-tau[mask])/lag[mask]
        return np.concatenate([qdot,acceleration,fdot,taudot])
    result=solve_ivp(fun,(0,duration),np.concatenate([q,v,f,tau]),method='Radau',
                     rtol=rtol,atol=atol,max_step=min(max_step,duration/10),dense_output=True)
    if not result.success:raise RuntimeError(result.message)
    # Sampling the dense interpolant avoids changing the reported peak merely
    # because the adaptive solver happened to select different output times.
    # This is still a sampled peak, not a certified continuous-time maximum.
    times=np.unique(np.concatenate([result.t,np.linspace(0,duration,int(np.ceil(duration/output_step))+1)]))
    return dict(q=result.y[:nq,-1],v=result.y[nq:nq+nv,-1],f=result.y[nq+nv:nq+nv+nr,-1],tau=result.y[-na:,-1],nfev=result.nfev,
                times=times,trajectory=result.sol(times),adaptive_times=result.t,
                settings=dict(method='Radau',rtol=rtol,atol=atol,max_step_s=max_step,output_step_s=output_step))
