"""Continuous assembly and posture change in one unchanged PhysX world."""
import time
import numpy as np
import pinocchio as pin
from amsrr_det.assembly import connected_view,forest_posture,dock_error,ready_to_attach,following_base_target
from amsrr_det.flight import FlightController
from amsrr_det.coupled_flight import CoupledFlightController
from amsrr_det.free_flight_trim import scheduled_support
from amsrr_det.grasp_task import smooth
from build.physx_seams import observe_connected,actuator_order


def audit_failure(audit):
    if audit.max_internal_support_impulse>1e-8:return 'internal_support_boundary_impulse'
    if audit.other_contacts:return 'unplanned_collision'
    if audit.clearance_violations:return 'clearance_violation'
    if audit.max_force>30.00001 or audit.max_moment>5.00001:return 'contact_load_limit'
    if audit.max_penetration>.002:return 'penetration_limit'
    if audit.max_joint_speed>3.00001 or audit.max_joint_limit_violation>1e-5:return 'joint_motion_limit'
    if audit.longest_gap>.05:return 'carried_pad_load_gap'
    if audit.max_material_displacement>.03:return 'contact_material_slip'
    if audit.min_clearance_carried<.1:return 'transport_clearance'


def assemble(task,plan,sim,seams,filters,audit,dt=.001,evidence_dir=None,coupled_morph=False,planned_morph_input=False):
    from collections import deque
    from pathlib import Path
    import json
    recent=deque(maxlen=500);sampled=[];failure_step=None
    p=plan.plant;views=[connected_view(p,plan.edges[:i],task.shape.payload()) for i in range(len(plan.edges)+1)]
    count=0;view=views[0];q,v=observe_connected(sim,view);flight=FlightController(view.pin,view.meta,q)
    morph_control_initialized=False
    order=actuator_order(view.meta,p.meta);ref=plan.parked.copy();startref=ref.copy()
    physical_data=p.pin.createData()
    phase='STAGE';origin=sim.time;stage_start=origin;started=time.perf_counter();dwell=0.;events=[]
    old_command=None;old_force=None;blend_time=None;reason='timeout';segment=0;max_seam_error={};control_times=[];loop_times=[];simulation_changes=[]
    filters.apply([0]);plan.filters.apply(audit.proximity_model,[0])
    command=np.array([q[a['q']] for a in view.meta['actuators']]);thrust=np.array(plan.component_trims[0]['thrust'])
    for step in range(65000):
        control_start=time.perf_counter()
        simulation_change_s=0.
        now=sim.time;elapsed=now-stage_start;fq,fv=sim.observation();q,v=observe_connected(sim,view)
        motion={}
        if phase in ['STAGE','APPROACH','VERIFY']:
            edge=plan.edges[count-1 if phase=='VERIFY' else count];k=next(b['q'] for b in p.meta['bases'] if b['module']==edge['b'])
            error=dock_error(p,fq,fv,edge,physical_data)
            if phase=='STAGE':
                goal=startref.copy();goal[k:k+7]=following_base_target(p,fq,edge,.15,physical_data)
                duration=max(8.,1.875*np.linalg.norm(plan.staging[count][:3]-startref[k:k+3])/.06)
                ref=pin.interpolate(p.pin,startref,goal,float(smooth(elapsed/duration)))
                good=elapsed>=duration and abs(error['axial_m']-.15)<.01 and error['transverse_m']<.01 and error['attitude_rad']<np.deg2rad(3) and error['linear_speed_m_s']<.05 and error['angular_speed_rad_s']<.1
                dwell=dwell+.01 if good else 0.
                if dwell>=.3:phase='APPROACH';stage_start=now;startref=ref.copy();dwell=0.
                elif elapsed>duration+20:reason='staging_timeout';break
            elif phase=='APPROACH':
                goal=startref.copy();goal[k:k+7]=following_base_target(p,fq,edge,data=physical_data)
                ref=pin.interpolate(p.pin,startref,goal,float(smooth(elapsed/16.)))
                good=elapsed>=16. and ready_to_attach(error) and abs(error['axial_m'])<.0002 and error['transverse_m']<.0002 and error['attitude_rad']<.002
                dwell=dwell+.01 if good else 0.
                if dwell>=.1:
                    before=fq.copy();change_start=time.perf_counter();seams[count].GetJointEnabledAttr().Set(True)
                    simulation_change_s+=time.perf_counter()-change_start
                    simulation_changes.append(dict(t=now-origin,action='enable_seam',wall_s=simulation_change_s))
                    events.append(dict(t=now-origin,edge=edge,error=error,instant_configuration_change=float(abs(sim.observation()[0]-before).max())))
                    old_command=command.copy();old_force=thrust.copy();blend_time=now
                    count+=1;phase='VERIFY';stage_start=now;dwell=0.;view=views[count];q,v=observe_connected(sim,view)
                    flight=FlightController(view.pin,view.meta,q);flight.command=command[actuator_order(p.meta,view.meta)].copy();flight.force=thrust.copy();order=actuator_order(view.meta,p.meta)
                elif elapsed>36.:reason='alignment_timeout';break
            elif phase=='VERIFY':
                dwell=dwell+.01 if ready_to_attach(error) else 0.
                if dwell>=1.25:
                    if count==len(plan.edges):phase='MORPH'
                    else:
                        phase='STAGE';change_start=time.perf_counter();filters.apply(range(count+1));plan.filters.apply(audit.proximity_model,range(count+1))
                        simulation_change_s+=time.perf_counter()-change_start
                        simulation_changes.append(dict(t=now-origin,action='collision_pair_filter',wall_s=simulation_change_s))
                    stage_start=now;startref=ref.copy();dwell=0.
                elif elapsed>10.:reason='attachment_drift';break
            target=forest_posture(view,p,ref,physical_data);trim=plan.component_trims[count]
            bias=np.array([j['effort_nm'] for j in trim['joints']]);forces=np.array(trim['thrust'])
        else:
            if coupled_morph and not morph_control_initialized:
                previous=flight
                flight=CoupledFlightController(view.pin,view.meta,q,track_vectoring=True,base_residual_feedback=True,planned_input_reference=planned_morph_input)
                flight.command=previous.command.copy();flight.force=previous.force.copy()
                flight.control_count=1  # These inputs have already been issued.
                morph_control_initialized=True
            first,last=plan.morph_path[segment:segment+2];difference=pin.difference(view.pin,first,last)
            docks=np.array([a['v'] for a in view.meta['actuators'] if a['role']=='dock']);gimbal_v=np.array([a['v'] for a in view.meta['actuators'] if a['role']=='vectoring'])
            duration=max(.5,1.875*np.linalg.norm(difference[:3])/.08,1.875*np.linalg.norm(difference[3:6])/.12,1.875*abs(difference[docks]).max()/.1,1.875*abs(difference[gimbal_v]).max()/.3)
            s=float(smooth(elapsed/duration));target=pin.interpolate(view.pin,first,last,s)
            if coupled_morph:
                from amsrr_det.numerics import quintic_reference
                target,tv,ta,s=quintic_reference(view.pin,view.meta,first,last,elapsed+.01,duration)
                motion=dict(target_velocity=tv,target_acceleration=ta)
            bias,forces=scheduled_support(plan.morph_schedules[segment],s)
            error=pin.difference(view.pin,q,last)
            good=elapsed>=duration and np.linalg.norm(error[:6])<.008 and abs(error[docks]).max()<.012 and np.linalg.norm(v[:6])<.02 and abs(v[docks]).max()<.03
            dwell=dwell+.01 if good else 0.
            if dwell>=.2:
                events.append(dict(t=now-origin,morph_segment=segment,duration=elapsed));segment+=1;stage_start=now;dwell=0.
                if segment==len(plan.morph_path)-1:reason='complete';break
            elif elapsed>duration+15.:reason='morph_tracking_timeout';break
        gimbals=np.array([view.meta['actuators'][r['gimbal']]['q'] for r in view.meta['rotors']])
        cmd,thrust=flight.update(q,v,target,joint_bias=bias,thrust_reference=forces,gimbal_reference=target[gimbals],**motion);command=cmd[order]
        if flight.allocation_failure:reason=flight.allocation_failure;break
        if blend_time is not None:
            s=smooth((now-blend_time)/.25);command=(1-s)*old_command+s*command;thrust=(1-s)*old_force+s*thrust
        if evidence_dir is not None:
            recent.append(dict(t=now-origin,phase=phase,segment=segment,attached=count,
                configuration=fq.tolist(),velocity=fv.tolist(),connected_configuration=q.tolist(),
                connected_velocity=v.tolist(),target=target.tolist(),command=command.tolist(),thrust=thrust.tolist(),
                coupled_diagnostics=getattr(flight,'coupled_diagnostics',None)))
            if step%10==0:sampled.append((now-origin,count,fq.copy(),fv.copy()))
        total=time.perf_counter()-control_start;loop_times.append(total);control_times.append(total-simulation_change_s)
        for _ in range(round(.01/dt)):
            if evidence_dir is not None:before_q,before_v=fq.copy(),fv.copy()
            sim.step(command,thrust,dt);fq,fv=sim.observation();audit.update(fq,fv,command,dt,grasp_contact_allowed=False)
            reason=audit_failure(audit) or 'timeout'
            if reason!='timeout':
                if evidence_dir is not None:failure_step=dict(before_q=before_q,before_v=before_v,q=fq,v=fv,command=command,thrust=thrust)
                break
        if step%10==0:
            for edge in plan.edges[:count]:
                err=dock_error(p,fq,fv,edge,physical_data)
                for key,value in err.items():max_seam_error[key]=max(max_seam_error.get(key,0.),abs(value))
                if not ready_to_attach(err) and reason=='timeout':reason='attached_seam_drift'
        if step%1000==0:print('ASSEMBLY',dict(t=sim.time-origin,phase=phase,attached=count,segment=segment,dock_error=error if phase!='MORPH' else None,wall_s=time.perf_counter()-started),flush=True)
        if reason!='timeout':break
    if evidence_dir is not None:
        directory=Path(evidence_dir)
        print('ASSEMBLY_END',dict(reason=reason,phase=phase,segment=segment,t=sim.time-origin),flush=True)
        np.savez_compressed(directory/'assembly_end_state.npz',q=fq,v=fv,connected_q=q,connected_v=v,command=command,thrust=thrust)
        if sampled:
            times,counts,positions,velocities=zip(*sampled)
            np.savez_compressed(directory/'assembly_trace.npz',t=times,attached=counts,q=positions,v=velocities)
        if failure_step is not None:np.savez_compressed(directory/'assembly_failure_step.npz',**failure_step)
        (directory/'assembly_last_controls.json').write_text(json.dumps(list(recent)))
    return dict(success=reason=='complete',reason=reason,phase=phase,simulation_s=sim.time-origin,wall_s=time.perf_counter()-started,events=events,sampled_seam_error_max=max_seam_error,planning=plan.evidence,
                simulation_change_times=simulation_changes,loop_time_including_simulator_changes_s=dict(mean=float(np.mean(loop_times)),p95=float(np.quantile(loop_times,.95)),maximum=max(loop_times)),
                control_time_s=dict(mean=float(np.mean(control_times)),p95=float(np.quantile(control_times,.95)),maximum=max(control_times))),command,thrust
