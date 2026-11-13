"""Compile the shared task requirements into the adopted task engines."""
import hashlib
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
from .task_spec import validate_spec,requirements


def write(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=lambda v:v.tolist() if hasattr(v,'tolist') else str(v)))
    tmp.replace(path)


def compile_request(spec):
    import numpy as np
    import pinocchio as pin
    r=requirements(spec);s=r['scene'];g=r['motion'];kind=spec['task']
    if kind=='grasp':
        from .objects import ObjectShape
        o=s['object'];shape=ObjectShape(o['shape'],o['size_m'],o['mass_kg'])
        q=pin.Quaternion(pin.rpy.rpyToMatrix(*o['initial']['rpy_rad'])).coeffs()
        source=np.r_[o['initial']['xy_m'],0.,q];source[2]=o['support_height_m']-shape.lowest_z(source)
        dest=source.copy();dest[:3]+=g['translation_m']
        request=dict(payload=shape.payload(),source=source.tolist(),destination=dest.tolist(),support_top=o['support_height_m'],timestep=.000125)
        r['derived']=dict(gravity_force_n=[0.,0.,-o['mass_kg']*9.80665],initial_object_pose_xyzw=source.tolist())
    elif kind=='valve':
        from .valve import ValveSpec
        o=s['object'];t=o['axis']['tilt_rad'];a=o['axis']['azimuth_rad'];axis=[np.sin(t)*np.cos(a),np.sin(t)*np.sin(a),np.cos(t)]
        request=ValveSpec(center=tuple(o['center_m']),axis=tuple(axis),initial_angle=o['initial_angle_rad'],target_angle=g['change_rad'],torque=o['resistance_torque_nm']).__dict__
        r['derived']=dict(axis_world=axis,resisting_torque_world_nm=(np.array(axis)*o['resistance_torque_nm']).tolist())
    else:
        request=dict(modules=r['resources']['module_count']['fixed'],speed=g['speed_m_s'],distance=g['distance_m'],curvature=g['curvature_m_inv'])
        r['derived']=dict(lateral_acceleration_m_s2=g['speed_m_s']**2*g['curvature_m_inv'],nominal_duration_s=g['distance_m']/g['speed_m_s'])
    return r,request


def execute(spec,folder):
    began=time.perf_counter()
    from .task_live import LiveState
    spec=validate_spec(spec);out=Path(folder);out.mkdir(parents=True,exist_ok=True)
    write(out/'task.json',spec);write(out/'status.json',dict(state='planning',message='Generating structure'))
    req,request=compile_request(spec);write(out/'requirements.json',req);write(out/'engine_request.json',request)
    kind=spec['task'];mode=spec['execution']['mode'];seed=spec['execution']['seed']
    planning_budget=spec['execution']['planning_budget_s']
    allowed=sorted(os.sched_getaffinity(0));control=allowed[0];worker=allowed[2] if len(allowed)>2 else allowed[-1]
    live=LiveState(out,preparation_started=began);run=out/'execution';plan_only=mode=='assembly'
    from .task_sequence import Sequence
    integration=Sequence(out,seed,live,worker) if mode=='integrated' else None
    try:
        if kind=='walking':
            from .walking import WalkSpec
            from scripts.run_walking_task import run as walk
            raw=walk(WalkSpec(**request),run,seed=seed,workers=min(24,len(allowed)),worker_cpu=worker,control_cpu=control,observer=None if plan_only else live,plan_only=plan_only,integration=integration,planning_budget_s=planning_budget)
        elif kind=='valve':
            from scripts.run_valve_task import run as valve
            args=SimpleNamespace(output=run,request=out/'engine_request.json',plan=None,seed=seed,workers=min(32,len(allowed)),planning_cpus=allowed,worker_cpu=worker)
            raw=valve(args,observer=None if plan_only else live,plan_only=plan_only,integration=integration)
        else:
            from build.check_relative_task import main
            raw=main([str(out/'engine_request.json'),str(run),'--seed',str(seed),'--structure-workers',str(min(32,len(allowed))),'--worker-cpu',str(worker)],observer=None if plan_only else live,plan_only=plan_only,integration=integration)
        result_path=run/'result.json'
        if raw is None and result_path.exists():raw=json.loads(result_path.read_text())
        if plan_only and raw and raw.get('reason')=='structure_generated':
            source=run/('generated_plan/plan.json' if kind=='grasp' else 'selection.json');record=json.loads(source.read_text())
            model=record['model'] if kind=='grasp' else record['plan']
            case=dict(id='gui/'+out.name,task=kind,modules=model['modules'],edges=model['edges'],initial_seed=seed,source=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest())
            write(out/'assembly_input.json',case);write(out/'status.json',dict(state='assembling',message='Executing takeoff, assembly, and stable hovering'))
            from scripts.run_assembly_task import run as assembly
            os.sched_setaffinity(0,{control});raw=assembly(case,out/'assembly',worker_cpu=worker,observer=live)
        if raw is None:raise RuntimeError('The execution engine did not return a result')
        timing=dict(raw.get('control',{}))
        if mode=='assembly':timing=dict(max_ms=1000*raw.get('publication_max_s',0.),over10=raw.get('publication_over_10ms',0))
        if 'maximum_ms' in timing:timing['max_ms']=timing['maximum_ms']
        if 'misses' in timing:timing['over10']=timing['misses']
        misses=timing.get('over10',0)
        physical=bool(raw.get('task_success',raw.get('success',False)))
        ready=None if live.ready_wall is None else live.ready_wall-began
        # Both the existing engine result and the common timing contract matter.
        success=physical and misses==0 and (ready is not None and ready<=planning_budget)
        reason=raw.get('reason') or ('completed' if physical else 'terminal_conditions_failed')
        if physical and misses:reason='control_deadline'
        if physical and ready is not None and ready>planning_budget:reason='planning_deadline'
        if integration is not None and integration.result is not None:
            success=success and bool(integration.result and integration.result['success'])
            if not integration.result or not integration.result['success']:reason='assembly_stage_failed'
            stages=[timing,dict(max_ms=1000*integration.result.get('publication_max_s',0.),over10=integration.result.get('publication_over_10ms',0)),dict(max_ms=1000*(integration.transfer_result or {}).get('publication_max_s',0.),over10=(integration.transfer_result or {}).get('publication_over_10ms',0))]
            timing=dict(max_ms=max(x.get('max_ms',0.) for x in stages),over10=sum(x.get('over10',0) for x in stages),stages=dict(zip(['task','assembly','transfer'],stages)))
            hold=[row['calculation_s'] for row in integration.hold_publications]
            timing['stages']['transition_hold']=dict(calls=len(hold),max_ms=max(hold,default=0.)*1000,over10=sum(t>=.01 for t in hold))
            timing['max_ms']=max(timing['max_ms'],max(hold,default=0.)*1000);timing['over10']+=sum(t>=.01 for t in hold)
            if timing['over10']:success=False;reason='control_deadline'
            if not physical and integration.transfer_result and integration.transfer_result['position_error_m']>.1:reason='transfer_target_not_reached'
        result=dict(success=success,task_engine_success=physical,reason=reason,task=kind,mode=mode,modules=raw.get('modules'),preparation_s=ready if ready is not None else time.perf_counter()-began,execution_started=ready is not None,wall_s=time.perf_counter()-began,simulation_s=(float(integration.data.time) if integration is not None and hasattr(integration,'data') else raw.get('whole_sequence',{}).get('simulation_s',raw.get('simulation_s'))),control=timing,engine_result=raw,assembly=None if integration is None else integration.result,transfer=None if integration is None else integration.transfer_result)
        write(out/'result.json',result);write(out/'status.json',dict(state='finished',message='Succeeded' if success else 'Failed',result=result))
        return result
    finally:live.close()
