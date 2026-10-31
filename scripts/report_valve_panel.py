"""Check saved valve endpoints, measured latency and predeclared strata."""
import argparse,collections,hashlib,json,math
from pathlib import Path
import numpy as np
import mujoco
from build.contact_step_control import pad_gap_jacobians
from scripts.report_grasp_panel import failure_concentration

from build.control_latency import sampling_intervals


def structure_breadth(rows,rule):
    """Only physical execution counts; a selected but unrun plan does not."""
    executed=[r for r in rows if r['executed']]
    counts={str(n):dict(executed=sum(r['modules']==n for r in executed),
        successes=sum(r['modules']==n and r['success'] for r in executed)) for n in range(1,9)}
    distinct=sum(v['executed']>0 for v in counts.values())
    bands=[dict(modules=[lo,hi],executed=sum(lo<=r['modules']<=hi for r in executed),
        successes=sum(lo<=r['modules']<=hi and r['success'] for r in executed)) for lo,hi in rule.get('module_bands',[])]
    return dict(counts=counts,bands=bands,distinct_executed=distinct,
        distinct_successful=sum(v['successes']>0 for v in counts.values()),
        passes=distinct>=rule.get('minimum_distinct_executed_module_counts',0) and
            all(v['successes']>=rule.get('minimum_successes_in_each_module_band',0) for v in bands))


def summarize(root):
    cases=json.loads((root/'cases.json').read_text());issues=[];rows=[];times=[];ages=[];planning=[];sampling=[]
    from amsrr_det.runtime_manifest import physical_library
    saved_environment=json.loads((root/'environment.json').read_text())
    if 'physical_library' in saved_environment and saved_environment['physical_library']['sha256']!=physical_library()['sha256']:
        issues.append(['different_physical_library_for_audit'])
    groups=collections.defaultdict(lambda:collections.defaultdict(list))
    for name,digest in json.loads((root/'sources.json').read_text()).items():
        path=(root/'source_snapshot'/name) if (root/'source_snapshot').exists() else Path(name)
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:issues.append(['source_changed',name])
    for case in cases:
        name=case['id'];folder=root/name
        result=json.loads((folder/'result.json').read_text()) if (folder/'result.json').exists() else dict(task_success=False,reason='missing_result')
        if result['reason']=='missing_result':issues.append(['missing_result',name])
        selection_path=folder/'selection.json'
        if selection_path.exists():
            from amsrr_det.valve import ValveSpec
            selection=json.loads(selection_path.read_text())
            expected_request=ValveSpec(**case['request']).__dict__
            if selection.get('seed')!=case['seed']:issues.append(['seed_changed',name])
            if json.dumps(selection['request'],sort_keys=True)!=json.dumps(expected_request,sort_keys=True):issues.append(['planning_input_changed',name])
            if result['task_success'] and (selection['status']!='accepted' or not selection['evidence']['complete']):issues.append(['incomplete_selection_evidence',name])
        elif result['task_success']:issues.append(['missing_selection',name])
        row=dict(id=name,success=result['task_success'],reason=result['reason'],modules=result.get('modules'),contacts=result.get('contacts'),executed='checks' in result)
        if 'checks' in result:
            request=json.loads((folder/'request.json').read_text())
            from amsrr_det.valve import ValveSpec
            expected_request=ValveSpec(**case['request']).__dict__
            if json.dumps(request,sort_keys=True)!=json.dumps(expected_request,sort_keys=True):issues.append(['input_changed',name])
            m=mujoco.MjModel.from_xml_path(str(folder/'physical.xml'));d=mujoco.MjData(m)
            saved=np.load(folder/'final.npz');d.qpos[:]=saved['q'];d.qvel[:]=saved['v'];mujoco.mj_fwdPosition(m,d)
            joint=m.joint('object').id;oi=m.jnt_qposadr[joint];ov=m.jnt_dofadr[joint]
            plan=json.loads((folder/'selection.json').read_text())['plan']
            links=[f'm{n}/'+('yaw_dock_mech' if port.startswith('yaw') else 'pitch_dock_mech')+port[-1] for n,port in plan['surfaces']]
            gaps,_=pad_gap_jacobians(m,d,links);goal=saved['retreat_goal']
            angle_error=abs(d.qpos[oi]-request['target_angle'])
            retreat=bool(len(goal) and np.linalg.norm(d.qpos[:3]-goal[:3])<.05 and
                2*math.acos(min(1.,abs(float(d.qpos[3:7]@goal[3:7]))))<.2 and
                np.linalg.norm(d.qvel[:3])<.1 and np.linalg.norm(d.qvel[3:6])<.2)
            settle=result['audit']['time_since_gripper_contact_s']
            expected=dict(angle=bool(angle_error<np.deg2rad(5)),speed=bool(abs(d.qvel[ov])<.02),
                released=bool(len(gaps) and min(gaps)>.02),settled=bool(settle is not None and settle>=2.),retreat_flight=retreat)
            if expected!=result['checks']:issues.append(['endpoint_check_mismatch',name,expected,result['checks']])
            if result['task_success'] and (not all(expected.values()) or result['audit']['peak_effort_ratio']>1.000001):issues.append(['invalid_success',name])
            calls=json.loads((folder/'control_calls.json').read_text());values=[c['publication_cpu_s'] for c in calls];times.extend(values)
            intervals,sampling_late=sampling_intervals(calls);sampling.extend(intervals)
            if abs(max(values)*1000-result['control']['max_ms'])>1e-9:issues.append(['timing_summary',name])
            for c in calls:
                if c['application_s'] is not None and c['application_s']+1e-8<c['observation_s']+c['calculation_s']:issues.append(['publisher_applied_early',name]);break
            by_tick={round(c['observation_s'],7):c for c in calls}
            allocations=json.loads((folder/'allocation_calls.json').read_text())
            for a in allocations:
                if abs(a['available_s']-a['observation_s']-a['calculation_s'])>1e-8:issues.append(['worker_delay_mapping',name]);break
                application=a.get('application_s')
                if application is None and 'application_tick_s' in a:
                    tick=by_tick.get(round(a['application_tick_s'],7));application=None if tick is None else tick['application_s']
                if application is not None:
                    if application+1e-8<a['available_s']:issues.append(['worker_applied_early',name]);break
                    ages.append(application-a['observation_s'])
            row.update(angle_error_deg=float(np.rad2deg(angle_error)),maximum_publication_ms=max(values)*1000,
                failed_checks=[k for k,v in expected.items() if not v])
            if max(values)>=.01:row.update(success=False,reason='publication_deadline_exceeded')
            if sampling_late:row.update(success=False,reason='sampling_deadline_exceeded')
        if 'planning_s' in result:
            planning.append(result['planning_s']);row['planning_s']=result['planning_s']
            if result['planning_s']>15.:row.update(success=False,reason='planning_timeout')
        rows.append(row)
        strata=dict(axis=case['axis_group'],torque=case['torque'],initial_angle=case['initial_angle_group'],
                    axis_torque=str((case['axis_group'],case['torque'])))
        if 'load_band' in case:strata['load_band']=case['load_band']
        for axis,value in strata.items():groups[axis][str(value)].append(row)
    def score(rs):return dict(successes=sum(r['success'] for r in rs),total=len(rs),fraction=sum(r['success'] for r in rs)/len(rs))
    def timing(values):
        x=np.asarray(values)*1000
        return dict(samples=len(x),mean_ms=float(x.mean()),p99_ms=float(np.quantile(x,.99)),max_ms=float(x.max()),over10=int(sum(x>=10))) if len(x) else None
    strata={axis:{value:score(rs) for value,rs in values.items()} for axis,values in groups.items()};overall=score(rows)
    report=dict(overall=overall,issues=issues,rows=rows,strata=strata,
        failure_concentration=failure_concentration(strata,overall),publisher=timing(times),observation_to_application=timing(ages),
        sampling=dict(intervals=len(sampling),maximum_ms=float(max(sampling)*1000),
                      late_intervals=int(sum(t>.01+1e-8 for t in sampling))) if sampling else None,
        planning_s=dict(mean=float(np.mean(planning)),max=max(planning)) if planning else None,
        scope='Fresh structure generation and measured-delay execution; final endpoint recomputed from saved physical state')
    contract=json.loads((root/'panel.json').read_text()).get('acceptance',{}) if (root/'panel.json').exists() else {}
    report['structure_breadth']=structure_breadth(rows,contract.get('breadth_engineering_rule',{}))
    report['acceptance_contract']=contract
    if contract:
        report['quantitative_requirements_pass']=bool(not issues and overall['fraction']>=contract['minimum_success_fraction'] and
            report['structure_breadth']['passes'] and not report['failure_concentration']['clear_concentration'])
        report['mechanical_failure_review_required']=True
    (root/'audit_summary.json').write_text(json.dumps(report,indent=2));return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path)
    report=summarize(parser.parse_args().root)
    print(json.dumps({k:report[k] for k in ['overall','issues','publisher','planning_s','failure_concentration']},indent=2))
