"""Independent saved-motion and delay audit for a fixed walking panel."""
import argparse,collections,hashlib,json
from pathlib import Path
import numpy as np
import mujoco
from amsrr_det.walking import WalkSpec
from amsrr_det.runtime_manifest import physical_library
from scripts.report_grasp_panel import failure_concentration


def energy_summary(rows):
    successful=[r.get('rotor_burden_hover_ratio') for r in rows if r['success']]
    valid=bool(successful) and all(v is not None and np.isfinite(v) and v>=0. for v in successful)
    return dict(successful_cases=len(successful),valid=valid,
                mean=float(np.mean(successful)) if valid else None,
                maximum=float(max(successful)) if valid else None,
                basis='arithmetic mean of successful task ratios; failed tasks remain in success and bias evaluation; uncalibrated rotor proxy')


def summarize(root):
    root=Path(root);panel=json.loads((root/'panel.json').read_text());issues=[];rows=[]
    groups=collections.defaultdict(lambda:collections.defaultdict(list));publication=[];ages=[];intervals=[]
    snapshot=json.loads((root/'sources.json').read_text())
    for name,digest in snapshot.items():
        file=root/'source_snapshot'/name
        if not file.exists() or hashlib.sha256(file.read_bytes()).hexdigest()!=digest:issues.append(['invalid_snapshot',name])
    env=json.loads((root/'environment.json').read_text())
    if env['physical_library']['sha256']!=physical_library()['sha256']:issues.append(['different_physical_library'])
    for case in panel['cases']:
        folder=root/case['id'];path=folder/'result.json'
        result=json.loads(path.read_text()) if path.exists() else dict(task_success=False,reason='missing_result')
        if not path.exists():issues.append(['missing_result',case['id']])
        row=dict(id=case['id'],success=result['task_success'],reason=result.get('reason'),modules=case['task']['modules'],
                 feet=result.get('feet'),planning_s=result.get('planning_s'),failed_checks=[k for k,v in result.get('checks',{}).items() if not v])
        selection=folder/'selection.json'
        if selection.exists():
            s=json.loads(selection.read_text())
            if s['request']!=case['task'] or s['seed']!=case['seed']:issues.append(['changed_input',case['id']])
            if row['success'] and (s['status']!='accepted' or not s['evidence']['complete']):issues.append(['incomplete_selection',case['id']])
        elif row['success']:issues.append(['missing_selection',case['id']])
        if 'checks' in result:
            if not result['fresh_structure_generation']:issues.append(['not_fresh_generation',case['id']])
            calls=json.loads((folder/'calls.json').read_text());alloc=json.loads((folder/'allocations.json').read_text())
            cpu=np.array([r['publication_cpu_s'] for r in calls]);dt=np.diff([r['observation_s'] for r in calls])
            publication.extend(cpu);intervals.extend(dt)
            if abs(cpu.max()*1000-result['control']['max_ms'])>1e-8:issues.append(['timing_summary',case['id']])
            for r in calls:
                if r['application_s'] is not None and r['application_s']+1e-8<r['observation_s']+r['calculation_s']:
                    issues.append(['publication_applied_early',case['id']]);break
            for r in alloc:
                if abs(r['available_s']-r['observation_s']-r['calculation_s'])>1e-8:issues.append(['invalid_delay_mapping',case['id']]);break
                if r.get('application_s') is not None:
                    if r['application_s']+1e-8<r['available_s']:issues.append(['allocation_applied_early',case['id']]);break
                    ages.append(r['application_s']-r['observation_s'])
            # Independently recover task path/speed from recorded COM and final CAD model.
            samples=json.loads((folder/'audit.json').read_text());task=WalkSpec(**case['task'])
            origin=np.array(s['plan']['gait']['origin']);times=np.array([r['time_s'] for r in samples]);xy=np.array([r['com'][:2] for r in samples])
            relative=xy-origin[:2];k=task.curvature
            progress=relative[:,0] if abs(k)<1e-10 else np.unwrap(np.arctan2(k*relative[:,0],1-k*relative[:,1]))/k
            travel=progress[-1]-progress[0];mean=travel/(times[-1]-times[0])
            boundaries=np.r_[np.arange(0,times[-1],5.),times[-1]]
            windows=np.diff(np.interp(boundaries,times,progress))/np.diff(boundaries)
            error=np.linalg.norm(xy-origin[:2]-task.path(times)[0],axis=1)
            m=mujoco.MjModel.from_xml_path(str(folder/'physical.xml'));d=mujoco.MjData(m);final=np.load(folder/'final.npz');d.qpos[:]=final['q'];d.qvel[:]=final['v'];mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
            if np.linalg.norm(d.subtree_com[0]-np.array(samples[-1]['com']))>1e-8:issues.append(['final_com_mismatch',case['id']])
            independent=dict(distance=bool(travel>=task.distance),minimum_speed=bool(travel/result['simulation_s']>=.2),
                mean_speed=bool(abs(mean/task.speed-1)<=.1),window_speed=bool(max(abs(windows/task.speed-1))<=.2),
                path=bool(error.max()<=.1 and error[-1]<=.05),control_deadline=bool(cpu.max()<.01 and max(dt)<=.01+1e-8),
                planning_deadline=bool(result['planning_s']<=15.))
            if any(result['checks'][key]!=value for key,value in independent.items()):issues.append(['independent_check_mismatch',case['id'],independent])
            if row['success'] and (not all(independent.values()) or not all(result['checks'].values()) or result['peak_actuator_ratio']>1.00001):issues.append(['invalid_success',case['id']])
            row.update(maximum_path_error_m=float(error.max()),ground_contact_fraction=result['ground_contact_fraction'],ground_load_fraction=result['ground_load_fraction'],
                rotor_burden_mean=result['rotor_burden_mean'],joint_burden_mean=result['joint_burden_mean'])
            hover=(m.body_mass.sum()*abs(m.opt.gravity[2]))**1.5/np.sqrt(4*task.modules)
            row['rotor_burden_hover_ratio']=result['rotor_burden_mean']/hover
            if abs(row['rotor_burden_hover_ratio']-result['rotor_burden_hover_ratio'])>1e-8:
                issues.append(['energy_normalization_mismatch',case['id']])
        rows.append(row)
        t=case['task'];strata=dict(modules=t['modules'],speed=t['speed'],path='straight' if t['curvature']==0 else 'left' if t['curvature']>0 else 'right',module_speed=str((t['modules'],t['speed'])))
        for group,key in strata.items():groups[group][str(key)].append(row)
    def score(rs):return dict(successes=sum(r['success'] for r in rs),total=len(rs),fraction=sum(r['success'] for r in rs)/len(rs))
    def timing(values):
        a=np.array(values)*1000
        return dict(samples=len(a),mean_ms=float(a.mean()),p99_ms=float(np.quantile(a,.99)),max_ms=float(a.max()),over10=int(sum(a>=10))) if len(a) else None
    overall=score(rows);strata={name:{key:score(rs) for key,rs in g.items()} for name,g in groups.items()}
    concentration=failure_concentration(strata,overall)
    energy=energy_summary(rows)
    energy['by_modules']={key:energy_summary(rs) for key,rs in groups['modules'].items()}
    energy['by_speed']={key:energy_summary(rs) for key,rs in groups['speed'].items()}
    target=panel['acceptance'].get('maximum_mean_rotor_hover_proxy')
    energy['target']=target
    energy_pass=target is None or (energy['valid'] and energy['mean']<target)
    planning=[r['planning_s'] for r in rows if r['planning_s'] is not None]
    summary=json.loads((root/'summary.json').read_text())
    if summary['source_changes']:issues.append(['runtime_changed_during_panel'])
    report=dict(overall=overall,issues=issues,rows=rows,strata=strata,failure_concentration=concentration,energy=energy,
        publisher=timing(publication),observation_to_application=timing(ages),planning_s=dict(max=max(planning),mean=float(np.mean(planning))),
        sampling=dict(count=len(intervals),over10=int(sum(t>.01+1e-8 for t in intervals)),maximum_ms=max(intervals)*1000) if intervals else None,
        quantitative_requirements_pass=bool(not issues and overall['fraction']>=.8 and not concentration['clear_concentration'] and energy_pass),
        mechanical_failure_review_required=True)
    (root/'audit_summary.json').write_text(json.dumps(report,indent=2));return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');r=summarize(p.parse_args().root)
    print(json.dumps({k:r[k] for k in ['overall','issues','publisher','planning_s','failure_concentration']},indent=2))
