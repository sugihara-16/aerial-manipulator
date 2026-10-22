"""Report every frozen input, with missing cases and subgroup failures visible."""
import argparse
import hashlib
from collections import Counter,defaultdict
import json
from pathlib import Path
import yaml
from scipy.stats import fisher_exact


def failure_concentration(strata,overall,alpha=.05):
    """Predeclared groups versus their complements; Fisher + Holm correction.

    No rejection is not evidence of equal failure rates. Mechanical failure
    causes still require review; a tiny subgroup below 80% is not by itself
    a proven concentration. Sources: SciPy fisher_exact; R stats p.adjust.
    """
    rows=[];failures=overall['total']-overall['successes']
    for axis,groups in strata.items():
        for name,group in groups.items():
            n=group['total'];other=overall['total']-n
            if not n or not other:continue
            failed=n-group['successes'];rest=failures-failed
            probability=float(fisher_exact([[failed,n-failed],[rest,other-rest]],alternative='greater').pvalue)
            rows.append(dict(axis=axis,group=name,total=n,failures=failed,
                failure_fraction=failed/n,other_failure_fraction=rest/other,p_value=probability))
    previous=0.;count=len(rows)
    for rank,row in enumerate(sorted(rows,key=lambda r:r['p_value'])):
        previous=max(previous,min(1.,(count-rank)*row['p_value']))
        row['adjusted_p_value']=previous;row['clear_concentration']=previous<=alpha
    return dict(method='one-sided Fisher exact; Holm across all predeclared group comparisons',
                alpha=alpha,clear_concentration=any(r['clear_concentration'] for r in rows),
                equal_failure_rates_certified=False,mechanical_cause_review_required=True,comparisons=rows)


def report(panel, roots, config):
    if isinstance(panel,list):panel=dict(role='unspecified',cases=panel)
    results={}
    expected={case['id'] for case in panel['cases']}
    if len(expected)!=len(panel['cases']):raise ValueError('duplicate input IDs')
    for root in roots:
        for path in Path(root).glob('*/result.json'):
            case=path.parent.name
            if case not in expected:raise ValueError('result is outside the frozen inputs: '+case)
            if case in results:raise ValueError('duplicate result: '+case)
            results[case]=(path,json.loads(path.read_text()))
    rows=[];groups=defaultdict(lambda:defaultdict(list));sources=[];settings=[]
    for case in panel['cases']:
        path,result=results.get(case['id'],(None,{}))
        if result:
            request=result.get('request')
            if request is None:request=json.loads((path.parent/'request.json').read_text())
            for key in ['source','destination','support_top']:
                if request[key]!=case[key]:raise ValueError('input mismatch: '+case['id']+'/'+key)
            for key in ['shape','size','mass']:
                if request['payload'][key]!=case['payload'][key]:raise ValueError('payload mismatch: '+case['id']+'/'+key)
        reported_pass=bool(result.get('task_success',False))
        full_force_audit=result.get('audit',{}).get('contact_force_components')=='normal_and_friction'
        complete_clearance=result.get('audit',{}).get('clearance_detection')=='refitted_body_bounds'
        complete_boundary=result.get('audit',{}).get('support_boundary_force_components')=='normal_and_friction_conservative_anchor_geometry'
        if result.get('audit',{}).get('engine')=='MuJoCo':
            complete_boundary=result['audit'].get('support_representation')=='native_unsplit'
        passed=reported_pass and full_force_audit and complete_clearance and complete_boundary
        reason=result.get('reason','missing_result')
        if reported_pass and not complete_boundary:reason='incomplete_support_boundary_audit'
        if reported_pass and not complete_clearance:reason='incomplete_clearance_audit'
        if reported_pass and not full_force_audit:reason='incomplete_contact_force_audit'
        row=dict(case=case['id'],success=passed,reported_task_success=reported_pass,
                 reason=reason,
                 scope=result.get('scope'),execution_entrypoint=result.get('execution_entrypoint'),result_path=None if path is None else str(path))
        rows.append(row)
        for key in config['required_strata']:groups[key][str(case['strata'][key])].append(row)
        if path:
            # Identical source code can still select different experimental
            # controllers. A panel must not combine case-specific options.
            execution_settings={key:result.get(key) for key in [
                'release_method','async_release','guard_servo','preserve_joint_action',
                'support_representation','collision_system','coupled_morph',
                'planned_morph_input','posture_tracking','preload_relief',
                'trajectory_feedforward','normal_moment_refinement','rest_to_rest_retreat','coupled_free_flight',
                'candidate_limit','search_seconds','relative_acquisition',
                'paired_load_transfer','dynamic_unwind','atomic_paired_start','dt','position_iterations','velocity_iterations',
                'minimum_preload_m','acquisition_gap_m','preserve_preload']}
            execution_settings.update({key:result.get(key) for key in [
                'engine','integrator','solver','solver_iterations','solver_tolerance','assembly_executed','execution_configuration_sha256']})
            contact_engine=result.get('contact_engine',{})
            execution_settings.update({'contact_engine_'+key:contact_engine.get(key) for key in [
                'sha256','patch_sha256','ccd_tolerance','ccd_iterations']})
            if result.get('scope') not in ['fresh_task_generation','assembly_planning']:
                settings.append(execution_settings)
            manifest=json.loads((path.parent/'source_snapshot/manifest.json').read_text())
            runtime={name:digest for name,digest in manifest.items() if name.startswith(('amsrr_det/','build/','configs/','native/','assets/generated/')) or name in ['scripts/run_grasp_task.py','CMakeLists.txt']}
            for runner in ['run_physx_task.py','run_physx_assembly_task.py','run_mujoco_task.py']:
                copied=path.parent/runner
                if copied.exists():runtime['build/'+runner]=hashlib.sha256(copied.read_bytes()).hexdigest()
            sources.append(runtime)
    def score(members):
        successes=sum(r['success'] for r in members)
        return dict(successes=successes,total=len(members),fraction=successes/len(members),
                    reaches_threshold=successes>=config['minimum_success_fraction']*len(members),
                    failures=dict(Counter(r['reason'] for r in members if not r['success'])))
    strata={key:{name:score(members) for name,members in values.items()} for key,values in groups.items()}
    names=set().union(*(set(s) for s in sources))
    changed={name:sorted({s.get(name) for s in sources},key=str) for name in names if len({s.get(name) for s in sources})>1}
    setting_changes={key:sorted({s[key] for s in settings},key=str)
                     for key in (settings[0] if settings else {})
                     if len({s[key] for s in settings})>1}
    overall=score(rows)
    concentration=failure_concentration(strata,overall)
    rates=overall['reaches_threshold'] and all(s['reaches_threshold'] for values in strata.values() for s in values.values())
    # The latest user allows spawning the generated complete structure.
    # Saved-candidate replays still do not establish fresh task generation.
    accepted_scopes={'fresh_generation_continuous_assembly_and_task'}
    if not config.get('assembly_required',True):accepted_scopes.add('fresh_generation_spawn_task')
    whole_sequence=all(r['scope'] in accepted_scopes or
                       (not r['success'] and r['execution_entrypoint']=='fresh_task_input') for r in rows)
    return dict(role=panel['role'],overall=overall,strata=strata,missing=len(expected-results.keys()),
                runtime_source_changes=changed,all_results_same_runtime_sources=not changed,
                execution_setting_changes=setting_changes,all_results_same_execution_settings=not setting_changes,
                scope_counts=dict(Counter(r['scope'] for r in rows)),
                reaches_all_rate_thresholds=rates,
                failure_concentration=concentration,
                all_cases_cover_required_sequence=whole_sequence,
                assembly_required=config.get('assembly_required',True),
                ready_for_completion_review=bool(panel['role']=='heldout' and len(results)==len(expected)
                    and overall['reaches_threshold'] and not concentration['clear_concentration']
                    and whole_sequence and not changed and not setting_changes),
                rows=rows)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--results',type=Path,nargs='+',required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();config=yaml.safe_load(Path('configs/evaluation/grasp_transport.yaml').read_text())
    result=report(json.loads(args.input.read_text()),args.results,config)
    args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:result[k] for k in ['role','overall','missing','all_results_same_runtime_sources','reaches_all_rate_thresholds']}))
