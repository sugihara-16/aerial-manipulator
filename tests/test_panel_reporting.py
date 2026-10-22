import json
import pytest
from scripts.report_grasp_panel import report,failure_concentration


def test_small_subgroup_rate_is_not_automatically_a_clear_concentration():
    result=failure_concentration({'family':{'a':dict(total=5,successes=3),
                                            'b':dict(total=95,successes=85)}},dict(total=100,successes=88))
    assert not result['clear_concentration']
    assert not result['equal_failure_rates_certified']


def test_all_failures_in_one_shape_are_detected_despite_overall_eighty_percent():
    result=failure_concentration({'family':{str(i):dict(total=20,successes=0 if i==0 else 20)
                                           for i in range(6)}},dict(total=120,successes=100))
    assert result['clear_concentration']
    assert [r['group'] for r in result['comparisons'] if r['clear_concentration']]==['0']


def case(index,family):
    return dict(id=str(index),payload=dict(shape=family,size=[.2,.3,.4],mass=.6),source=[0,0,.2,0,0,0,1],
                destination=[.3,0,.2,0,0,0,1],support_top=0.,strata=dict(family=family))


def save(root,item,success):
    out=root/item['id'];(out/'source_snapshot').mkdir(parents=True)
    (out/'source_snapshot/manifest.json').write_text(json.dumps({'amsrr_det/flight.py':'one-version'}))
    (out/'result.json').write_text(json.dumps(dict(task_success=success,reason='complete' if success else 'failed',
        request={k:item[k] for k in ['payload','source','destination','support_top']},scope='test',
        audit=dict(contact_force_components='normal_and_friction',clearance_detection='refitted_body_bounds',
                   support_boundary_force_components='normal_and_friction_conservative_anchor_geometry'))))


def test_overall_eighty_percent_cannot_hide_a_failed_shape_or_missing_result(tmp_path):
    cases=[case(i,'box' if i<4 else 'cylinder') for i in range(5)]
    for item in cases[:4]:save(tmp_path,item,True)
    panel=dict(role='development',cases=cases);config=dict(required_strata=['family'],minimum_success_fraction=.8)
    result=report(panel,[tmp_path],config)
    assert result['overall']['fraction']==.8
    assert result['missing']==1
    assert result['strata']['family']['cylinder']['fraction']==0.
    assert not result['reaches_all_rate_thresholds']
    save(tmp_path,cases[-1],False)
    result=report(panel,[tmp_path],config)
    assert result['missing']==0 and not result['reaches_all_rate_thresholds']


def test_duplicate_results_cannot_increase_denominator(tmp_path):
    item=case(0,'box');save(tmp_path,item,True)
    with pytest.raises(ValueError,match='duplicate result'):
        report(dict(role='development',cases=[item]),[tmp_path,tmp_path],dict(required_strata=['family'],minimum_success_fraction=.8))


def test_old_normal_only_force_audit_cannot_certify_success(tmp_path):
    item=case(0,'box');save(tmp_path,item,True)
    path=tmp_path/'0/result.json';result=json.loads(path.read_text());result.pop('audit');path.write_text(json.dumps(result))
    result=report(dict(role='development',cases=[item]),[tmp_path],dict(required_strata=['family'],minimum_success_fraction=.8))
    assert result['overall']['successes']==0
    assert result['rows'][0]['reported_task_success']
    assert result['rows'][0]['reason']=='incomplete_contact_force_audit'


def test_internal_boundary_friction_must_be_audited_before_acceptance(tmp_path):
    item=case(0,'box');save(tmp_path,item,True)
    path=tmp_path/'0/result.json';saved=json.loads(path.read_text())
    saved['audit'].pop('support_boundary_force_components');path.write_text(json.dumps(saved))
    result=report(dict(role='development',cases=[item]),[tmp_path],dict(required_strata=['family'],minimum_success_fraction=.8))
    assert result['overall']['successes']==0
    assert result['rows'][0]['reason']=='incomplete_support_boundary_audit'


@pytest.mark.parametrize('option',['planned_morph_input','trajectory_feedforward',
                                  'normal_moment_refinement','rest_to_rest_retreat','relative_acquisition'])
def test_different_controller_options_cannot_certify_a_common_method(tmp_path,option):
    cases=[case(0,'box'),case(1,'box')]
    for i,item in enumerate(cases):
        save(tmp_path,item,True)
        path=tmp_path/item['id']/'result.json';saved=json.loads(path.read_text())
        saved.update(scope='fresh_generation_continuous_assembly_and_task')
        saved[option]=bool(i)
        path.write_text(json.dumps(saved))
    result=report(dict(role='heldout',cases=cases),[tmp_path],
                  dict(required_strata=['family'],minimum_success_fraction=.8))
    assert result['reaches_all_rate_thresholds']
    assert result['all_results_same_runtime_sources']
    assert result['execution_setting_changes']=={option:[False,True]}
    assert not result['ready_for_completion_review']


def test_generated_spawn_is_allowed_but_saved_candidates_do_not_certify_generation(tmp_path):
    item=case(0,'box');save(tmp_path,item,True)
    path=tmp_path/'0/result.json';saved=json.loads(path.read_text())
    saved.update(scope='fresh_generation_spawn_task',execution_entrypoint='fresh_task_input',engine='MuJoCo')
    saved['audit'].pop('support_boundary_force_components')
    saved['audit'].update(engine='MuJoCo',support_representation='native_unsplit')
    config=dict(required_strata=['family'],minimum_success_fraction=.8,assembly_required=False)
    path.write_text(json.dumps(saved))
    assert report(dict(role='heldout',cases=[item]),[tmp_path],config)['ready_for_completion_review']
    saved['scope']='prepared_structure_task_execution_only'
    saved['execution_entrypoint']='saved_development_candidate';path.write_text(json.dumps(saved))
    result=report(dict(role='heldout',cases=[item]),[tmp_path],config)
    assert result['overall']['successes']==1 and not result['ready_for_completion_review']


def test_mujoco_support_representation_is_required_for_boundary_exemption(tmp_path):
    item=case(0,'box');save(tmp_path,item,True)
    path=tmp_path/'0/result.json';saved=json.loads(path.read_text())
    saved['audit'].update(engine='MuJoCo');saved['audit'].pop('support_boundary_force_components')
    path.write_text(json.dumps(saved))
    result=report(dict(role='development',cases=[item]),[tmp_path],dict(required_strata=['family'],minimum_success_fraction=.8))
    assert result['overall']['successes']==0


def test_different_collision_libraries_cannot_certify_one_method(tmp_path):
    cases=[case(0,'box'),case(1,'box')]
    for i,item in enumerate(cases):
        save(tmp_path,item,True)
        path=tmp_path/item['id']/'result.json';saved=json.loads(path.read_text())
        saved.update(scope='fresh_generation_continuous_assembly_and_task',
                     contact_engine=dict(sha256=str(i),ccd_tolerance=1e-9,ccd_iterations=100))
        path.write_text(json.dumps(saved))
    result=report(dict(role='heldout',cases=cases),[tmp_path],
                  dict(required_strata=['family'],minimum_success_fraction=.8))
    assert result['reaches_all_rate_thresholds']
    assert result['execution_setting_changes']=={'contact_engine_sha256':['0','1']}
    assert not result['ready_for_completion_review']
