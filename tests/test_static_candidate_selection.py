"""Static choice must account for all phases without consuming task outcomes."""
from types import SimpleNamespace
import numpy as np
import pytest
from amsrr_det.structure_search import select_static_candidate,SearchBudgetExpired


def candidate(modules,trims,restoring,tag):
    return SimpleNamespace(plant=SimpleNamespace(meta=dict(modules=modules,edges=[])),
        surfaces=[(0,'yaw1')],closed_q=np.zeros(7),open_q=np.zeros(7),
        points=np.zeros((1,3)),normals=np.array([[1.,0.,0.]]),
        trims=[dict(max_normalized_joint_effort=value,minimum_peak_effort_ratio=0.) for value in trims],
        fit=dict(restoring_load_screen=dict(directions=[dict(max_normalized_joint_effort=restoring)]),
                 task_level_search=[tag]),wall_s=0.)


@pytest.mark.parametrize('bad_trim,bad_restore',[([.9,.2,.2],.3),([.2,.2,.2],.9)])
def test_empty_posture_and_restoring_loads_are_not_hidden_by_low_carried_load(bad_trim,bad_restore):
    first=candidate(3,bad_trim,bad_restore,'first')
    second=candidate(5,[.5,.4,.6],.65,'second')
    selected=select_static_candidate(iter([first,second]),2)
    assert selected is second
    assert selected.fit['candidate_selection']['selected_candidate']==2
    assert [r['static_burden'] for r in selected.fit['candidate_selection']['rows']]==[.9,.65]


def test_lower_static_burden_can_retain_first_and_keeps_unselected_search_record():
    first=candidate(3,[.2,.3,.4],.5,'first')
    second=candidate(5,[.4,.5,.6],.7,'second')
    selected=select_static_candidate(iter([first,second]),2)
    assert selected is first
    assert selected.fit['task_level_search']==['second']
    assert len(selected.fit['candidate_selection']['rows'])==2


@pytest.mark.parametrize('error_type',[ValueError,SearchBudgetExpired])
def test_second_search_failure_retains_valid_first_and_its_failure_cost_record(error_type):
    first=candidate(3,[.2,.3,.4],.5,'first')
    def generated():
        yield first
        error=error_type('second candidate not found');error.evidence={'attempts':4}
        raise error
    selected=select_static_candidate(generated(),2)
    report=selected.fit['candidate_selection']
    assert selected is first and report['selected_candidate']==1
    assert report['search_error']==dict(reason='second candidate not found',evidence={'attempts':4})
    assert report['wall_s']>=0 and selected.wall_s==report['wall_s']


def test_equal_burden_keeps_first_even_when_second_has_fewer_modules():
    first=candidate(5,[.2,.3,.4],.5,'first')
    second=candidate(3,[.2,.3,.4],.5,'second')
    assert select_static_candidate(iter([first,second]),2) is first


def test_late_lower_burden_candidate_cannot_displace_on_time_result(monkeypatch):
    import amsrr_det.structure_search as search
    clock=[0.];monkeypatch.setattr(search.time,'perf_counter',lambda:clock[0])
    first=candidate(3,[.5],.5,'first');late=candidate(5,[.1],.1,'late')
    def generated():
        clock[0]=1.;yield first
        clock[0]=3.;yield late
    selected=select_static_candidate(generated(),deadline=2.)
    assert selected is first
    assert [row['completed_within_budget'] for row in selected.fit['candidate_selection']['rows']]==[True,False]
    assert selected.fit['task_level_search']==['late']


def test_first_candidate_after_deadline_is_not_a_usable_plan(monkeypatch):
    import amsrr_det.structure_search as search
    clock=[0.];monkeypatch.setattr(search.time,'perf_counter',lambda:clock[0])
    def generated():
        clock[0]=3.;yield candidate(3,[.1],.1,'late')
    with pytest.raises(SearchBudgetExpired,match='no candidate completed'):
        select_static_candidate(generated(),deadline=2.)
