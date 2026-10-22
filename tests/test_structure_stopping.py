"""Independent distribution checks, including ties and unsuccessful trials."""
from itertools import product
import math
import pytest
from amsrr_det.structure_stopping import required_trials,selection_evidence


def test_order_statistic_bound_for_discrete_costs_and_failures():
    # Enumerate the sampling distribution exactly, instead of checking only
    # the implementation's formula. Atoms include equal-cost outcomes and
    # failed solves; no continuity assumption is available.
    values=[3.,8.,math.inf]; probabilities=[.1,.25,.65]
    n=6
    for epsilon in [.05,.1,.2,.34,.36,.8]:
        violation_probability=0.
        for sample in product(range(len(values)),repeat=n):
            best=min(values[i] for i in sample)
            next_improvement=sum(p for value,p in zip(values,probabilities) if value<best)
            if next_improvement>epsilon:
                violation_probability+=math.prod(probabilities[i] for i in sample)
        assert violation_probability <= (1-epsilon)**n+1e-13


def test_required_count_and_full_group_coverage():
    assert required_trials()==99
    bounds=[dict(modules=n,lower_bound=50*n) for n in range(1,9)]
    report=selection_evidence(bounds,{1:[math.inf]*99,2:[120.]*99},120.,protocol_id='test')
    assert report['complete']
    assert [g['status'] for g in report['groups'][:2]]==['fixed_trial_probability_bound']*2
    assert all(g['status']=='excluded_by_weight' for g in report['groups'][2:])
    assert all(g['one_more_trial_improvement_probability_upper']<=.05 for g in report['groups'])


def test_parallel_completed_suffix_cannot_replace_an_unfinished_prefix():
    bounds=[dict(modules=1,lower_bound=10)]
    n=required_trials(groups=1)
    report=selection_evidence(bounds,{1:[None]+[math.inf]*(n+20)},100.,protocol_id='test')
    assert not report['complete']
    assert report['groups'][0]['one_more_trial_improvement_probability_upper'] is None
    report=selection_evidence(bounds,{1:[math.inf]*(n+21)},100.,protocol_id='test')
    assert report['complete']


def test_better_candidate_cannot_be_discarded_or_missing_group_ignored():
    with pytest.raises(ValueError,match='best completed'):
        selection_evidence([dict(modules=1,lower_bound=10)],{1:[90.]},100.,protocol_id='test')
    with pytest.raises(ValueError,match='all module groups'):
        selection_evidence([dict(modules=2,lower_bound=10)],{},100.,protocol_id='test')
