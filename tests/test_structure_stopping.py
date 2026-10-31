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
    assert required_trials()==59
    bounds=[dict(modules=n,lower_bound=50*n) for n in range(1,9)]
    report=selection_evidence(bounds,{1:[math.inf]*59,2:[120.]*59},120.,protocol_id='test')
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


def test_common_incumbent_gives_simultaneous_marginal_bound_without_union_allocation():
    # Dependent count outputs within a block, with atoms and failed solves.
    # Enumerate the sampling distribution independently of the implementation.
    values=[(1.,math.inf),(math.inf,2.),(4.,4.),(math.inf,math.inf)]
    probabilities=[.05,.15,.3,.5];n=5
    for auxiliary in [math.inf,3.]:
        for epsilon in [.04,.08,.18,.35,.55,.85]:
            violation=0.
            for sample in product(range(len(values)),repeat=n):
                best=min(auxiliary,min(min(values[i]) for i in sample))
                marginal=[sum(p for v,p in zip(values,probabilities) if v[group]<best)
                          for group in range(2)]
                if max(marginal)>epsilon:
                    violation+=math.prod(probabilities[i] for i in sample)
            assert violation<=(1-epsilon)**n+1e-13


def test_one_unfinished_count_prevents_the_shared_probability_claim():
    n=required_trials();bounds=[dict(modules=i,lower_bound=10*i) for i in range(1,9)]
    r=selection_evidence(bounds,{1:[100.]*n,2:[None]+[math.inf]*(n-1)},25.,protocol_id='test')
    assert not r['complete'] and r['block_improvement_probability_upper'] is None
    assert r['groups'][0]['completed_prefix_trials']==n
    assert r['groups'][0]['one_more_trial_improvement_probability_upper'] is None


def test_data_dependent_physical_pruning_retains_the_global_latent_minimum():
    values=[(1.,4.),(3.,5.),(math.inf,math.inf)];lower=[1.,4.]
    for sample in product(range(len(values)),repeat=5):
        best=5.
        for i in sample:
            for group in range(2):
                if lower[group]>best:continue
                best=min(best,values[i][group])
        assert best==min(5.,min(min(values[i]) for i in sample))


def test_finalized_component_pruning_preserves_fixed_generator_minimum():
    # The first component is costly but the second may improve. Pruning is
    # valid for a maximum over finalized components, not an interim optimizer.
    outputs=[(3.,8.),(6.,2.),(1.,4.),(12.,math.inf)]
    for order in product(range(len(outputs)),repeat=5):
        best=10.;bounded=[]
        for index in order:
            components=outputs[index]
            if components[0]>best:
                bounded.append(components[0]);continue
            best=min(best,max(components))
        assert best==min(10.,min(max(outputs[i]) for i in order))
        assert all(lower>best for lower in bounded)


def test_bounded_trials_are_complete_evidence_but_not_feasible_candidates():
    bounds=[dict(modules=1,lower_bound=1.)]
    report=selection_evidence(bounds,{1:[8.]+[9.]*58},8.,protocol_id='fixed',
        censored_bounds={1:{i:9. for i in range(1,59)}})
    assert report['complete']
    assert report['groups'][0]['successful_prefix_trials']==1
    assert report['groups'][0]['bounded_nonimprovement_trials']==58
    with pytest.raises(ValueError,match='nonimproving'):
        selection_evidence(bounds,{1:[7.]*59},8.,protocol_id='fixed',censored_bounds={1:{0:7.}})
