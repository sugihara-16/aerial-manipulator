"""Fixed-sample evidence for one more independent structure-generation trial.

For IID trial costs Y (failure = +infinity), an incumbent c <= min(Y_1..Y_n)
satisfies P[P(Y_new < c) > epsilon] <= (1-epsilon)**n, including ties.
This does not estimate whether an undiscovered better structure exists.
Reference: https://arxiv.org/abs/0805.1569

The caller must freeze the generator, acceptance rules and per-trial work
budget before sampling. Lists are indexed by preassigned independent trial
number, NOT completion order. None means unfinished; infinity means a finished
unsuccessful trial. Only the predetermined sample prefix certifies stopping,
so repeated calls and faster parallel completions cannot change the sample.
"""
import math
from .structure_bounds import compare_count_groups


def required_trials(improvement_probability=.05, total_error=.05, groups=8):
    if (not 0 < improvement_probability < 1 or not 0 < total_error < 1
            or not isinstance(groups, int) or not 1 <= groups <= 8):
        raise ValueError('invalid stopping probabilities or group count')
    return math.ceil(math.log(total_error/groups)/math.log1p(-improvement_probability))


def selection_evidence(bounds, outcomes, incumbent, *, protocol_id,
                       improvement_probability=.05, total_error=.05):
    """Combine count-wide physical bounds and fixed-prefix statistical evidence.

This reports conditional statistical evidence, not a test of IID sampling.
The protocol identifier should hash the frozen code, settings and task input.
Incumbent must be a physically accepted candidate under that same criterion.
"""
    groups=len(bounds)
    count=required_trials(improvement_probability,total_error,groups)
    if (not protocol_id or {r['modules'] for r in bounds} != set(range(1,groups+1))
            or len({r['modules'] for r in bounds}) != groups
            or set(outcomes)-set(range(1,groups+1))):
        raise ValueError('all module groups and a frozen protocol are required')
    rows=compare_count_groups(bounds,incumbent)
    for row in rows:
        values=outcomes.get(row['modules'],[])
        for value in values:
            if value is None:continue
            if math.isnan(value) or value < 0:
                raise ValueError('trial costs must be nonnegative or positive infinity')
            if incumbent > value:
                raise ValueError('incumbent must retain the best completed candidate')
        prefix=values[:count]
        completed=sum(value is not None for value in prefix)
        row.update(required_trials=count,completed_prefix_trials=completed,
                   successful_prefix_trials=sum(value is not None and math.isfinite(value) for value in prefix))
        if row['status']=='excluded_by_weight':
            row['one_more_trial_improvement_probability_upper']=0.
        elif completed==count:
            row.update(status='fixed_trial_probability_bound',
                       one_more_trial_improvement_probability_upper=-math.expm1(math.log(total_error/groups)/count),
                       allocated_error_probability=total_error/groups)
        else:
            row['status']='incomplete_trials'
            row['one_more_trial_improvement_probability_upper']=None
    return dict(protocol_id=protocol_id,complete=all(r['status']!='incomplete_trials' for r in rows),
                scope='one additional independent run of the same frozen generator per module count',
                undiscovered_structure_existence_probability_claimed=False,
                improvement_probability_target=improvement_probability,
                simultaneous_error_probability=total_error,groups=rows)
