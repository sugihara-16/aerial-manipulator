"""Simultaneous evidence from fixed, aligned blocks of count-specific trials.

Let C[n,k] be the cost of the frozen generator at count n and trial index k,
including failure (+infinity). The latent block costs Y[k] = min_n C[n,k]
are IID across k. For the final COMMON incumbent B <= min_k Y[k],
P[exists n: P(C[n,new] < B) > epsilon] <= (1-epsilon)**K.
The count event is contained in the block-minimum event; no union bound over
counts is needed. This is not a guarantee about each count's own incumbent,
nor about the existence of an undiscovered better structure.

An auxiliary accepted candidate can only lower B. A count-wide physical lower
bound L[n] > B permits skipping that count: its latent outputs cannot beat the
final B. Individual observed/censored block minima need not remain IID; the
proof uses outputs of the fixed procedures had no pruning occurred.

A censored per-trial value may also be a finalized component of the fixed
procedure's eventual maximum cost. It is accepted only above the complete
incumbent; it counts as nonimprovement evidence, never as a feasible output.
The bound concerns the frozen numerical procedure, not physical optimality.

Lists use preassigned trial indices, not completion order. None is unfinished;
+infinity is a completed unsuccessful trial. All needed aligned prefixes must
finish. Numerical budgets, generation and acceptance rules are frozen first.
"""
import math
from .structure_bounds import compare_count_groups


def required_trials(improvement_probability=.05, total_error=.05, groups=8):
    if (not 0 < improvement_probability < 1 or not 0 < total_error < 1
            or not isinstance(groups, int) or not 1 <= groups <= 8):
        raise ValueError('invalid stopping probabilities or group count')
    # groups validates the declared search scope; the block bound shares alpha.
    return math.ceil(math.log(total_error)/math.log1p(-improvement_probability))


def fixed_count_evidence(modules, outcomes, incumbent, *, protocol_id,
                         improvement_probability=.05, total_error=.05):
    """The same IID order-statistic bound when module count is a task input."""
    count=required_trials(improvement_probability,total_error,1)
    if not isinstance(modules,int) or not 1<=modules<=8 or not protocol_id:
        raise ValueError('a fixed module count and frozen protocol are required')
    if not math.isfinite(incumbent) or incumbent<0:
        raise ValueError('a finite accepted incumbent is required')
    for cost in outcomes:
        if cost is not None and (math.isnan(cost) or cost<incumbent):
            raise ValueError('incumbent must retain every completed improvement')
    complete=len(outcomes)>=count and all(x is not None for x in outcomes)
    return dict(complete=complete,modules=modules,protocol_id=protocol_id,
        method='fixed_count_independent_trial_minimum',required_trials=count,
        completed_trials=sum(x is not None for x in outcomes),
        successful_trials=sum(x is not None and math.isfinite(x) for x in outcomes),
        one_more_trial_improvement_probability_upper=-math.expm1(math.log(total_error)/len(outcomes)) if complete else None,
        error_probability=total_error,improvement_probability_target=improvement_probability,
        undiscovered_structure_existence_probability_claimed=False)


def selection_evidence(bounds, outcomes, incumbent, *, protocol_id,
                       improvement_probability=.05, total_error=.05,censored_bounds=None):
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
    censored_bounds={} if censored_bounds is None else censored_bounds
    for row in rows:
        values=outcomes.get(row['modules'],[])
        censored=censored_bounds.get(row['modules'],{})
        for index,lower in censored.items():
            if not 0<=index<len(values) or values[index]!=lower or lower<=incumbent:
                raise ValueError('censored trial needs a nonimproving fixed-output lower bound')
        for value in values:
            if value is None:continue
            if math.isnan(value) or value < 0:
                raise ValueError('trial costs must be nonnegative or positive infinity')
            if incumbent > value:
                raise ValueError('incumbent must retain the best completed candidate')
        prefix=values[:count]
        completed=sum(value is not None for value in prefix)
        row.update(required_trials=count,completed_prefix_trials=completed,
                   successful_prefix_trials=sum(value is not None and math.isfinite(value) and i not in censored for i,value in enumerate(prefix)),
                   bounded_nonimprovement_trials=sum(i in censored for i in range(len(prefix))))
        if row['status']=='excluded_by_weight':
            row['one_more_trial_improvement_probability_upper']=0.
        elif completed==count:
            row.update(status='fixed_trial_probability_bound',
                       one_more_trial_improvement_probability_upper=-math.expm1(math.log(total_error)/count),
                       shared_error_probability=total_error)
        else:
            row['status']='incomplete_trials'
            row['one_more_trial_improvement_probability_upper']=None
    complete=all(r['status']!='incomplete_trials' for r in rows)
    if not complete:
        for row in rows:
            if row['status']!='excluded_by_weight':
                row['one_more_trial_improvement_probability_upper']=None
                row.pop('shared_error_probability',None)
    return dict(protocol_id=protocol_id,complete=complete,
                method='aligned_count_block_minimum',required_blocks=count,
                scope='one additional fixed independent trial in every count; each count is contained in this block event',
                block_improvement_probability_upper=(-math.expm1(math.log(total_error)/count) if complete else None),
                undiscovered_structure_existence_probability_claimed=False,
                improvement_probability_target=improvement_probability,
                simultaneous_error_probability=total_error,groups=rows)
