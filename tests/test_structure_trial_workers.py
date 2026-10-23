"""A trial contributes evidence only after all predetermined starts finish."""
from types import SimpleNamespace
import pytest
from amsrr_det.structure_workers import _TrialFuture


def test_late_screen_remains_required_and_can_improve_the_trial():
    completed=[False]*4
    screens=[]
    for i,cost in enumerate([9.,8.,7.,1.]):
        screens.append(SimpleNamespace(ready=lambda i=i:completed[i],
            get=lambda i=i,cost=cost:({'candidate':i},str(i),{'burden':cost})))
    class Pool:
        def apply_async(self,fn,args,kwds=None):
            if kwds is None:
                return SimpleNamespace(ready=lambda:True,get=lambda:list(range(4)))
            return screens[args[1]]
    result=_TrialFuture(Pool(),({},3,18,73),{'task_wrench':True})
    completed[:3]=[True]*3
    assert not result.ready()
    with pytest.raises(ValueError):result.get()
    completed[3]=True
    answer=result.get()
    assert answer['burden']==1. and answer['plan']=='3'
    assert [r['start'] for r in answer['rows']]==list(range(4))
    assert (answer['modules'],answer['trial'],answer['seed'])==(3,18,73)
