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


def _large_result_shutdown(connection):
    import multiprocessing as mp
    import os
    from amsrr_det.structure_workers import TrialPool
    os.setsid()
    pool=TrialPool(mp.get_context('fork'),processes=16)
    futures=[pool.apply_async(bytes,(1024*1024,)) for _ in range(48)]
    # No consumer calls get until shutdown: result messages exceed pipe
    # capacity, and shutdown must keep receiving them without stranded writers.
    pool.close();pool.join()
    connection.send([len(f.get()) for f in futures])
    connection.close()


def test_shutdown_drains_large_results_without_stranded_workers():
    import multiprocessing as mp
    import os,signal,sys
    if sys.platform!='linux':pytest.skip('Linux planning process contract')
    context=mp.get_context('fork');parent,child=context.Pipe(duplex=False)
    process=context.Process(target=_large_result_shutdown,args=(child,))
    process.start();child.close()
    try:
        process.join(timeout=15.)
        assert not process.is_alive(), 'planning shutdown is blocked'
        assert process.exitcode==0
        assert parent.poll() and parent.recv()==[1024*1024]*48
    finally:
        if process.is_alive():
            os.killpg(process.pid,signal.SIGKILL);process.join()
        parent.close()
