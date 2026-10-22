"""Scheduler-only regression with a simulated clock, not a speed benchmark."""
from types import SimpleNamespace
import json
import build.select_structure as selector


def check(initial_delay, initial_cost, trial_cost, slow_last=False):
    clock=SimpleNamespace(now=0.)
    selector.time=SimpleNamespace(perf_counter=lambda:clock.now,
        sleep=lambda step:setattr(clock,'now',clock.now+step))
    selector.build_model=lambda *a,**k:(None,dict(mass=2.,object=dict(mass=.5)))
    selector.count_lower_bounds=lambda *a,**k:[dict(modules=n,lower_bound=50.*n)for n in range(1,9)]
    def initial(*args):
        return dict(plan='initial',prepared=dict(burden=initial_cost),rows=[],wall_s=initial_delay)
    def trial(request,n,index,seed,**kwargs):
        cost=trial_cost if n==1 and index==0 else float('inf')
        return dict(modules=n,trial=index,rows=[],burden=cost,
            plan='trial'if cost<float('inf')else None,
            prepared=dict(burden=cost)if cost<float('inf')else None,wall_s=0.)
    selector.initial_candidate=initial;selector.trial=trial
    class Pool:
        def __init__(self,processes):self.available=[0.]*processes
        def apply_async(self,fn,args,kwds=None):
            duration=initial_delay if fn is initial else (.3 if args[1]!=1 or(slow_last and args[2]==98)else .00001)
            slot=min(range(len(self.available)),key=lambda i:self.available[i])
            end=max(clock.now,self.available[slot])+duration;self.available[slot]=end
            return SimpleFuture(fn,args,kwds or{},end)
        def terminate(self):pass
        def join(self):pass
    class SimpleFuture:
        def __init__(self,fn,args,kwargs,end):self.fn,self.args,self.kwargs,self.end=fn,args,kwargs,end
        def ready(self):return clock.now>=self.end
        def get(self):
            assert self.ready()
            return self.fn(*self.args,**self.kwargs)
    selector.planning_context=lambda:SimpleNamespace(Pool=Pool,get_start_method=lambda:'simulated')
    request=dict(payload=dict(shape='box',size=[.2]*3,mass=.5))
    _,_,report=selector.select_structure(request,workers=2,seed=7)
    assert report['selection_evidence']['complete']
    necessary=report['selection_evidence']['groups'][0]
    assert necessary['completed_prefix_trials']==99
    assert report['selected_burden']==min(initial_cost,trial_cost)
    return dict(clock_s=clock.now,cancelled=report['cancelled_excluded_trials'],cost=report['selected_burden'])



def test_only_physically_excluded_unfinished_work_is_cancelled(monkeypatch):
    with monkeypatch.context() as context:
        # The fixture replaces scheduler inputs for this test only.
        previous={key:getattr(selector,key) for key in ['time','build_model','count_lower_bounds','initial_candidate','trial','planning_context']}
        try:
            first=check(.2,75.,float('inf'))
            assert first['clock_s']<.3 and first['cancelled']
            late=check(.2,60.,75.)
            assert late['clock_s']>=.2 and late['cost']==60.
            necessary=check(.005,75.,float('inf'),True)
            assert necessary['clock_s']>=.3
        finally:
            for key,value in previous.items():setattr(selector,key,value)
