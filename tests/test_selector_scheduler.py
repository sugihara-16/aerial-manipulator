"""Scheduler-only regression with a simulated clock, not a speed benchmark."""
from types import SimpleNamespace
import json
import build.select_structure as selector


def check(initial_delay, initial_cost, trial_cost, slow_last=False, accepted_index=0, slow_extra=False):
    clock=SimpleNamespace(now=0.)
    selector.time=SimpleNamespace(perf_counter=lambda:clock.now,
        sleep=lambda step:setattr(clock,'now',clock.now+step))
    selector.build_model=lambda *a,**k:(None,dict(mass=2.,object=dict(mass=.5)))
    selector.count_lower_bounds=lambda *a,**k:[dict(modules=n,lower_bound=50.*n)for n in range(1,9)]
    def initial(*args):
        return dict(plan='initial' if initial_cost<float('inf') else None,prepared=dict(burden=initial_cost) if initial_cost<float('inf') else None,rows=[],wall_s=initial_delay)
    def trial(request,n,index,seed,**kwargs):
        cost=trial_cost if n==1 and index==accepted_index else float('inf')
        return dict(modules=n,trial=index,rows=[],burden=cost,
            plan='trial'if cost<float('inf')else None,
            prepared=dict(burden=cost)if cost<float('inf')else None,wall_s=0.)
    selector.task_initial_candidate=initial;selector.trial=trial
    prefix_ends={}
    class Pool:
        def __init__(self,processes,initializer=None,initargs=()):self.available=[0.]*processes
        def apply_async(self,fn,args,kwds=None):
            if fn is not initial and args[2]>=selector.required_trials():
                assert len(prefix_ends)==8*selector.required_trials()
                assert max(prefix_ends.values())<=clock.now
            duration=initial_delay if fn is initial else (.3 if args[1]!=1 or(slow_last and args[2]==selector.required_trials()-1)or(slow_extra and args[2]>accepted_index)else .00001)
            if slow_extra and fn is not initial and args[1]==1 and args[2]==accepted_index:duration=1.
            slot=min(range(len(self.available)),key=lambda i:self.available[i])
            end=max(clock.now,self.available[slot])+duration;self.available[slot]=end
            if fn is not initial and args[2]<selector.required_trials():prefix_ends[args[1:3]]=end
            return SimpleFuture(fn,args,kwds or{},end)
        def terminate(self):pass
        def close(self):pass
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
    assert necessary['completed_prefix_trials']==selector.required_trials()
    expected=trial_cost if report['cancelled_auxiliary_initial'] else min(initial_cost,trial_cost)
    assert report['selected_burden']==expected
    return dict(clock_s=clock.now,cancelled=report['cancelled_excluded_trials'],cost=report['selected_burden'],
                cancelled_auxiliary=report['cancelled_auxiliary_initial'], extras=report['cancelled_extra_trials'], rows=report['rows'])



def test_only_physically_excluded_unfinished_work_is_cancelled(monkeypatch):
    with monkeypatch.context() as context:
        # The fixture replaces scheduler inputs for this test only.
        previous={key:getattr(selector,key) for key in ['time','build_model','count_lower_bounds','task_initial_candidate','trial','planning_context','TrialPool']}
        try:
            selector.TrialPool=lambda context,**kwargs:context.Pool(**kwargs)
            first=check(.2,75.,float('inf'))
            assert first['clock_s']<.3 and first['cancelled']
            late=check(.2,60.,75.)
            # The auxiliary is not a required independent observation. A late
            # improvement may be forgone only after all count evidence exists.
            assert late['clock_s']<.2 and late['cost']==75. and late['cancelled_auxiliary']
            necessary=check(.005,75.,float('inf'),True)
            assert necessary['clock_s']>=.3
            late_candidate=check(.005,float('inf'),75.,accepted_index=61)
            assert late_candidate['cost']==75.
            assert late_candidate['clock_s']>.3
            extra=check(.005,float('inf'),75.,accepted_index=61,slow_extra=True)
            assert extra['extras']
            assert all(x['trial']>=selector.required_trials() for x in extra['extras'])
            observed={(x['modules'],x['trial']) for x in extra['rows']}
            assert all((x['modules'],x['trial']) not in observed for x in extra['extras'])
        finally:
            for key,value in previous.items():setattr(selector,key,value)
