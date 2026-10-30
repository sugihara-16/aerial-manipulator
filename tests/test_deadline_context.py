import threading
import time
import numpy as np
import pytest
from build import deadline_context


def test_deadline_keeps_first_feasible_then_stops_optional_work():
    output=[]
    try:
        deadline_context.configure(time.perf_counter()-1,lambda u,loss:output.append((u,loss)))
        assert deadline_context.available()  # worker may still find a first usable nominal
        deadline_context.publish(np.array([2.]),3.)
        assert not deadline_context.available()
        assert output[0][1]==3.
    finally:deadline_context.configure()


def test_background_planner_does_not_inherit_actuation_deadline():
    result=[]
    try:
        deadline_context.configure(time.perf_counter()-1,lambda *args:None)
        deadline_context.publish(np.zeros(1))
        t=threading.Thread(target=lambda:result.append((deadline_context.active(),deadline_context.available())))
        t.start();t.join()
        assert result==[(False,True)]
        assert not deadline_context.available()
    finally:deadline_context.configure()


def test_observation_expiry_stops_first_feasible_retries_and_publication(monkeypatch):
    clock=[6.];output=[]
    monkeypatch.setattr(deadline_context.time,'perf_counter',lambda:clock[0])
    try:
        deadline_context.configure(5.,lambda *args:output.append(args),expires=10.)
        assert deadline_context.available()  # a late first solution can still be used
        assert deadline_context.remaining()==4.
        clock[0]=10.
        assert not deadline_context.available()  # too old even without an incumbent
        assert deadline_context.remaining()==0.
        deadline_context.publish(np.zeros(1))
        assert not output
        from build.contact_condensed_solver import scaled_recovery
        # No algebra/native solver may start from an already discarded state.
        with pytest.raises(ValueError,match='time budget exhausted'):
            scaled_recovery(None,None,None,None,None)
    finally:deadline_context.configure()
