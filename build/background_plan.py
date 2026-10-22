"""Plan on an observation copy while simulation and current control continue.

An accelerated/slowed evaluator must not make a planning delay disappear:
acceptance also waits for at least the measured planning duration in simulated
time. This is a delay experiment, not a hard realtime scheduling guarantee.
"""
from concurrent.futures import ThreadPoolExecutor
import time


class BackgroundPlan:
    def __init__(self,simulation_time,operation,*args):
        self.started_sim=float(simulation_time);self.started_wall=time.perf_counter()
        self.pool=ThreadPoolExecutor(max_workers=1);self.elapsed_wall=None
        def calculate():
            try:return operation(*args)
            finally:self.elapsed_wall=time.perf_counter()-self.started_wall
        self.future=self.pool.submit(calculate)

    def ready(self,simulation_time):
        return self.future.done() and simulation_time-self.started_sim>=self.elapsed_wall

    def result(self,simulation_time):
        if not self.ready(simulation_time):raise ValueError('planning result is not available yet')
        value=self.future.result();self.pool.shutdown()
        return value,dict(wall_s=self.elapsed_wall,simulated_wait_s=float(simulation_time-self.started_sim))

    def close(self):
        self.pool.shutdown(wait=True)
