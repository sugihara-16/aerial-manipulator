"""Run a picklable planning result in an isolated Linux worker process.

The worker inherits read-only planning inputs at fork. Only its result crosses
the pipe; controller state stays in the parent. Physical waiting includes the
measured worker startup and computation time.
"""
import multiprocessing as mp
import time


class ProcessPlan:
    def __init__(self,simulation_time,operation,*args):
        self.started_sim=float(simulation_time);self.started_wall=time.perf_counter()
        self.received=None;context=mp.get_context('fork')
        self.connection,child=context.Pipe(duplex=False)
        def calculate():
            try:answer=(True,operation(*args))
            except Exception as exc:answer=(False,type(exc).__name__+': '+str(exc))
            child.send((answer,time.perf_counter()-self.started_wall));child.close()
        self.process=context.Process(target=calculate);self.process.start();child.close()

    def ready(self,simulation_time):
        if self.received is None and self.connection.poll():self.received=self.connection.recv()
        if self.received is None and self.process.exitcode is not None:
            raise ValueError('planning worker exited without a result')
        return self.received is not None and simulation_time-self.started_sim>=self.received[1]

    def result(self,simulation_time):
        if not self.ready(simulation_time):raise ValueError('planning result is not available yet')
        (ok,value),elapsed=self.received;self.close()
        if not ok:raise ValueError(value)
        return value,dict(wall_s=elapsed,simulated_wait_s=float(simulation_time-self.started_sim),worker='process')

    def close(self):
        # Drain a completed result before joining so a full pipe cannot stall
        # the worker when the independent physical audit ends the experiment.
        if self.received is None:
            try:self.received=self.connection.recv()
            except EOFError:pass
        self.process.join();self.connection.close()
