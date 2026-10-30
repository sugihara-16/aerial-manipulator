"""One outstanding allocation, with a nonblocking actuator publisher.

Worker completion is mapped from measured wall latency into plant event time.
Numerical integration time cannot make a worker result available early. The
publisher retains the last physical servo reference while a result is pending.
"""
import gc
import os
import pickle
import signal
import socket
import time


class AsyncAllocation:
    def __init__(self, controller, worker_cpu):
        a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM)
        a.setblocking(False)
        self.pid = os.fork()
        if self.pid == 0:
            a.close()
            try:
                if worker_cpu is not None:
                    os.sched_setaffinity(0, {worker_cpu})
                gc.disable()
                while True:
                    request = pickle.loads(b.recv(262144))
                    started = time.perf_counter()
                    try:
                        args, kwargs = request['args'], request['kwargs']
                        kwargs['deadline'] = request['observed_wall'] + .008
                        command = controller.update(*args, **kwargs)
                        computed = time.perf_counter()
                        payload = (command, controller.last)
                        error = None
                    except Exception as exception:
                        computed = time.perf_counter()
                        payload, error = None, repr(exception)
                    b.send(pickle.dumps(dict(kind='result', payload=payload,
                        started=started, computed=computed, error=error), protocol=5))
                    # The command is now in the receiver's kernel queue. This
                    # second packet is timing instrumentation only: its stamp
                    # bounds availability of the preceding command, including
                    # serialization/send, without charging physics integration.
                    sent = time.perf_counter()
                    b.send(pickle.dumps(dict(kind='sent', finished=sent), protocol=5))
            finally:
                b.close()
                os._exit(0)
        b.close()
        self.sock = a
        self.controller = controller
        self.job = None
        self.pending = None
        self.reply = None
        self.command = controller.last_command.copy()
        self.last = {}
        self.records = []
        self.held_ticks = 0
        self.closed = False

    def update(self, now, phase, args, kwargs, *, observed_wall=None):
        """Never wait for the worker. Every tick returns a servo command."""
        if observed_wall is None:
            observed_wall = time.perf_counter()
        applied=self.poll(now)
        if self.job is None:
            sent = time.perf_counter()
            request = dict(args=args, kwargs=kwargs, sent_wall=sent, observed_wall=observed_wall)
            try:
                self.sock.send(pickle.dumps(request, protocol=5))
                self.job = dict(sent_wall=sent, observed_wall=observed_wall, observation_s=now, phase=phase)
            except BlockingIOError:
                pass
        self.held_ticks += not applied
        self.controller.last = self.last
        return self.command.copy()

    def poll(self,now):
        if self.job is not None and self.pending is None:
            for _ in range(2):
                try:
                    packet = pickle.loads(self.sock.recv(262144))
                except BlockingIOError:
                    break
                if packet['kind'] == 'result':
                    self.reply = packet
                    continue
                if packet['kind'] != 'sent' or self.reply is None:
                    raise RuntimeError('invalid allocation response order')
                elapsed = packet['finished'] - self.job['observed_wall']
                self.pending = self.reply
                self.reply = None
                self.pending['available_s'] = self.job['observation_s'] + elapsed
                self.records.append(dict(phase=self.job['phase'], observation_s=self.job['observation_s'],
                    calculation_s=elapsed, available_s=self.pending['available_s'],
                    worker_calculation_s=self.pending['computed']-self.pending['started'],
                    response_transport_s=packet['finished']-self.pending['computed'],
                    dispatch_delay_s=self.job['sent_wall']-self.job['observed_wall'],
                    queue_delay_s=self.pending['started']-self.job['sent_wall'],
                    error=self.pending['error']))
                break
        applied = False
        if self.pending is not None and self.pending['available_s'] <= now+1e-10:
            if self.pending['error'] is None:
                self.command, self.last = self.pending['payload']
                self.records[-1]['application_tick_s'] = now
                applied = True
            self.job = self.pending = None
        self.controller.last = self.last
        return applied

    def close(self):
        if not self.closed:
            self.closed = True
            self.sock.close()
            try:
                os.kill(self.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            os.waitpid(self.pid, 0)
