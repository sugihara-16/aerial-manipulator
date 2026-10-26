"""Optional SSH generation workers with an independent, fixed routing law.

Only newly generated contact/connection candidates cross the connection. All
physical screens run locally. Machine choice is part of the fixed generator;
it never depends on earlier results, worker load, or completion order.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import os
import pickle
import platform
import shlex
import struct
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
import numpy as np
from .structure_workers import _TrialFuture, _generate_trial_candidates, planning_context


def remote_choice(seed, modules, index, probability):
    rng = np.random.default_rng(np.random.SeedSequence([seed, modules, index, 914]))
    return bool(rng.random() < probability)


def send(stream, value, lock):
    raw = pickle.dumps(value, protocol=5)
    with lock:
        stream.write(struct.pack('!Q', len(raw))+raw)
        stream.flush()


def receive(stream):
    def exact(size):
        pieces = []
        while size:
            part = stream.read(size)
            if not part:
                raise EOFError('remote generation connection closed')
            pieces.append(part)
            size -= len(part)
        return b''.join(pieces)
    return pickle.loads(exact(struct.unpack('!Q', exact(8))[0]))


def initialize(directory):
    # Solver output must not corrupt the framed result stream.
    os.dup2(2, 1)
    from build.frozen_structure_trial import initialize_screen_memo
    initialize_screen_memo(directory)


def server(workers):
    output, lock = sys.stdout.buffer, threading.Lock()
    expected = receive(sys.stdin.buffer)
    mismatches = [name for name, digest in expected.items()
                  if not Path(name).is_file() or hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest]
    import scipy, pinocchio, mujoco
    identity = dict(machine=platform.machine(), processor=next((line.split(':', 1)[1].strip()
        for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')), ''),
        versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                      pinocchio=pinocchio.__version__, mujoco=mujoco.__version__), mismatches=mismatches)
    send(output, identity, lock)
    if mismatches:
        return
    with tempfile.TemporaryDirectory(prefix='amsrr_remote_generation_', dir='/dev/shm') as folder:
        pool = planning_context().Pool(workers, initializer=initialize, initargs=(folder,))
        try:
            while True:
                item = receive(sys.stdin.buffer)
                if item is None:
                    break
                key, args = item
                pool.apply_async(_generate_trial_candidates, args,
                    callback=lambda value, key=key: send(output, (key, value, None), lock),
                    error_callback=lambda error, key=key: send(output, (key, None, str(error)), lock))
        except EOFError:
            pass
        finally:
            # Keep draining result messages until workers have finished.
            # Abrupt Pool termination can strand a writer on a full pipe.
            Path(folder, 'cancelled').touch()
            pool.close()
            pool.join()


class RemoteFuture:
    def __init__(self, client, key):
        self.client, self.key = client, key

    def ready(self):
        if self.client.failure:
            raise RuntimeError(self.client.failure)
        return self.key in self.client.answers

    def get(self):
        if not self.ready():
            raise ValueError('remote generation is incomplete')
        value, error = self.client.answers[self.key]
        if error:
            raise RuntimeError(error)
        return value


class RemoteClient:
    def __init__(self, config, sources):
        root = config['workspace']
        arguments = ['env', '-u', 'LD_PRELOAD', 'LD_LIBRARY_PATH='+root+'/.venv/lib',
            'PYTHONPATH=.:build/body_bounds/python', 'OPENBLAS_NUM_THREADS=1', 'OMP_NUM_THREADS=1',
            '.venv/bin/python', '-u', '-m', 'amsrr_det.remote_structure_workers', str(config['workers'])]
        command = 'cd '+shlex.quote(root)+' && '+shlex.join(arguments)
        self.process = subprocess.Popen(['ssh', *config.get('ssh_options', []), '-o', 'BatchMode=yes',
            '-p', str(config['port']), config['host'], command], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        self.lock = threading.Lock()
        self.answers, self.counter, self.failure = {}, 0, None
        send(self.process.stdin, sources, self.lock)
        self.identity = receive(self.process.stdout)
        if self.identity['mismatches']:
            self.process.wait()
            raise ValueError('remote generation source mismatch: '+str(self.identity['mismatches']))
        self.thread = threading.Thread(target=self._receive, daemon=True)
        self.thread.start()

    def _receive(self):
        try:
            while True:
                key, value, error = receive(self.process.stdout)
                self.answers[key] = (value, error)
        except Exception as error:
            self.failure = str(error)

    def submit(self, args):
        key = self.counter
        self.counter += 1
        send(self.process.stdin, (key, args), self.lock)
        return RemoteFuture(self, key)

    def close(self):
        if self.process.poll() is None:
            send(self.process.stdin, None, self.lock)
            self.process.stdin.close()
            self.process.wait(timeout=15)
        self.thread.join(timeout=1)


class GenerationBackend:
    def __init__(self, context, processes, config, sources, **kwargs):
        # Fork local workers before the SSH reader starts a thread.
        self.local = context.Pool(processes=processes, **kwargs)
        self.probability = config['workers']/(processes+config['workers'])
        try:
            self.remote = RemoteClient(config, sources)
        except Exception:
            self.local.terminate()
            self.local.join()
            raise

    def apply_async(self, function, args, kwds=None):
        if function is _generate_trial_candidates:
            _, modules, index, seed, _ = args
            if remote_choice(seed, modules, index, self.probability):
                return self.remote.submit(args)
        return self.local.apply_async(function, args, kwds or {})


class RemoteGenerationPool:
    def __init__(self, context, processes, config, sources, **kwargs):
        self.backend = GenerationBackend(context, processes, config, sources, **kwargs)
        self.identity = self.backend.remote.identity
        self.probability = self.backend.probability

    def apply_async(self, function, args, kwds=None):
        from build.frozen_structure_trial import trial
        if function is trial:
            return _TrialFuture(self.backend, args, kwds)
        return self.backend.apply_async(function, args, kwds)

    def close(self):
        # Stop independent machines concurrently, but finish both cleanups
        # before returning a plan or deleting the task-local memo.
        with ThreadPoolExecutor(max_workers=1) as closing:
            remote = closing.submit(self.backend.remote.close)
            self.backend.local.close()
            self.backend.local.join()
            remote.result()

    def join(self):
        self.backend.local.join()


if __name__ == '__main__':
    server(int(sys.argv[1]))
