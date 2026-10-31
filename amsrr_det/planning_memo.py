"""Exact task-local memoization shared by planning worker processes.

This stores numerical evaluations, never proposals or a persistent catalogue.
The owner creates and removes the directory for each planning call. A lock per
exact input lets concurrent duplicate calls share one evaluation.
"""
import fcntl
import hashlib
import json
from pathlib import Path
import pickle


active = None


class PlanningCancelled(RuntimeError):
    """Unobserved work stopped after the selector has made its decision."""


class FixedOutputDominated(RuntimeError):
    """A finalized component bounds this frozen procedure's eventual output."""
    def __init__(self, lower_bound, incumbent, component):
        self.evidence=dict(lower_bound=float(lower_bound),incumbent=float(incumbent),
                           component=component,scope='fixed_procedure_output_not_physical_optimum')
        super().__init__('fixed procedure output cannot improve incumbent')


def check_output_bound(lower_bound, component):
    if active is None:return
    target=active.directory/'incumbent.json'
    if not target.exists():return
    incumbent=float(json.loads(target.read_text()))
    if lower_bound > incumbent + 1e-6*(1+incumbent):
        raise FixedOutputDominated(lower_bound,incumbent,component)


def check_cancelled():
    if active is not None:
        active.check_cancelled()


def evaluate(inputs, calculate):
    return active.evaluate(inputs, calculate)[0] if active is not None else calculate()


class PlanningMemo:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.task_group = None

    def check_cancelled(self):
        if (self.directory / 'cancelled').exists() or (self.task_group is not None and (self.directory / ('excluded_'+str(self.task_group))).exists()):
            raise PlanningCancelled('unobserved planning work cancelled')

    def evaluate(self, inputs, calculate):
        self.check_cancelled()
        raw = json.dumps(inputs, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
        key = hashlib.sha256(raw).hexdigest()
        target = self.directory / (key + '.pickle')
        with (self.directory / (key + '.lock')).open('wb') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            self.check_cancelled()
            if target.exists():
                with target.open('rb') as stream:
                    saved_input, result = pickle.load(stream)
                if saved_input != raw:
                    raise ValueError('planning memo input digest collision')
                return result, True
            result = calculate()
            with target.open('wb') as stream:
                pickle.dump((raw, result), stream, protocol=pickle.HIGHEST_PROTOCOL)
            return result, False
