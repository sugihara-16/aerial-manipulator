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


def evaluate(inputs, calculate):
    return active.evaluate(inputs, calculate)[0] if active is not None else calculate()


class PlanningMemo:
    def __init__(self, directory):
        self.directory = Path(directory)

    def evaluate(self, inputs, calculate):
        raw = json.dumps(inputs, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
        key = hashlib.sha256(raw).hexdigest()
        target = self.directory / (key + '.pickle')
        with (self.directory / (key + '.lock')).open('wb') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
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
