"""Thread-local worker budget and publication of checked incumbents only."""
import threading
import time

_state=threading.local()


def configure(deadline=None,callback=None,*,expires=None):
    _state.deadline=deadline
    _state.expires=expires
    _state.callback=callback
    _state.published=False


def active():
    return getattr(_state,'callback',None) is not None


def available(reserve=0.):
    expires=getattr(_state,'expires',None)
    if expires is not None and time.perf_counter()+reserve>=expires:return False
    deadline=getattr(_state,'deadline',None)
    return (not getattr(_state,'published',False) or deadline is None
            or time.perf_counter()+reserve<deadline)


def remaining():
    """Native-solver limit: never keep solving an unpublishable observation."""
    ends=[getattr(_state,'expires',None)]
    if getattr(_state,'published',False):ends.append(getattr(_state,'deadline',None))
    ends=[t for t in ends if t is not None]
    return max(0.,min(ends)-time.perf_counter()) if ends else float('inf')


def publish(command,loss=None):
    callback=getattr(_state,'callback',None)
    expires=getattr(_state,'expires',None)
    if callback is not None and (expires is None or time.perf_counter()<expires):
        callback(command,loss)
        _state.published=True
