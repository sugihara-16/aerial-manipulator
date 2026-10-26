"""Routing belongs to the fixed trial, not to its scheduling history."""
import io
import threading
import numpy as np
import pytest
from amsrr_det.remote_structure_workers import remote_choice, send, receive


def test_routing_is_order_independent_and_keeps_both_machines():
    keys=[(n,k) for n in range(1,9) for k in range(59)]
    first={(n,k):remote_choice(2026102404,n,k,12/44) for n,k in keys}
    order=np.random.default_rng(2).permutation(len(keys))
    later={keys[i]:remote_choice(2026102404,*keys[i],12/44) for i in order}
    assert first==later
    assert any(first.values()) and not all(first.values())
    assert not any(remote_choice(1,n,k,0.) for n,k in keys)
    assert all(remote_choice(1,n,k,1.) for n,k in keys)


def test_framed_candidates_and_truncated_response():
    stream=io.BytesIO();candidate=dict(q=np.arange(7,dtype=float),surfaces=[(0,'yaw1')])
    send(stream,candidate,threading.Lock());raw=stream.getvalue()
    received=receive(io.BytesIO(raw))
    np.testing.assert_array_equal(received['q'],candidate['q'])
    assert received['surfaces']==candidate['surfaces']
    with pytest.raises(EOFError):receive(io.BytesIO(raw[:-1]))


def test_shutdown_waits_for_both_machines_without_serializing_them():
    from types import SimpleNamespace
    import threading
    from amsrr_det.remote_structure_workers import RemoteGenerationPool
    stopped=threading.Event();closed=[]
    def close():
        assert stopped.wait(timeout=1.)
        closed.append(True)
    pool=RemoteGenerationPool.__new__(RemoteGenerationPool)
    pool.backend=SimpleNamespace(local=SimpleNamespace(close=stopped.set,join=lambda:None),
                                 remote=SimpleNamespace(close=close))
    pool.close()
    assert closed == [True]
