import numpy as np
import pytest
from amsrr_det.path_progress import ScalarProgress,stopping_distance


def test_stopping_distance_matches_independent_held_input_integration():
    for speed in [.0,.001,.02,.113,1.]:
        distance=0.;v=speed;h=.01;b=.077
        while v>1e-14:
            a=-min(b,v/h);distance+=h*v+.5*h*h*a;v+=h*a
        assert abs(distance-stopping_distance(speed,b,h))<1e-11


def test_progress_brakes_pauses_and_resumes_without_reversing_or_overshooting():
    p=ScalarProgress(.48,6.);paused=False;resumed=False
    for step in range(1500):
        if step==100:p.pause();paused=True
        if step==400:
            assert p.speed<1e-8 and p.position<p.distance
            p.resume();resumed=True
        acceleration=p.choose((-1.,1.))
        previous=p.position;p.advance(acceleration)
        assert previous-1e-10<=p.position<=p.distance+1e-9
        assert p.speed<=p.maximum_speed+1e-10
        assert p.position+stopping_distance(p.speed,p.peak,p.period)<=p.distance+1e-9
        if p.finished:break
    assert paused and resumed and p.finished


def test_current_feasibility_is_not_called_a_safe_hold():
    p=ScalarProgress(.5,6.)
    with pytest.raises(ValueError,match='attainable'):p.choose(None)
    p.position=.49;p.speed=.12
    with pytest.raises(ValueError,match='stopping'):p.choose((-.001,.1))
