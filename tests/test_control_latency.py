import time
import mujoco
import numpy as np
import pytest
from build.control_latency import ControlCycle


def plant():
    m = mujoco.MjModel.from_xml_string('''<mujoco>
      <option timestep="0.000125" gravity="0 0 0" integrator="Euler"/>
      <worldbody><body><joint name="x" type="slide" axis="1 0 0"/>
      <geom type="sphere" size=".1" mass="1"/></body></worldbody>
      <actuator><general joint="x" gainprm="1" biastype="affine" biasprm="0 0 0"/></actuator>
      </mujoco>''')
    d = mujoco.MjData(m)
    d.ctrl[:] = 1.
    def advance(n):
        for _ in range(n):mujoco.mj_step(m, d)
    return m, d, advance


def test_slow_computation_holds_old_input_and_skips_ticks():
    m,d,advance = plant()
    cycle = ControlCycle(m,d,.01)
    cycle.wait_for_result(.025,advance)
    assert d.time == pytest.approx(.025)
    assert d.qvel[0] == pytest.approx(.025)
    assert d.ctrl[0] == 1.
    assert cycle.skipped_ticks == 2
    cycle.apply(np.array([-1.]))
    advance(cycle.remaining_steps())
    assert d.time == pytest.approx(.03)
    assert d.qvel[0] == pytest.approx(.02)
    assert cycle.record('TEST')['application_s'] == pytest.approx(.025)


def test_servo_gains_are_not_applied_retroactively():
    m,d,advance = plant()
    cycle = ControlCycle(m,d,.01)
    m.actuator_gainprm[0,0] = 3.
    m.actuator_biasprm[0,0] = 2.
    cycle.wait_for_result(.005,advance)
    assert d.qvel[0] == pytest.approx(.005)
    cycle.apply(np.array([1.]))
    advance(cycle.remaining_steps())
    assert d.qvel[0] == pytest.approx(.03)


def test_violation_during_calculation_prevents_new_input():
    m,d,advance = plant()
    cycle = ControlCycle(m,d,.01)
    def checked(n):
        for _ in range(n):
            advance(1)
            if d.qvel[0] >= .01:return 'velocity_limit'
    assert cycle.wait_for_result(.1,checked) == 'velocity_limit'
    assert d.time < .011
    with pytest.raises(ValueError,match='physical violation'):cycle.apply(np.array([-1.]))
    assert d.ctrl[0] == 1.


def test_actual_wall_delay_advances_physical_time_even_below_one_period():
    m,d,advance = plant()
    cycle = ControlCycle(m,d,.01)
    start = time.perf_counter()
    time.sleep(.003)
    elapsed = time.perf_counter()-start
    cycle.wait_for_result(elapsed,advance)
    assert elapsed-1e-12 <= d.time < elapsed+m.opt.timestep+1e-12
    assert d.qvel[0] == pytest.approx(d.time)


@pytest.mark.parametrize('delay',[.0,.01,.02,.010001])
def test_sampling_grid_and_rounding(delay):
    m,d,advance = plant()
    cycle = ControlCycle(m,d,.01)
    cycle.wait_for_result(delay,advance)
    assert d.time+1e-12 >= delay
    cycle.apply(np.array([1.]))
    advance(cycle.remaining_steps())
    expected = .01 if delay <= .01 else .02
    assert d.time == pytest.approx(expected)


def test_ideal_mode_is_explicit_legacy_comparison():
    m,d,advance = plant()
    cycle = ControlCycle(m,d,.01,measured=False)
    cycle.wait_for_result(.5,advance)
    assert d.time == 0
    cycle.apply(np.array([-1.]))
    advance(cycle.remaining_steps())
    assert d.time == pytest.approx(.01)
    assert d.qvel[0] == pytest.approx(-.01)


def test_sampling_audit_detects_one_physics_step_of_lateness():
    from build.control_latency import sampling_intervals
    _,late=sampling_intervals([{'observation_s':t} for t in [0.,.01,.020125]])
    assert late


def test_sampling_audit_accepts_floating_point_time_accumulation():
    from build.control_latency import sampling_intervals
    _,late=sampling_intervals([{'observation_s':t} for t in [30.,30.01000000000005,30.0200000000001]])
    assert not late
