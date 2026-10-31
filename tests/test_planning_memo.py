"""Duplicate numerical work may be shared, independent draws may not change."""
import multiprocessing as mp
from pathlib import Path
from amsrr_det.planning_memo import PlanningMemo
import pytest


def _evaluate(directory):
    def calculate():
        with (Path(directory)/'evaluations').open('a') as stream:
            stream.write('evaluated\n')
        return {'value': [7.]}
    return PlanningMemo(directory).evaluate({'physical_input': [1., 2.]}, calculate)


def test_simultaneous_duplicates_compute_once_and_load_independent_values(tmp_path):
    with mp.get_context('spawn').Pool(4) as pool:
        results = pool.map(_evaluate, [str(tmp_path)]*8)
    assert (tmp_path/'evaluations').read_text().splitlines() == ['evaluated']
    assert sum(not hit for _, hit in results) == 1
    results[0][0]['value'][0] = 99
    assert _evaluate(str(tmp_path)) == ({'value': [7.]}, True)


def test_exact_input_change_and_new_task_directory_recompute(tmp_path):
    memo = PlanningMemo(tmp_path)
    a, hit = memo.evaluate({'x': 1.}, lambda: 'first')
    b, hit2 = memo.evaluate({'x': 1.0000000000000002}, lambda: 'second')
    assert (a, b, hit, hit2) == ('first', 'second', False, False)
    other = tmp_path/'next_task'; other.mkdir()
    assert PlanningMemo(other).evaluate({'x': 1.}, lambda: 'third') == ('third', False)


def test_cancelled_work_is_not_returned_as_an_evaluation_or_reused_by_next_task(tmp_path):
    from amsrr_det.planning_memo import PlanningCancelled
    memo = PlanningMemo(tmp_path)
    assert memo.evaluate({'x': 1}, lambda: 7) == (7, False)
    (tmp_path/'cancelled').touch()
    for value in [1, 2]:
        with pytest.raises(PlanningCancelled):
            memo.evaluate({'x': value}, lambda: pytest.fail('cancelled calculation ran'))
    other = tmp_path/'next'; other.mkdir()
    assert PlanningMemo(other).evaluate({'x': 1}, lambda: 9) == (9, False)


def _cancel_at_checkpoint(directory, index):
    import time
    from amsrr_det import planning_memo
    planning_memo.active = PlanningMemo(directory)
    planning_memo.check_cancelled()
    time.sleep(.02)
    planning_memo.check_cancelled()
    return index


def test_process_workers_drain_cancelled_jobs_without_turning_them_into_failures(tmp_path):
    from amsrr_det.planning_memo import PlanningCancelled
    with mp.get_context('spawn').Pool(2) as pool:
        completed = pool.apply_async(_cancel_at_checkpoint, (str(tmp_path), 0))
        assert completed.get(timeout=10) == 0
        (tmp_path/'cancelled').touch()
        pending = [pool.apply_async(_cancel_at_checkpoint, (str(tmp_path), i)) for i in range(1, 12)]
        pool.close(); pool.join()
        for future in pending:
            with pytest.raises(PlanningCancelled):future.get(timeout=1)


def test_reused_physical_models_preserve_dynamics_and_do_not_share_mutations(tmp_path,monkeypatch):
    import numpy as np
    import mujoco
    from amsrr_det import planning_memo
    from amsrr_det.mujoco_model import make_plant
    kwargs=dict(n=2,edges=[dict(a=0,port_a='pitch1',b=1,port_b='yaw2')],
                payload=dict(shape='ellipsoid',size=[.3,.2,.18],mass=.7),meshes=False)
    reference=make_plant(**kwargs)
    monkeypatch.setattr(planning_memo,'active',PlanningMemo(tmp_path))
    first=make_plant(**kwargs)
    first.mj.body_mass[:]=0.;first.meta['mass']=-1.;first.mj.opt.timestep=1.
    loaded=make_plant(**kwargs)
    assert loaded.meta['mass']==reference.meta['mass']
    assert loaded.mj.opt.timestep==reference.mj.opt.timestep
    for name in dir(reference.mj):
        value=getattr(reference.mj,name)
        if isinstance(value,np.ndarray):np.testing.assert_array_equal(getattr(loaded.mj,name),value)
    states=[]
    for plant in [reference,loaded]:
        d=mujoco.MjData(plant.mj);d.qpos[2]=2.;d.ctrl[-8:]=3.
        mujoco.mj_step(plant.mj,d,nstep=10);states.append(d)
    for name in ['qpos','qvel','qfrc_actuator']:
        np.testing.assert_array_equal(getattr(states[0],name),getattr(states[1],name))


def test_excluded_count_cancels_only_that_count_without_returning_a_cost(tmp_path):
    import pytest
    from amsrr_det.planning_memo import PlanningCancelled
    memo=PlanningMemo(tmp_path);memo.task_group=3
    (tmp_path/'excluded_3').touch()
    with pytest.raises(PlanningCancelled):memo.evaluate({'x':1},lambda: 7)
    memo.task_group=2
    assert memo.evaluate({'x':1},lambda: 7)==(7,False)
