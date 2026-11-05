"""Deadline-bounded improvement of an already screened periodic plan.

The independent structure trials remain unchanged. Only a fully rechecked
path with lower cost may replace their best result; the old result survives
an unfinished auxiliary computation.
"""

import multiprocessing as mp
import os, time, signal
from functools import partial
import numpy as np
from .walking import foot_link
from .walking_control import GroundControl
from .walking_direction import optimize_direction
from .walking_gait import validate_cycle
from .walking_plan import screen_motion
from .walking_landing import screen_landing


def _improve(pipe, plant, plan, spec, deadline, cpu, cpus, mode="both"):
    # The deadline owner can stop this worker and all phase-check children.
    os.setsid()
    if cpu is not None:
        os.sched_setaffinity(0, {cpu})
    control = GroundControl(
        plant,
        np.asarray(plan["support_pose"]),
        [foot_link(*s) for s in plan["surfaces"]],
        foot_points=plan["gait"]["foot_points"],
    )
    screen = partial(screen_motion, cpus=cpus)
    best = float(plan["cost"])
    current = plan

    def inspect(gait, iterations):
        nonlocal best
        if time.perf_counter() >= deadline:
            return
        try:
            geometry = validate_cycle(plant, gait, spec=spec)
            if not geometry["passed"]:
                return
            report = screen(plant, control, spec, gait)
        except ValueError:
            return
        if report["cost"] < best - 1e-7 and time.perf_counter() < deadline:
            try:
                landing=screen_landing(plant,dict(plan,gait=gait),spec)
            except ValueError:
                return
            if time.perf_counter()>=deadline:return
            best = report["cost"]
            pipe.send(
                dict(
                    gait=gait,
                    geometry=geometry,
                    report=report,
                    cost=best,
                    iterations=iterations,landing=landing,
                )
            )

    try:
        # Keep the established full-then-condensed improvement. A separate
        # condensed branch starts immediately so large full solves cannot
        # consume all of the remaining planning budget.
        from .walking_vectoring import _direction_iterations

        gait = {
            k: np.asarray(v) if isinstance(v, list) else v
            for k, v in plan["gait"].items()
        }
        if mode != "condensed":
            try:
                new, info = _direction_iterations(
                    plant, control, spec, gait, screen, 3, improve=True
                )
                report = info.get("accepted_report")
                if (
                    report is not None
                    and report["cost"] < best - 1e-7
                    and time.perf_counter() < deadline
                ):
                    landing=screen_landing(plant,dict(plan,gait=new),spec)
                    best = report["cost"]
                    current = dict(
                        plan, gait=new, geometry=info["geometry"], report=report, cost=best
                    )
                    pipe.send(
                        dict(
                            gait=new,
                            geometry=info["geometry"],
                            report=report,
                            cost=best,
                            iterations=3,landing=landing,
                        )
                    )
            except ValueError:
                pass
        if mode != "full":
            optimize_direction(
                plant,
                current,
                spec,
                iterations=8,
                budget=max(0.0, deadline - time.perf_counter()),
                inspect=inspect,
            )
    except ValueError:
        pass
    finally:
        pipe.close()


def refine(plant, plan, spec, *, deadline, cpu=None, cpus=None):
    start = time.perf_counter()
    original = float(plan["cost"])
    updates = []
    if deadline - start < 0.9:
        return plan, dict(
            status="insufficient_remaining_time", seconds=0.0, original_cost=original
        )
    context = mp.get_context("fork")
    from multiprocessing.connection import wait
    available=sorted(cpus or ({cpu} if cpu is not None else os.sched_getaffinity(0)))
    modes=["both","condensed"] if len(available)>=2 else ["both"]
    groups=[{int(c) for c in g} for g in np.array_split(available,len(modes))]
    workers=[];readers=[];selected=plan
    for mode,group in zip(modes,groups):
        reader,writer=context.Pipe(duplex=False)
        process=context.Process(target=_improve,args=(writer,plant,plan,spec,deadline,min(group),group,mode))
        process.start();writer.close();workers.append(process);readers.append(reader)
    try:
        while readers and time.perf_counter()<deadline:
            for reader in wait(readers,timeout=min(.02,max(0.,deadline-time.perf_counter()))):
                try:update=reader.recv()
                except EOFError:
                    readers.remove(reader);reader.close();continue
                if time.perf_counter()<deadline and update['cost']<selected['cost']:
                    updates.append(dict(iterations=update.pop('iterations'),cost=update['cost']))
                    selected=dict(plan,**update)
    finally:
        for process in workers:
            if process.is_alive():
                try:os.killpg(process.pid,signal.SIGTERM)
                except ProcessLookupError:process.terminate()
            process.join()
        for reader in readers:reader.close()
    return selected, dict(
        status="improved" if updates else "original_retained",
        original_cost=original,
        final_cost=float(selected["cost"]),
        updates=updates,modes=modes,worker_exitcodes=[p.exitcode for p in workers],
        seconds=time.perf_counter() - start,
        evidence_scope="Auxiliary updates are not independent trials. The next basic trial improvement probability bound is conservative at this lower cost.",
    )
