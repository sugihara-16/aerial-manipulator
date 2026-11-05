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


def _improve(pipe, plant, plan, spec, deadline, cpu, cpus):
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
            best = report["cost"]
            pipe.send(
                dict(
                    gait=gait,
                    geometry=geometry,
                    report=report,
                    cost=best,
                    iterations=iterations,
                )
            )

    try:
        # Check the existing full-angle update first: a small-angle update can
        # otherwise spend its whole allowance on an obstructed linearization.
        from .walking_vectoring import _direction_iterations

        gait = {
            k: np.asarray(v) if isinstance(v, list) else v
            for k, v in plan["gait"].items()
        }
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
                        iterations=3,
                    )
                )
        except ValueError:
            pass
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
    reader, writer = context.Pipe(duplex=False)
    process = context.Process(
        target=_improve, args=(writer, plant, plan, spec, deadline, cpu, cpus)
    )
    process.start()
    writer.close()
    selected = plan
    try:
        while time.perf_counter() < deadline:
            if reader.poll(min(0.02, max(0.0, deadline - time.perf_counter()))):
                try:
                    update = reader.recv()
                except EOFError:
                    break
                if update["cost"] < selected["cost"]:
                    updates.append(
                        dict(iterations=update.pop("iterations"), cost=update["cost"])
                    )
                    selected = dict(plan, **update)
            elif not process.is_alive():
                break
    finally:
        if process.is_alive():
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                process.terminate()
        process.join()
        reader.close()
    return selected, dict(
        status="improved" if updates else "original_retained",
        original_cost=original,
        final_cost=float(selected["cost"]),
        updates=updates,
        seconds=time.perf_counter() - start,
        evidence_scope="Auxiliary updates are not independent trials. The next basic trial improvement probability bound is conservative at this lower cost.",
    )
