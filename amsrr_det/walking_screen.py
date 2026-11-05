"""Exact periodic load and disturbance checks, optionally split by phase."""

import multiprocessing as mp, os
import numpy as np, pinocchio as pin
from amsrr_det.walking_cycle import CycleReference
from amsrr_det.walking_load import (
    support_switch_times,
    support_quadrature,
    motion_load,
    ground_contacts,
)
from amsrr_det.walking import rotation_z
from amsrr_det.equilibrium import static_balance, certify_load_changes


def _map_rows(fn, rows, cpus):
    if not cpus or len(cpus) < 2:
        return list(map(fn, rows))
    count = min(4, len(rows), len(cpus))
    context = mp.get_context("fork")
    children = []
    pipes = []

    def work(pipe, indices):
        os.sched_setaffinity(0, cpus)
        try:
            pipe.send((True, [(int(i), fn(rows[i])) for i in indices]))
        except Exception as error:
            pipe.send((False, str(error)))
        finally:
            pipe.close()

    try:
        for indices in np.array_split(np.arange(len(rows)), count):
            reader, writer = context.Pipe(False)
            p = context.Process(target=work, args=(writer, indices))
            p.start()
            writer.close()
            children.append(p)
            pipes.append(reader)
        results = []
        for reader in pipes:
            good, value = reader.recv()
            if not good:
                raise ValueError(value)
            results.extend(value)
        return [v for _, v in sorted(results)]
    finally:
        for p in children:
            if p.is_alive():
                p.terminate()
            p.join()
        for pipe in pipes:
            pipe.close()


def screen_motion(plant, controller, spec, gait, nominal_scores=None, *, cpus=None):
    ref = CycleReference(plant, spec, gait)
    samples = [
        (t, i % 4 == 0, 0.0) for i, t in enumerate(np.linspace(0, ref.period, 97)[:-1])
    ]
    transitions = support_switch_times(
        ref.period, len(gait["feet"]), gait.get("duty", 0.62), gait.get("phases")
    )
    samples += [(t, True, 0.0) for t in transitions]
    peak = 0.0
    if nominal_scores is not None:
        scores = np.asarray(nominal_scores)
        if scores.shape != (96,) or not np.isfinite(scores).all() or scores.max() > 0.9:
            raise ValueError("invalid_nominal_screen")
        peak = float(scores.max())
        samples = [row for row in samples if row[1]]
    times, quadrature = support_quadrature(
        ref.period, len(gait["feet"]), gait.get("duty", 0.62), gait.get("phases")
    )
    samples.extend((t, False, w) for t, w in zip(times, quadrature))
    weight = plant.meta["mass"] * 9.80665
    radius = max(np.linalg.norm(gait["feet"][:, :2] - gait["origin"][:2], axis=1))
    disturbance = np.diag([0.02 * weight] * 3 + [0.01 * weight * radius] * 3)

    def sample(row):
        t, check_margin, w = row
        q, v, a = ref.rates(t)
        qp, load = motion_load(plant, controller.drive, q, v, a)
        contacts = ground_contacts(controller, ref.feet(t)[3])
        workspace = {}
        report = static_balance(
            plant.pin,
            plant.meta,
            qp,
            contacts,
            generalized_load=load,
            fast_lp=True,
            workspace=workspace,
            minimize_burden=w > 0.0,
        )
        value = report.get("minimum_peak_effort_ratio", float("inf"))
        margin_peak = 0.0
        if value > 0.9:
            raise ValueError("periodic_actuator_load")
        if check_margin:
            R = pin.Quaternion(qp[3:7]).matrix()
            heading = rotation_z(spec.path(t)[1])
            changes = np.array(
                [
                    sign
                    * np.r_[
                        R.T @ heading @ force[:3],
                        R.T @ heading @ force[3:],
                        np.zeros(plant.pin.nv - 6),
                    ]
                    for force in disturbance
                    for sign in [-1.0, 1.0]
                ]
            )
            certificate = certify_load_changes(workspace, changes)
            for i, change in enumerate(changes):
                if certificate is not None and certificate[0][i]:
                    margin_peak = max(margin_peak, float(certificate[1][i]))
                    continue
                margin = static_balance(
                    plant.pin,
                    plant.meta,
                    qp,
                    contacts,
                    generalized_load=load + change,
                    fast_lp="simplex",
                    workspace=workspace,
                )
                p = margin.get("minimum_peak_effort_ratio", float("inf"))
                margin_peak = max(margin_peak, p)
                if p > 1.0:
                    raise ValueError("external_wrench_margin")
        if w == 0:
            return value, margin_peak, None
        rotor = (
            float(
                np.maximum(report["thrust"], 0.0) ** 1.5
                @ np.ones(len(report["thrust"]))
            )
            / weight**1.5
        )
        joint = float(np.mean([x["normalized"] ** 2 for x in report["joints"]]))
        ground = sum(x["normal_force"] for x in report["contacts"]) / weight
        return (
            value,
            margin_peak,
            (qp.copy(), load.copy(), contacts, rotor, joint, ground, w),
        )

    values = _map_rows(sample, sorted(samples), cpus)
    peak = max(peak, max(v[0] for v in values))
    margin_peak = max(v[1] for v in values)
    states = [v[2] for v in values if v[2] is not None]
    rotor_energy = np.array([v[3] for v in states])
    joint_energy = np.array([v[4] for v in states])
    ground = np.array([v[5] for v in states])
    energy_weights = np.array([v[6] for v in states])
    constrained_support = False
    if energy_weights @ ground < 0.52 or energy_weights @ (ground >= 0.52) < 0.6:

        def capacity(state):
            qp, load, contacts = state[:3]
            r = static_balance(
                plant.pin,
                plant.meta,
                qp,
                contacts,
                generalized_load=load,
                fast_lp=True,
                maximize_contact_load=True,
                peak_effort_bound=0.9,
            )
            return sum(c["normal_force"] for c in r.get("contacts", []))

        capacities = np.array(_map_rows(capacity, states, cpus))
        average = float(energy_weights @ capacities)
        if (
            average < 0.52 * weight
            or energy_weights @ (capacities >= 0.52 * weight) < 0.6
        ):
            raise ValueError("periodic_ground_capacity")

        def constrained(pair):
            capacity, state = pair
            qp, load, contacts = state[:3]
            r = static_balance(
                plant.pin,
                plant.meta,
                qp,
                contacts,
                generalized_load=load,
                fast_lp=True,
                minimize_burden=True,
                minimum_contact_normal_force_n=max(
                    capacity * 0.52 * weight / average,
                    (
                        min(capacity * 0.98, 0.55 * weight)
                        if capacity >= 0.52 * weight
                        else 0.0
                    ),
                ),
                peak_effort_bound=0.9,
            )
            if (
                r.get("status") != "static_screen_pass"
                or r.get("max_normalized_joint_effort", float("inf")) > 0.9 + 1e-7
            ):
                raise ValueError("ground_constrained_effort")
            return (
                sum(max(t, 0) ** 1.5 for t in r["thrust"]) / weight**1.5,
                float(np.mean([x["normalized"] ** 2 for x in r["joints"]])),
                sum(x["normal_force"] for x in r["contacts"]) / weight,
            )

        rotor_energy, joint_energy, ground = np.array(
            _map_rows(constrained, list(zip(capacities, states)), cpus)
        ).T
        constrained_support = True
    if energy_weights @ ground < 0.52 - 1e-8 or energy_weights @ (ground >= 0.5) < 0.6:
        raise ValueError("periodic_ground_support")
    motion_load(plant, controller.drive, *ref.rates(max(s[0] for s in samples)))
    return dict(
        peak=peak,
        margin_peak=margin_peak,
        cost=float(energy_weights @ (rotor_energy + 0.05 * joint_energy)),
        rotor_hover_ratio=float(
            energy_weights @ rotor_energy * np.sqrt(len(plant.meta["rotors"]))
        ),
        mean_normalized_joint_torque_squared=float(energy_weights @ joint_energy),
        margin_peak_kind="checked feasible upper bound, not necessarily the minimum",
        mean_ground_fraction=float(energy_weights @ ground),
        ground_half_time=float(energy_weights @ (ground >= 0.5)),
        support_samples=ground.tolist(),
        support_sample_weights=energy_weights.tolist(),
        support_constrained_allocation=constrained_support,
        nominal_samples=96 + len(transitions) + len(times),
        nominal_reused=nominal_scores is not None,
        margin_samples=24 + len(transitions),
        energy_samples=len(times),
        energy_integration="three-point Gauss quadrature in each constant support interval",
        margin_force_n=0.02 * weight,
        margin_moment_nm=0.01 * weight * radius,
        margin_frame="path_heading",
        margin_set="individual signed axis disturbances, not simultaneous box corners",
        objective="mean(sum(rotor_thrust**1.5)/weight**1.5 + .05*mean(normalized_joint_torque**2))",
    )
