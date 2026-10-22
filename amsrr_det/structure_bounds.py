"""Necessary thrust-cost bounds for every topology and contact assignment.

The criterion is sum(thrust**1.5), an ideal induced-power proxy, not measured
electrical power. These bounds apply to airborne zero-COM-acceleration balance;
they are not certificates for a complete transport trajectory.
"""
import numpy as np


def hover_input_burden(thrust):
    force = np.asarray(thrust, dtype=float)
    if force.ndim != 1 or not np.isfinite(force).all() or np.any(force < 0):
        raise ValueError('thrust must be a finite nonnegative vector')
    return float(np.sum(force ** 1.5))


def count_lower_bounds(module_mass, payload_mass, gravity, *, maximum_modules=8,
                       rotors_per_module=4,force_balance_tolerance=0.):
    """Triangle inequality plus Jensen: J >= W**1.5 / sqrt(rotor_count).

    Dropping direction, moment, contact, and joint constraints enlarges the
    feasible set. The bound therefore covers every legal structure and pose
    of a module count, including contact assignments not explicitly generated.
    No monotonicity in module count is assumed. Two free bodies (connected
    robot and object) each permit three translational residuals bounded by the
    supplied tolerance; relax total weight by 2*sqrt(3)*tolerance accordingly.
    """
    if (not np.isfinite([module_mass, payload_mass, gravity]).all()
            or module_mass <= 0 or payload_mass < 0 or gravity <= 0):
        raise ValueError('invalid physical masses or gravity magnitude')
    if (not isinstance(maximum_modules, int) or not 1 <= maximum_modules <= 8
            or not isinstance(rotors_per_module, int) or rotors_per_module < 1):
        raise ValueError('invalid module or rotor count')
    if not np.isfinite(force_balance_tolerance) or force_balance_tolerance<0:
        raise ValueError('invalid balance tolerance')
    return [dict(modules=n, lower_bound=float(
        max(0.,(n*module_mass+payload_mass)*gravity-2*np.sqrt(3)*force_balance_tolerance)**1.5 / np.sqrt(n*rotors_per_module)),
        force_balance_tolerance=force_balance_tolerance)
        for n in range(1, maximum_modules+1)]


def compare_count_groups(bounds, incumbent_burden):
    """Keep unresolved improvements explicit; a low bound is NOT rejection.

    This is a component of a selector, not its stopping criterion. Groups that
    can still beat the incumbent require further reasoning or exploration.
    """
    if not np.isfinite(incumbent_burden) or incumbent_burden < 0:
        raise ValueError('invalid feasible incumbent cost')
    rows = []
    for item in bounds:
        lower = item['lower_bound']
        if not np.isfinite(lower) or lower < 0:
            raise ValueError('invalid group lower bound')
        # A numerical guard avoids claiming exclusion from a roundoff tie.
        excluded = lower > incumbent_burden + 1e-8*(1+incumbent_burden)
        rows.append(dict(item, status='excluded_by_weight' if excluded else 'unresolved',
                         maximum_improvement=max(0., incumbent_burden-lower)))
    return rows
