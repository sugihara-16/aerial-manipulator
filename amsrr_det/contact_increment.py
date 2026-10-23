"""Input-to-velocity increment of the existing one-period contact equations."""
import numpy as np
from scipy.linalg import cho_factor, cho_solve


def velocity_response(effective_mass, input_map, contact_map, compliance):
    """Eliminate reactions, allowing redundant rigid support constraints.

    Solve E dv - C.T df = B du and C dv + D df = 0. The
    velocity response is unique even when a support reaction is not.
    This local metric does not guarantee the next contact mode or peak load.
    """
    factor = cho_factor(effective_mass, lower=True, check_finite=False)
    free = cho_solve(factor, input_map, check_finite=False)
    reaction = cho_solve(factor, contact_map.T, check_finite=False)
    schur = contact_map @ reaction + compliance
    force = np.linalg.lstsq(schur, contact_map @ free, rcond=1e-12)[0]
    return free - reaction @ force
