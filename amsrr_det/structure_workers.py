"""Share immutable geometry on Linux before any numerical worker starts."""
import multiprocessing as mp
from pathlib import Path
import sys


def planning_context():
    # Fork only a single-threaded parent. After control/solver threads exist,
    # use fresh interpreters so their locks cannot be inherited into a child.
    if sys.platform!='linux' or len(list(Path('/proc/self/task').iterdir()))!=1:
        return mp.get_context('spawn')
    from .structure_groups import skeleton_groups
    from .grasp_geometry import part_bounds
    from .contact_geometry import material_triangles
    from .mujoco_model import CONTACT_CACHE,_part_templates
    from .model import PORTS
    skeleton_groups();part_bounds();_part_templates(str(CONTACT_CACHE.resolve()))
    for port in PORTS:material_triangles(port,str(CONTACT_CACHE.resolve()))
    return mp.get_context('fork')
