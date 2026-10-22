"""Optional cache of authored single-part bounds, never assembled structures."""
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import coal
import numpy as np
import pinocchio as pin
from .model import ROOT

CACHE = ROOT / 'assets/generated/authored_part_bounds.json'


def identity():
    urdf = ROOT / 'assets/robots/holon/holon.urdf'
    files = {urdf, ROOT / 'configs/robot/holon_nominal_collision_pairs_v1.json'}
    files.update(Path(__file__).with_name(name) for name in
                 ['part_bounds_cache.py', 'geometry.py', 'grasp_geometry.py', 'model.py'])
    files.update((urdf.parent / mesh.get('filename')).resolve()
                 for mesh in ET.fromstring(urdf.read_bytes()).iter('mesh'))
    return dict(files={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sorted(files)},
                versions=dict(coal=coal.__version__, numpy=np.__version__, pinocchio=pin.__version__))


def read_bounds():
    try:
        saved = json.loads(CACHE.read_text())
        if saved['inputs'] != identity():
            return None
        rows = [(row['link'], np.array(row['center']), np.array(row['half']))
                for row in saved['bounds']]
        if not rows or any(c.shape != (3,) or h.shape != (3,) or
                           not np.isfinite(np.r_[c, h]).all() or np.any(h < 0)
                           for _, c, h in rows):
            return None
        return rows
    except (OSError, ValueError, KeyError, TypeError):
        return None
