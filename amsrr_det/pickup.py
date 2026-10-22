"""Recheck a nominal transport path after observing a shifted pickup position."""
import time
import numpy as np
from .release import check_opening_clearance


def check_pickup_path(meta,shape,surfaces,closed,source,pickup,destination,lift_height,obstacles):
    start=time.perf_counter();oq=meta['object']['q']
    # Source scene and destination remain fixed in the world. Only the pickup
    # horizontal start changes. Actual tracking is audited independently.
    initial=np.array(closed,copy=True);shift=np.asarray(pickup[:3])-source[:3]
    if not np.array_equal(pickup[2:],source[2:]):raise ValueError('pickup capture may change horizontal position only')
    initial[:3]+=shift;initial[oq:oq+3]+=shift
    lifted=initial.copy();lifted[2]+=lift_height;lifted[oq+2]+=lift_height
    delta=destination[:3]-pickup[:3]
    carried=lifted.copy();carried[:3]+=delta;carried[oq:oq+3]+=delta
    placed=initial.copy();placed[:3]+=delta;placed[oq:oq+7]=destination
    evidence=check_opening_clearance(meta,shape,surfaces,[initial,lifted,carried,placed],obstacles)
    return dict(scope='nominal path interval geometry; not a contact/tracking certificate',
        source=np.asarray(source).tolist(),pickup=np.asarray(pickup).tolist(),destination=np.asarray(destination).tolist(),
        clearance=evidence,planning_s=time.perf_counter()-start)
