"""Save exact common pad triangles; never save a task-dependent structure."""
import numpy as np
from amsrr_det.model import ROOT,PORTS
from amsrr_det.mujoco_model import CONTACT_CACHE
from amsrr_det.contact_geometry import material_triangles,material_cache_identity


def main():
    entries={}
    for cache in [None,str(CONTACT_CACHE.resolve())]:
        for port in PORTS:
            entries[material_cache_identity(port,cache)]=material_triangles(port,cache,use_cache=False)
    destination=ROOT/'assets/generated/material_surfaces.npz'
    np.savez(destination,**entries)
    print(f'Saved {len(entries)} authored pad surfaces to {destination}')


if __name__=='__main__':main()
