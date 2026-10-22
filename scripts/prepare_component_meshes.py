"""Development collision cache from connected source components.

Each hull retains exact source vertices, so its support function equals the
component's. Concavities inside each component are filled: this is still an
approximation, not an exact collision model. It avoids voxel protrusions.
"""
from pathlib import Path
import hashlib
import json
import shutil
import xml.etree.ElementTree as ET
import numpy as np
import trimesh


def main():
    root=Path(__file__).resolve().parents[1]
    source=root/'assets/robots/holon/holon.urdf'
    cache=root/'build/collision_components';cache.mkdir(exist_ok=True)
    records={}
    for link in ET.parse(source).findall('link'):
        mesh=link.find('collision/geometry/mesh')
        if mesh is None:continue
        path=source.parent/mesh.get('filename');digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in records:continue
        if 'dock_mech' not in link.get('name'):
            shutil.copytree(root/'build/collision_fast'/digest,cache/digest,dirs_exist_ok=True)
            records[digest]=json.loads((cache/digest/'manifest.json').read_text());continue
        original=trimesh.load_mesh(path);parts=original.split(only_watertight=False,repair=False)
        target=cache/digest;target.mkdir(exist_ok=True);files=[];flat=[]
        for part in parts:
            if len(part.vertices)<4 or np.linalg.matrix_rank(part.vertices-part.vertices[0],tol=1e-10)<3:
                flat.append(part.vertices);continue
            hull=part.convex_hull
            name=f'{len(files)}.obj';out=target/name;out.write_text(hull.export(file_type='obj'))
            files.append(dict(file=name,sha256=hashlib.sha256(out.read_bytes()).hexdigest()))
        # Degenerate disconnected CAD faces have no solid volume. Record rather
        # than silently claiming exact geometry. Their global extrema are audited.
        record=dict(source=str(path.relative_to(root)),source_sha256=digest,
                    settings=dict(method='source_connected_component_hulls_v1'),parts=files,
                    source_components=len(parts),zero_volume_components=len(flat),
                    note='source support planes preserved for volumetric components; concavities filled')
        (target/'manifest.json').write_text(json.dumps(record,indent=2)+'\n');records[digest]=record
        print(path.name,len(files),'hulls',len(flat),'zero-volume components',flush=True)
    (cache/'manifest.json').write_text(json.dumps(records,indent=2)+'\n')


if __name__=='__main__':main()
