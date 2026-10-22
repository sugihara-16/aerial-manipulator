"""Cache convex decomposition of each distinct *part*, never whole structures.

Run in its own process: CoACD and Pinocchio's bundled native libraries conflict
when imported together on this host. All source vertices remain in metres.
"""
from pathlib import Path
import argparse
import hashlib
import json
import time
import xml.etree.ElementTree as ET
import coacd
import numpy as np
import trimesh


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, default=Path('build/collision_meshes'))
    p.add_argument('--threshold', type=float, default=.04)
    p.add_argument('--max-hulls', type=int, default=32)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = root / 'assets/robots/holon/holon.urdf'
    args.output.mkdir(parents=True, exist_ok=True)
    settings = dict(threshold=args.threshold, preprocess_mode='auto',
                    preprocess_resolution=50, resolution=1000, max_convex_hull=args.max_hulls,
                    mcts_iterations=40, mcts_nodes=20, mcts_max_depth=3,
                    merge=True, seed=0)
    coacd.set_log_level('warn')
    records = {}
    for mesh in ET.parse(source).findall('link/collision/geometry/mesh'):
        path = source.parent / mesh.get('filename')
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in records:
            continue
        target = args.output / digest
        record_path = target / 'manifest.json'
        if record_path.exists():
            record = json.loads(record_path.read_text())
            if record['settings'] == settings:
                records[digest] = record
                continue
        target.mkdir(exist_ok=True)
        shape = trimesh.load_mesh(path)
        start = time.perf_counter()
        print('decomposing', path.name, flush=True)
        parts = coacd.run_coacd(coacd.Mesh(np.asarray(shape.vertices),
                                         np.asarray(shape.faces, dtype=np.int32)),
                                **settings)
        files = []
        for i, (vertices, faces) in enumerate(parts):
            part = target / f'{i}.obj'
            part.write_text(trimesh.Trimesh(vertices, faces, process=False).export(file_type='obj'))
            files.append(dict(file=part.name,
                              sha256=hashlib.sha256(part.read_bytes()).hexdigest()))
        record = dict(source=str(path.relative_to(root)), source_sha256=digest,
                      settings=settings, parts=files,
                      wall_s=time.perf_counter()-start,
                      note='approximate decomposition; threshold is not a certified distance bound')
        record_path.write_text(json.dumps(record, indent=2)+'\n')
        records[digest] = record
        print(path.name, len(parts), 'parts', round(record['wall_s'], 2), 's', flush=True)
    (args.output / 'manifest.json').write_text(json.dumps(records, indent=2)+'\n')


if __name__ == '__main__':
    main()
