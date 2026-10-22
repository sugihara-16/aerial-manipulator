"""Preserve contact-part concavities, without voxel remeshing of the work face.

Offline per-part cache only. Planar CAD boundary loops are explicitly recorded;
no structure or task outcome is cached. Run separately from Pinocchio/CoACD.
"""
from pathlib import Path
import hashlib,json,shutil,time,xml.etree.ElementTree as ET
import numpy as np,trimesh,networkx as nx,coacd


def close_planar_loops(mesh):
    edges,count=np.unique(mesh.edges_sorted,axis=0,return_counts=True)
    graph=nx.from_edgelist(edges[count==1]);added=[];loops=[]
    for component in nx.connected_components(graph):
        sub=graph.subgraph(component)
        if any(degree!=2 for _,degree in sub.degree):continue
        cycle=nx.cycle_basis(sub)[0];v=mesh.vertices[cycle];_,singular,V=np.linalg.svd(v-v.mean(axis=0),full_matrices=False)
        if singular[-1]>1e-4:continue
        points=(v-v.mean(axis=0))@V[:2].T
        cross=lambda a,b:float(a[0]*b[1]-a[1]*b[0])
        area=sum(cross(a,b) for a,b in zip(points,np.roll(points,-1,axis=0)))
        remaining=list(range(len(cycle)));triangles=[]
        if area<0:remaining.reverse()
        while len(remaining)>3:
            for i,b in enumerate(remaining):
                a,c=remaining[i-1],remaining[(i+1)%len(remaining)]
                if cross(points[b]-points[a],points[c]-points[b])<=1e-18:continue
                others=[p for p in remaining if p not in [a,b,c]]
                if any(all(cross(points[y]-points[x],points[p]-points[x])>=-1e-18 for x,y in [(a,b),(b,c),(c,a)]) for p in others):continue
                triangles.append([cycle[a],cycle[b],cycle[c]]);remaining.pop(i);break
            else:raise ValueError('could not triangulate CAD boundary')
        triangles.append([cycle[i] for i in remaining]);added.extend(triangles)
        loops.append(dict(vertices=len(cycle),extent_m=np.ptp(v,axis=0).tolist(),max_plane_error_m=float(np.max(abs((v-v.mean(axis=0))@V[-1])))))
    if added:mesh.faces=np.vstack([mesh.faces,np.array(added)]);mesh.fix_normals()
    return loops


def main():
    root=Path(__file__).resolve().parents[1];source=root/'assets/robots/holon/holon.urdf'
    cache=root/'build/collision_contacts';cache.mkdir(exist_ok=True);records={}
    settings=dict(threshold=.01,preprocess_mode='off',resolution=3000,max_convex_hull=-1,
                  mcts_iterations=40,mcts_nodes=10,mcts_max_depth=3,merge=True,seed=194)
    coacd.set_log_level('warn')
    for link in ET.parse(source).findall('link'):
        mesh=link.find('collision/geometry/mesh')
        if mesh is None:continue
        path=source.parent/mesh.get('filename');digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in records:continue
        target=cache/digest;target.mkdir(exist_ok=True)
        if 'dock_mech' not in link.get('name'):
            shutil.copytree(root/'build/collision_components'/digest,target,dirs_exist_ok=True)
            records[digest]=json.loads((target/'manifest.json').read_text());continue
        recordfile=target/'manifest.json'
        if recordfile.exists():records[digest]=json.loads(recordfile.read_text());continue
        original=trimesh.load_mesh(path);components=original.split(only_watertight=False,repair=False)
        files=[];repairs=[];start=time.perf_counter();volumetric=0
        for ci,part in enumerate(components):
            if len(part.vertices)<4 or np.linalg.matrix_rank(part.vertices-part.vertices[0],tol=1e-10)<3:continue
            hull=part.convex_hull;volumetric+=1
            # Small screw details retain the earlier conservative hull. Large
            # concave pieces include the work faces and must keep their cavities.
            if np.max(part.extents)>.06 and part.volume/hull.volume<.98:
                loops=close_planar_loops(part);repairs.append(dict(component=ci,loops=loops))
                if not part.is_watertight:raise ValueError(f'{path.name} component {ci} still open')
                print('decomposing',path.name,ci,len(part.faces),flush=True)
                outputs=coacd.run_coacd(coacd.Mesh(np.asarray(part.vertices),np.asarray(part.faces,dtype=np.int32)),**settings)
                pieces=[trimesh.Trimesh(v,f,process=False) for v,f in outputs]
            else:pieces=[hull]
            for piece in pieces:
                out=target/f'{len(files)}.obj';out.write_text(piece.export(file_type='obj'))
                files.append(dict(file=out.name,sha256=hashlib.sha256(out.read_bytes()).hexdigest(),source_component=ci))
        record=dict(source=str(path.relative_to(root)),source_sha256=digest,settings=settings,parts=files,
                    planar_boundary_repairs=repairs,source_components=len(components),zero_volume_components=len(components)-volumetric,
                    wall_s=time.perf_counter()-start,note='large concavities decomposed without remeshing; small mechanical details retain conservative hulls')
        recordfile.write_text(json.dumps(record,indent=2));records[digest]=record
        print(path.name,len(files),'parts',record['wall_s'],flush=True)
    (cache/'manifest.json').write_text(json.dumps(records,indent=2))

if __name__=='__main__':main()
