"""Conservative, fixed-configuration geometry checks using authored meshes.

Convex hull separation proves mesh separation at this configuration only.
Hull overlap is a rejection/unknown, not proof the original meshes intersect.
The authored exception manifest is hash-checked; no new blanket exclusions.
"""
from pathlib import Path
import hashlib
import json
import time
import xml.etree.ElementTree as ET
import numpy as np
import pinocchio as pin
import coal
from .model import ROOT, transform, PORTS


class Geometry:
    def __init__(self):
        begin=time.perf_counter()
        urdf=ROOT/'assets/robots/holon/holon.urdf'
        self.manifest=json.loads((ROOT/'configs/robot/holon_nominal_collision_pairs_v1.json').read_text())
        if hashlib.sha256(urdf.read_bytes()).hexdigest()!=self.manifest['urdf_sha256']:
            raise ValueError('stale collision exclusion manifest')
        tree=ET.parse(urdf).getroot();self.shapes={};self.entries=[]
        linkorder=[];hashes=[]
        loader=coal.MeshLoader()
        for link in tree.findall('link'):
            for collider in link.findall('collision'):
                mesh=collider.find('geometry/mesh')
                if mesh is None:raise ValueError('unsupported collision geometry')
                path=urdf.parent/mesh.get('filename');raw=path.read_bytes();digest=hashlib.sha256(raw).hexdigest()
                scale=np.fromstring(mesh.get('scale','1 1 1'),sep=' ')
                key=(digest,tuple(scale))
                if key not in self.shapes:
                    bvh=loader.load(str(path),scale)
                    bvh.buildConvexHull(True,'Qt') # return value tests original convexity, not construction success
                    if bvh.convex is None:raise ValueError('convex hull failure')
                    hull=bvh.convex;hull.computeLocalAABB();self.shapes[key]=hull
                self.entries.append((link.get('name'),transform(collider.find('origin')),self.shapes[key]))
                linkorder.append(link.get('name'));hashes.append(digest)
        if linkorder!=self.manifest['collision_link_ids'] or hashes!=self.manifest['mesh_sha256']:
            raise ValueError('mesh order/content mismatches exclusion manifest')
        self.allowed={tuple(p) for p in self.manifest['authored_overlap_geometry_pairs']}
        # Same rigid body and directly joined parts are mechanically adjacent;
        # source manifest only lists additional nonadjacent authored overlaps.
        self.adjacent=set()
        bychild={j.find('child').get('link'):j.find('parent').get('link') for j in tree.findall('joint')}
        for a,parent in bychild.items():self.adjacent.add(frozenset([a,parent]))
        self.preparation_s=time.perf_counter()-begin

    def check(self,model,meta,q,clearance=.005,query_limit=20000):
        start=time.perf_counter();data=model.createData();pin.framesForwardKinematics(model,data,q)
        objects=[]
        for module in range(meta['modules']):
            for index,(link,placement,shape) in enumerate(self.entries):
                fid=meta['frames'][f'm{module}/{link}'];T=data.oMf[fid]*placement
                center=(shape.aabb_local.min_+shape.aabb_local.max_)/2
                half=(shape.aabb_local.max_-shape.aabb_local.min_)/2
                wc=T.act(center);wh=np.abs(T.rotation)@half
                objects.append((module,index,link,fid,shape,T,wc-wh,wc+wh))
        seams=set()
        for e in meta['edges']:
            def mechanism(port):
                return ('pitch_dock_mech' if port.startswith('pitch') else 'yaw_dock_mech')+port[-1]
            seams.add(frozenset([(e['a'],mechanism(e['port_a'])),(e['b'],mechanism(e['port_b']))]))
        bad=[];queries=0;pruned=0;excluded=0
        request=coal.DistanceRequest();request.enable_signed_distance=True
        for i,a in enumerate(objects):
            for b in objects[i+1:]:
                same=a[0]==b[0]
                if same and (model.frames[a[3]].parentJoint==model.frames[b[3]].parentJoint or
                    frozenset([a[2],b[2]]) in self.adjacent or (a[1],b[1]) in self.allowed):
                    excluded+=1;continue
                if frozenset([(a[0],a[2]),(b[0],b[2])]) in seams:
                    excluded+=1;continue
                separation=np.maximum(np.maximum(a[6]-b[7],b[6]-a[7]),0.)
                if np.linalg.norm(separation)>=clearance:
                    pruned+=1;continue
                if queries>=query_limit:
                    return dict(status='unknown_query_limit',queries=queries,violations=bad,wall_s=time.perf_counter()-start)
                result=coal.DistanceResult()
                distance=coal.distance(a[4],coal.Transform3s(a[5].rotation,a[5].translation),b[4],coal.Transform3s(b[5].rotation,b[5].translation),request,result)
                queries+=1
                if not np.isfinite(distance) or distance<clearance:
                    bad.append(dict(a=f'm{a[0]}/{a[2]}',b=f'm{b[0]}/{b[2]}',convex_distance_m=float(distance)))
        return dict(status='clear_at_this_configuration' if not bad else 'convex_clearance_violation',
                    queries=queries,aabb_pruned=pruned,excluded_mechanical_pairs=excluded,violations=bad,
                    wall_s=time.perf_counter()-start,clearance_m=clearance,trajectory_certified=False)
