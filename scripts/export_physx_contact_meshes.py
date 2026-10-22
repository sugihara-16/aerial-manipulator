"""Offline export of the source's PhysX convex decomposition, independent of mainline."""
from pathlib import Path
import hashlib,json,time,shutil,xml.etree.ElementTree as ET
import numpy as np
from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'hide_ui':True,'disable_viewport_updates':True})
from pxr import Usd,UsdGeom,UsdPhysics,UsdUtils,PhysicsSchemaTools,Sdf,Gf
import omni.usd
from omni.physx import get_physx_cooking_interface
from omni.physx.bindings._physx import PhysxCollisionRepresentationResult
import trimesh
root=Path(__file__).resolve().parents[1];source=root/'assets/robots/holon/holon.urdf';out=root/'build/collision_physx';out.mkdir(exist_ok=True)
context=omni.usd.get_context();context.new_stage();stage=context.get_stage();UsdGeom.SetStageMetersPerUnit(stage,1.);UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z)
records={};cook=get_physx_cooking_interface();stage_id=UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
for link in ET.parse(source).findall('link'):
    mesh=link.find('collision/geometry/mesh')
    if mesh is None:continue
    path=source.parent/mesh.get('filename');digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if digest in records:continue
    directory=out/digest;directory.mkdir(exist_ok=True);recordfile=directory/'manifest.json'
    if recordfile.exists():records[digest]=json.loads(recordfile.read_text());continue
    name=link.get('name');original=trimesh.load_mesh(path);usd=UsdGeom.Mesh.Define(stage,'/World/'+name)
    usd.CreatePointsAttr([Gf.Vec3f(*v) for v in original.vertices]);usd.CreateFaceVertexCountsAttr([3]*len(original.faces));usd.CreateFaceVertexIndicesAttr(original.faces.ravel().tolist());usd.CreateSubdivisionSchemeAttr('none')
    prim=usd.GetPrim();UsdPhysics.CollisionAPI.Apply(prim);UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr('convexDecomposition' if 'dock_mech' in name else 'convexHull')
    if 'dock_mech' in name:
        prim.AddAppliedSchema('PhysxConvexDecompositionCollisionAPI');prim.CreateAttribute('physxConvexDecompositionCollision:maxConvexHulls',Sdf.ValueTypeNames.Int).Set(128);prim.CreateAttribute('physxConvexDecompositionCollision:shrinkWrap',Sdf.ValueTypeNames.Bool).Set(True)
    result_holder={}
    def on_result(result,convexes):
        result_holder['result']=str(result);result_holder['pieces']=[]
        if result!=PhysxCollisionRepresentationResult.RESULT_VALID:return
        for convex in convexes:
            vertices=np.array([[v.x,v.y,v.z] for v in convex.vertices]);faces=[]
            for polygon in convex.polygons:
                indices=convex.indices[polygon.index_base:polygon.index_base+polygon.num_vertices]
                faces.extend([[indices[0],indices[j],indices[j+1]] for j in range(1,len(indices)-1)])
            result_holder['pieces'].append((vertices,np.asarray(faces)))
    started=time.perf_counter();cook.request_convex_collision_representation(stage_id=stage_id,collision_prim_id=PhysicsSchemaTools.sdfPathToInt(str(prim.GetPath())),run_asynchronously=False,on_result=on_result)
    if not result_holder.get('pieces'):raise RuntimeError(name+': '+str(result_holder))
    parts=[]
    for i,(vertices,faces) in enumerate(result_holder['pieces']):
        file=directory/f'{i}.obj';file.write_text(trimesh.Trimesh(vertices,faces,process=False).export(file_type='obj'));parts.append(dict(file=file.name,sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
    record=dict(source=str(path.relative_to(root)),source_sha256=digest,engine='installed Isaac Sim 6.0.1 PhysX 110.1.13',max_convex_hulls=128 if 'dock_mech' in name else 1,shrink_wrap=True,parts=parts,wall_s=time.perf_counter()-started)
    recordfile.write_text(json.dumps(record,indent=2));records[digest]=record;print('EXPORTED',name,len(parts),record['wall_s'],flush=True)
(out/'manifest.json').write_text(json.dumps(records,indent=2));stage.GetRootLayer().Export(str(out/'cooking_scene.usda'))
# Dock shapes follow the source PhysX recipe. Keep the existing component
# representation elsewhere: a single hull would fill gaps between robot parts.
combined=root/'build/collision_physx_docks';combined.mkdir(exist_ok=True);combined_records={}
for link in ET.parse(source).findall('link'):
    mesh=link.find('collision/geometry/mesh')
    if mesh is None:continue
    digest=hashlib.sha256((source.parent/mesh.get('filename')).read_bytes()).hexdigest()
    if digest in combined_records:continue
    directory=(out if 'dock_mech' in link.get('name') else root/'build/collision_components')/digest
    manifest=json.loads((directory/'manifest.json').read_text())
    assert manifest['source_sha256']==digest
    for part in manifest['parts']:
        assert hashlib.sha256((directory/part['file']).read_bytes()).hexdigest()==part['sha256']
    shutil.copytree(directory,combined/digest,dirs_exist_ok=True);combined_records[digest]=manifest
(combined/'manifest.json').write_text(json.dumps(combined_records,indent=2))
(combined/'provenance.json').write_text(json.dumps(dict(
    dock_representation='source PhysX 128-hull shrinkWrap collision recipe',
    other_parts='existing CAD-preserving component hulls; single hull creates phantom self-overlap',
    finished_structure_cache=False),indent=2))
app.close()
