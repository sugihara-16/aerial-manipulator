"""Read immutable cooked object geometry before execution, not contact truth."""
import numpy as np
import trimesh
from pxr import UsdGeom,UsdPhysics,PhysicsSchemaTools
from omni.physx import get_physx_cooking_interface
from omni.physx.bindings._physx import PhysxCollisionRepresentationResult
from amsrr_det.mesh_surface import MeshSurface


def cooked_object_surface(sim,geometry,plant,nominal):
    if nominal.family!='ellipsoid':return nominal,None
    gid=plant.mj.geom('object').id
    root=sim.stage.GetPrimAtPath(geometry['paths'][gid]);body=sim.stage.GetPrimAtPath(sim.paths[plant.meta['object']['joint']])
    cache=UsdGeom.XformCache();pieces=[]
    for prim in root.GetChildren():
        if not prim.HasAPI(UsdPhysics.CollisionAPI):continue
        result={}
        def callback(status,convexes):
            if status!=PhysxCollisionRepresentationResult.RESULT_VALID:raise ValueError('cooked object unavailable')
            result['convexes']=convexes
        get_physx_cooking_interface().request_convex_collision_representation(stage_id=sim.stage_id,collision_prim_id=PhysicsSchemaTools.sdfPathToInt(str(prim.GetPath())),run_asynchronously=False,on_result=callback)
        matrix=np.asarray(cache.ComputeRelativeTransform(prim,body)[0]).T
        for convex in result['convexes']:
            vertices=np.array([[v.x,v.y,v.z] for v in convex.vertices]);faces=[]
            vertices=vertices@matrix[:3,:3].T+matrix[:3,3]
            for polygon in convex.polygons:
                indices=convex.indices[polygon.index_base:polygon.index_base+polygon.num_vertices]
                faces.extend([[indices[0],indices[j],indices[j+1]] for j in range(1,len(indices)-1)])
            pieces.append(trimesh.Trimesh(vertices,np.asarray(faces),process=True))
    # Boolean union removes covered internal partition faces. It is geometry
    # preprocessing shared by planner/controller, independent of trajectory.
    mesh=trimesh.boolean.union(pieces,engine='manifold',check_volume=True)
    return MeshSurface(nominal,mesh,pieces),dict(parts=len(pieces),vertices=len(mesh.vertices),faces=len(mesh.faces),volume_m3=float(mesh.volume),watertight=bool(mesh.is_watertight))
