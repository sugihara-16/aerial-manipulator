"""Experimental single exterior mesh, avoiding convex partition interior faces.

Not the production geometry until contact/shape accuracy is qualified.
"""
import json
from pathlib import Path
import numpy as np
import trimesh
from pxr import UsdGeom, UsdPhysics, PhysxSchema, PhysicsSchemaTools, Gf


def replace_ellipsoid(stage, geometry, plant, resolution=300):
    gid=plant.mj.geom('object').id
    root=stage.GetPrimAtPath(geometry['paths'][gid])
    if int(plant.mj.geom_type[gid])!=4:
        raise ValueError('expected an ellipsoid')
    for child in list(root.GetChildren()):
        geometry['lookup'].pop(int(PhysicsSchemaTools.sdfPathToInt(str(child.GetPath()))))
        stage.RemovePrim(child.GetPath())
    manifest=json.loads(Path('build/ellipsoid_unit_parts.json').read_text())
    mesh=trimesh.creation.icosphere(subdivisions=manifest['subdivisions'])
    mesh.vertices*=plant.mj.geom_size[gid]
    surface=UsdGeom.Mesh.Define(stage,str(root.GetPath())+'/surface')
    surface.CreatePointsAttr([Gf.Vec3f(*map(float,v)) for v in mesh.vertices])
    surface.CreateFaceVertexCountsAttr([3]*len(mesh.faces))
    surface.CreateFaceVertexIndicesAttr(mesh.faces.ravel().tolist())
    surface.CreateSubdivisionSchemeAttr('none')
    prim=surface.GetPrim();UsdPhysics.CollisionAPI.Apply(prim)
    UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(PhysxSchema.Tokens.sdf)
    api=PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(prim)
    api.CreateSdfResolutionAttr(resolution)
    api.CreateSdfEnableRemeshingAttr(False)
    api.CreateSdfTriangleCountReductionFactorAttr(1.)
    contact=PhysxSchema.PhysxCollisionAPI.Apply(prim)
    contact.CreateRestOffsetAttr(0.);contact.CreateContactOffsetAttr(.001)
    geometry['lookup'][int(PhysicsSchemaTools.sdfPathToInt(str(prim.GetPath())))]=gid
    return dict(resolution=resolution,vertices=len(mesh.vertices),faces=len(mesh.faces),
                mesh_approximation_bound_m=float(plant.mj.geom_size[gid].max()*manifest['radial_error_bound_unit']),
                sdf_interpolation_error='not yet qualified',remeshing=False)
