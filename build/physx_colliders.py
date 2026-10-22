"""Owned compiled geometry -> PhysX colliders, preserving physical exclusions.

MuJoCo is used only as an asset/coordinate container here, not a dynamics oracle.
Each already-convex source part stays a separate shape with its own contact law.
"""
import numpy as np
from pathlib import Path
import json,hashlib
import xml.etree.ElementTree as ET
import mujoco
import trimesh
import pinocchio as pin
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, PhysxSchema, Gf, PhysicsSchemaTools
from .physx_tree import quat
from amsrr_det.convex_parts import split_convex
from amsrr_det.model import ROOT,transform


def add_colliders(plant, stage, paths, selected_links, support_representation='box'):
    if support_representation not in ['box','triangle_mesh','surface_layer','surface_mesh_layer','union_mesh']:raise ValueError('unknown support representation')
    layered=support_representation in ['surface_layer','surface_mesh_layer']
    m=plant.mj;pd=plant.pin.createData();q=pin.neutral(plant.pin)
    pin.framesForwardKinematics(plant.pin,pd,q)
    d=mujoco.MjData(m);d.qpos[:]=plant.configuration(q);mujoco.mj_kinematics(m,d)
    union_mesh=None;union_info=None;union_ids=[]
    if support_representation=='union_mesh':
        from amsrr_det.support_geometry import box_union_surface
        union_ids=[m.geom(name).id for name in ['support','destination_support']]
        if any(m.geom_bodyid[g]!=0 or m.geom_type[g]!=mujoco.mjtGeom.mjGEOM_BOX or
               not np.allclose(d.geom_xmat[g].reshape(3,3),np.eye(3),atol=1e-12) for g in union_ids):
            raise ValueError('matching fixed axis-aligned support boxes required')
        union_mesh,union_info=box_union_surface(d.geom_xpos[union_ids],m.geom_size[union_ids])
    materials={}
    for name,friction,coated in [('ordinary',.6,False),('floor',.8,False),('coated',4.5,True)]:
        material=UsdShade.Material.Define(stage,'/World/Materials/'+name);api=UsdPhysics.MaterialAPI.Apply(material.GetPrim());api.CreateStaticFrictionAttr().Set(friction);api.CreateDynamicFrictionAttr().Set(friction);api.CreateRestitutionAttr().Set(0.)
        px=PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim());px.CreateFrictionCombineModeAttr().Set('max')
        if coated:px.CreateCompliantContactStiffnessAttr().Set(7500.);px.CreateCompliantContactDampingAttr().Set(75.)
        materials[name]=material
    groups={};geom_paths={};lookup={};link_ids={};parts_cache={};approximation_bound=0.;authored_docks={};support_layer_ids={}
    source=ROOT/'assets/robots/holon/holon.urdf';robot=ET.parse(source)
    for gid in range(m.ngeom):
        name=m.geom(gid).name;bid=int(m.geom_bodyid[gid]);body=m.body(bid).name
        if union_ids and gid==union_ids[1]:
            geom_paths[gid]=geom_paths[union_ids[0]];link_ids[gid]=0
            continue
        if body in authored_docks:
            geom_paths[gid],link_ids[gid]=authored_docks[body]
            continue
        if bid not in groups:
            group=UsdPhysics.CollisionGroup.Define(stage,f'/World/Groups/body_{bid}');groups[bid]=group
        ancestor=bid
        while ancestor and not m.body(ancestor).name.startswith('joint'):ancestor=int(m.body_parentid[ancestor])
        jid=int(m.body(ancestor).name[5:]) if ancestor else 0
        path=(paths[jid] if jid else '/World/Environment')+f'/geom_{gid}'
        T=pin.SE3(d.geom_xmat[gid].reshape(3,3),d.geom_xpos[gid]);T=pd.oMi[jid].inverse()*T if jid else T
        kind=int(m.geom_type[gid]);size=m.geom_size[gid];colliders=None
        if 'dock_mech' in body:
            # Author the source CAD once. Re-cooking each exported convex hull
            # changes its surface and therefore no longer reproduces the cache.
            collision=robot.find(f"link[@name='{body.split('/')[-1]}']/collision")
            mesh=collision.find('geometry/mesh');shape=trimesh.load_mesh(source.parent/mesh.get('filename'))
            vertices=np.asarray(shape.vertices)*np.fromstring(mesh.get('scale','1 1 1'),sep=' ');faces=np.asarray(shape.faces)
            geom=UsdGeom.Mesh.Define(stage,path);geom.CreatePointsAttr([Gf.Vec3f(*map(float,v)) for v in vertices]);geom.CreateFaceVertexCountsAttr([3]*len(faces));geom.CreateFaceVertexIndicesAttr(faces.ravel().tolist());geom.CreateSubdivisionSchemeAttr('none')
            UsdPhysics.MeshCollisionAPI.Apply(geom.GetPrim()).CreateApproximationAttr('convexDecomposition')
            api=PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(geom.GetPrim());api.CreateMaxConvexHullsAttr().Set(128);api.CreateShrinkWrapAttr().Set(True)
            T=pd.oMi[jid].inverse()*pd.oMf[plant.meta['frames'][body]]*transform(collision.find('origin'))
            authored_docks[body]=(path,jid)
        elif kind==mujoco.mjtGeom.mjGEOM_MESH:
            mesh=int(m.geom_dataid[gid]);vertices=m.mesh_vert[m.mesh_vertadr[mesh]:m.mesh_vertadr[mesh]+m.mesh_vertnum[mesh]];faces=m.mesh_face[m.mesh_faceadr[mesh]:m.mesh_faceadr[mesh]+m.mesh_facenum[mesh]]
            if mesh not in parts_cache:parts_cache[mesh]=split_convex(vertices)
            geom=UsdGeom.Xform.Define(stage,path);colliders=[]
            for index,(vv,ff) in enumerate(parts_cache[mesh]):
                part=UsdGeom.Mesh.Define(stage,path+f'/part_{index}');part.CreatePointsAttr([Gf.Vec3f(*map(float,v)) for v in vv*1000]);part.AddScaleOp().Set(Gf.Vec3f(.001));part.CreateFaceVertexCountsAttr([3]*len(ff));part.CreateFaceVertexIndicesAttr(ff.ravel().tolist());part.CreateSubdivisionSchemeAttr('none')
                UsdPhysics.MeshCollisionAPI.Apply(part.GetPrim()).CreateApproximationAttr('convexHull')
                PhysxSchema.PhysxConvexHullCollisionAPI.Apply(part.GetPrim()).CreateHullVertexLimitAttr().Set(255)
                colliders.append(part.GetPrim())
        elif kind==mujoco.mjtGeom.mjGEOM_ELLIPSOID:
            # Shared unit geometry only: the cache contains no robot structures.
            cache_path=Path(__file__).with_name('ellipsoid_unit_parts.npz')
            manifest=json.loads(cache_path.with_suffix('.json').read_text())
            if hashlib.sha256(cache_path.read_bytes()).hexdigest()!=manifest['sha256']:raise ValueError('ellipsoid geometry cache hash mismatch')
            cache=np.load(cache_path);geom=UsdGeom.Xform.Define(stage,path);colliders=[]
            for index in range(manifest['parts']):
                vv,ff=cache['v'+str(index)]*size,cache['f'+str(index)]
                part=UsdGeom.Mesh.Define(stage,path+f'/part_{index}');part.CreatePointsAttr([Gf.Vec3f(*map(float,v)) for v in vv*1000]);part.AddScaleOp().Set(Gf.Vec3f(.001));part.CreateFaceVertexCountsAttr([3]*len(ff));part.CreateFaceVertexIndicesAttr(ff.ravel().tolist());part.CreateSubdivisionSchemeAttr('none')
                UsdPhysics.MeshCollisionAPI.Apply(part.GetPrim()).CreateApproximationAttr('convexHull');PhysxSchema.PhysxConvexHullCollisionAPI.Apply(part.GetPrim()).CreateHullVertexLimitAttr().Set(255);colliders.append(part.GetPrim())
            approximation_bound=float(size.max()*manifest['radial_error_bound_unit'])
        elif kind==mujoco.mjtGeom.mjGEOM_BOX:
            if union_ids and gid==union_ids[0]:
                T=pin.SE3.Identity()
                geom=UsdGeom.Mesh.Define(stage,path)
                geom.CreatePointsAttr([Gf.Vec3f(*map(float,v)) for v in union_mesh.vertices])
                geom.CreateFaceVertexCountsAttr([3]*len(union_mesh.faces));geom.CreateFaceVertexIndicesAttr(union_mesh.faces.ravel().tolist());geom.CreateSubdivisionSchemeAttr('none')
                UsdPhysics.MeshCollisionAPI.Apply(geom.GetPrim()).CreateApproximationAttr('none')
                union_info.update(path=path,face_centers=union_mesh.triangles_center.tolist(),face_normals=union_mesh.face_normals.tolist())
            elif jid==0 and layered:
                from amsrr_det.support_geometry import support_layers
                geom=UsdGeom.Xform.Define(stage,path);colliders=[]
                for index,(center,half) in enumerate(support_layers(size)):
                    if index==0 and support_representation=='surface_mesh_layer':
                        # Same closed top slab, including sides and bottom.
                        # Only the collision representation changes, not its
                        # dimensions, material, offsets, or lower box.
                        box=trimesh.creation.box(extents=np.ones(3))
                        part=UsdGeom.Mesh.Define(stage,path+f'/layer_{index}')
                        part.CreatePointsAttr([Gf.Vec3f(*map(float,v)) for v in box.vertices])
                        part.CreateFaceVertexCountsAttr([3]*len(box.faces));part.CreateFaceVertexIndicesAttr(box.faces.ravel().tolist());part.CreateSubdivisionSchemeAttr('none')
                        UsdPhysics.MeshCollisionAPI.Apply(part.GetPrim()).CreateApproximationAttr('none')
                    else:
                        part=UsdGeom.Cube.Define(stage,path+f'/layer_{index}');part.CreateSizeAttr(1.)
                    part.AddTranslateOp().Set(Gf.Vec3d(*map(float,center)))
                    part.AddScaleOp().Set(Gf.Vec3f(*map(float,2*half)));colliders.append(part.GetPrim())
            elif jid==0 and support_representation=='triangle_mesh':
                # The same closed box surface and transform, with no union,
                # inflation or change of material. Only static support boxes
                # change their collision representation; Dock parts do not.
                box=trimesh.creation.box(extents=np.ones(3))
                geom=UsdGeom.Mesh.Define(stage,path);geom.CreatePointsAttr([Gf.Vec3f(*map(float,v)) for v in box.vertices]);geom.CreateFaceVertexCountsAttr([3]*len(box.faces));geom.CreateFaceVertexIndicesAttr(box.faces.ravel().tolist());geom.CreateSubdivisionSchemeAttr('none')
                UsdPhysics.MeshCollisionAPI.Apply(geom.GetPrim()).CreateApproximationAttr('none')
            else:geom=UsdGeom.Cube.Define(stage,path);geom.CreateSizeAttr(1.)
        elif kind==mujoco.mjtGeom.mjGEOM_CYLINDER:
            from carb.settings import get_settings
            from omni.physx.bindings._physx import SETTING_COLLISION_APPROXIMATE_CYLINDERS
            get_settings().set_bool(SETTING_COLLISION_APPROXIMATE_CYLINDERS,False)
            geom=UsdGeom.Cylinder.Define(stage,path);geom.CreateRadiusAttr(float(size[0]));geom.CreateHeightAttr(float(2*size[1]));geom.CreateAxisAttr('Z')
        elif kind==mujoco.mjtGeom.mjGEOM_PLANE:
            geom=UsdGeom.Plane.Define(stage,path);geom.CreateAxisAttr('Z');geom.CreateWidthAttr(200.);geom.CreateLengthAttr(200.)
        else:raise ValueError(f'Unqualified PhysX object representation: {name} type {kind}')
        geom.AddTranslateOp().Set(Gf.Vec3d(*map(float,T.translation)));geom.AddOrientOp().Set(quat(T.rotation))
        if kind==mujoco.mjtGeom.mjGEOM_BOX and not (jid==0 and layered) and not (union_ids and gid==union_ids[0]):geom.AddScaleOp().Set(Gf.Vec3f(*map(float,2*size)))
        prim=geom.GetPrim()
        if colliders is None:colliders=[prim]
        for collider in colliders:
            UsdPhysics.CollisionAPI.Apply(collider)
            api=PhysxSchema.PhysxCollisionAPI.Apply(collider);api.CreateRestOffsetAttr().Set(0.);api.CreateContactOffsetAttr().Set(.001)
            lookup[int(PhysicsSchemaTools.sdfPathToInt(str(collider.GetPath())))]=gid
            if jid==0 and layered and kind==mujoco.mjtGeom.mjGEOM_BOX:
                support_layer_ids[int(PhysicsSchemaTools.sdfPathToInt(str(collider.GetPath())))]=dict(geom=gid,layer=collider.GetName(),rotation=T.rotation.copy(),center=T.translation.copy(),half_size=np.array(size).copy())
        material='coated' if any(name.startswith(link+'/') for link in selected_links) else ('floor' if name=='floor' else 'ordinary')
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(materials[material],UsdShade.Tokens.weakerThanDescendants,'physics')
        groups[bid].GetCollidersCollectionAPI().CreateIncludesRel().AddTarget(path)
        geom_paths[gid]=path;link_ids[gid]=jid
    for sig in m.exclude_signature:
        first,second=int(sig)>>16,int(sig)&65535
        if first in groups and second in groups:
            groups[first].CreateFilteredGroupsRel().AddTarget(groups[second].GetPath())
            groups[second].CreateFilteredGroupsRel().AddTarget(groups[first].GetPath())
    for path in paths.values():PhysxSchema.PhysxContactReportAPI.Apply(stage.GetPrimAtPath(path)).CreateThresholdAttr().Set(0.)
    return dict(support_union=union_info,paths=geom_paths,lookup=lookup,link_ids=link_ids,support_layer_ids=support_layer_ids,ellipsoid_polyhedral_error_bound_m=approximation_bound)
