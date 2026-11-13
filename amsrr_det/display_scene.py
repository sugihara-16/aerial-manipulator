"""Display-only decoration shared by desktop and browser renderers."""
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path
import numpy as np


@lru_cache(maxsize=64)
def _joined_mesh(sources):
    """Keep every source triangle, but draw a part in one OpenGL call."""
    vertices=[];faces=[]
    for filename,scale in sources:
        local=[];triangles=[]
        for line in Path(filename).read_text().splitlines():
            fields=line.split()
            if not fields:continue
            if fields[0]=='v':local.append([float(x) for x in fields[1:4]])
            elif fields[0]=='f':
                ids=[int(x.split('/')[0]) for x in fields[1:]]
                ids=[i-1 if i>0 else len(local)+i for i in ids]
                triangles.extend((ids[0],ids[i],ids[i+1]) for i in range(1,len(ids)-1))
        offset=len(vertices)
        vertices.extend(np.asarray(local)*np.asarray(scale))
        faces.extend(tuple(offset+i for i in f) for f in triangles)
    return (' '.join(format(float(x),'.9g') for v in vertices for x in v),
            ' '.join(str(i) for f in faces for i in f))


def _batch_robot_meshes(root):
    # The simulator keeps its separate collision hulls. Only the read-only
    # renderer combines identically placed/coloured hulls into triangle meshes.
    asset=root.find('asset')
    if asset is None:return
    meshes={e.get('name'):e for e in asset.findall('mesh')};joined={}
    for body in root.findall('.//body'):
        groups={}
        for geom in body.findall('geom'):
            source=meshes.get(geom.get('mesh'))
            if source is None or not (geom.get('name') or '').startswith('m'):continue
            if not source.get('file','').endswith('.obj'):continue
            if any(k not in ('name','file','scale') for k in source.attrib):continue
            key=tuple(sorted((k,v) for k,v in geom.attrib.items() if k not in ('name','mesh')))
            groups.setdefault(key,[]).append(geom)
        for group in groups.values():
            if len(group)<2:continue
            sources=tuple((meshes[g.get('mesh')].get('file'),tuple(map(float,meshes[g.get('mesh')].get('scale','1 1 1').split()))) for g in group)
            if sources not in joined:
                name=f'console_part_{len(joined)}';vertex,face=_joined_mesh(sources)
                ET.SubElement(asset,'mesh',name=name,vertex=vertex,face=face)
                joined[sources]=name
            group[0].set('mesh',joined[sources])
            for geom in group[1:]:body.remove(geom)
    if joined:
        used={g.get('mesh') for g in root.findall('.//geom')}
        for mesh in list(asset.findall('mesh')):
            if mesh.get('name') not in used:asset.remove(mesh)
        # Contact pairs are irrelevant in a model used only for kinematics.
        contact=root.find('contact')
        if contact is not None:
            for pair in list(contact.findall('pair')):contact.remove(pair)


def display_xml(xml):
    """Decorate only the renderer's model; simulation geometry is untouched."""
    root=ET.fromstring(xml)
    _batch_robot_meshes(root)
    asset=root.find('asset')
    if asset is None:asset=ET.SubElement(root,'asset')
    ET.SubElement(asset,'texture',name='console_floor_texture',type='2d',builtin='checker',rgb1='.22 .28 .33',rgb2='.55 .62 .67',width='512',height='512')
    ET.SubElement(asset,'material',name='console_floor_material',texture='console_floor_texture',texrepeat='2 2',texuniform='true',reflectance='0',shininess='.15')
    for geom in root.iter('geom'):
        if geom.get('type')=='plane':
            geom.set('material','console_floor_material');geom.set('rgba','1 1 1 1')
    world=root.find('worldbody')
    ET.SubElement(world,'light',name='console_sun',directional='true',pos='0 0 8',dir='0 0 -1',diffuse='.8 .8 .8',ambient='.12 .12 .12',specular='.2 .2 .2',castshadow='true')
    return ET.tostring(root,encoding='unicode')

