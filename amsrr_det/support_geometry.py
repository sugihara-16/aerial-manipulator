"""Exact rectangular union for a diagnostic of overlapping fixed supports."""
import xml.etree.ElementTree as ET
import numpy as np


def support_layers(half_size, interface_depth=.004):
    """Partition a box exactly, placing one internal face below its top.

    This is a collision-representation experiment. The default internal face
    is below the 2 mm penetration limit plus the pair's 2 mm contact distance.
    Shape, occupied volume and material are unchanged.
    """
    half=np.asarray(half_size,dtype=float)
    if half.shape!=(3,) or np.any(half<=0) or not 0<interface_depth<2*half[2]:
        raise ValueError('invalid support partition')
    top=half.copy();top[2]=interface_depth/2
    bottom=half.copy();bottom[2]-=interface_depth/2
    return [(np.array([0.,0.,half[2]-interface_depth/2]),top),
            (np.array([0.,0.,-interface_depth/2]),bottom)]


def merge_rectangular_supports(xml):
    doc=ET.fromstring(xml);world=doc.find('worldbody')
    boxes=[world.find(f"geom[@name='{name}']") for name in ['support','destination_support']]
    if any(box is None or box.get('type')!='box' for box in boxes):raise ValueError('two fixed support boxes required')
    attributes=[{k:v for k,v in box.attrib.items() if k not in ['name','pos','size']} for box in boxes]
    if attributes[0]!=attributes[1] or any(any(key in box.attrib for key in ['quat','euler','axisangle','xyaxes','zaxis']) for box in boxes):
        raise ValueError('matching axis-aligned support materials/attributes required')
    center=np.array([np.fromstring(box.get('pos'),sep=' ') for box in boxes]);half=np.array([np.fromstring(box.get('size'),sep=' ') for box in boxes])
    lower,upper=center-half,center+half
    lo,hi=lower.min(axis=0),upper.max(axis=0)
    intersection=float(np.prod(np.maximum(0.,upper.min(axis=0)-lower.max(axis=0))))
    union=float(np.prod(2*half,axis=1).sum()-intersection);outer=float(np.prod(hi-lo))
    if abs(outer-union)>1e-12*outer:raise ValueError('support union is not a box; refusing to fill empty space')
    boxes[0].set('pos',' '.join(map(str,(lo+hi)/2)));boxes[0].set('size',' '.join(map(str,(hi-lo)/2)));world.remove(boxes[1])
    return ET.tostring(doc,encoding='unicode'),dict(scope='same_occupied_volume_rectangular_union',original_bounds=np.stack([lower,upper],axis=1).tolist(),union_bounds=[lo.tolist(),hi.tolist()],union_volume_m3=union,bounding_volume_m3=outer)


def box_union_surface(centers,half_sizes):
    """Exact boundary cells of axis-aligned boxes; no convexification.

    Grid planes are the original face coordinates, so this is a Boolean
    arrangement, not a sampled voxel approximation. Internal cell faces are
    omitted. Edge-only self-touching unions are rejected by the mesh check.
    """
    import trimesh
    from itertools import product
    centers=np.asarray(centers,dtype=float);half=np.asarray(half_sizes,dtype=float)
    if centers.ndim!=2 or centers.shape[1]!=3 or centers.shape!=half.shape or not len(centers) or np.any(half<=0):
        raise ValueError('positive axis-aligned boxes required')
    if not np.isfinite(centers).all() or not np.isfinite(half).all():raise ValueError('finite boxes required')
    lower,upper=centers-half,centers+half
    planes=[np.unique(np.r_[lower[:,i],upper[:,i]]) for i in range(3)]
    grid=tuple(len(p)-1 for p in planes);occupied=set()
    for cell in product(*(range(n) for n in grid)):
        middle=np.array([(planes[i][cell[i]]+planes[i][cell[i]+1])/2 for i in range(3)])
        if np.any(np.all((middle>lower)&(middle<upper),axis=1)):occupied.add(cell)
    vertices=[];faces=[];index={};volume=0.
    for cell in sorted(occupied):
        volume+=np.prod([planes[i][cell[i]+1]-planes[i][cell[i]] for i in range(3)])
        for axis in range(3):
            others=[(axis+1)%3,(axis+2)%3]
            for side in [-1,1]:
                neighbour=list(cell);neighbour[axis]+=side
                if tuple(neighbour) in occupied:continue
                corners=[]
                for u,w in [(0,0),(1,0),(1,1),(0,1)]:
                    point=list(cell);point[axis]+=int(side>0);point[others[0]]+=u;point[others[1]]+=w
                    key=tuple(point)
                    if key not in index:
                        index[key]=len(vertices);vertices.append([planes[i][point[i]] for i in range(3)])
                    corners.append(index[key])
                if side<0:corners.reverse()
                faces.extend([[corners[0],corners[1],corners[2]],[corners[0],corners[2],corners[3]]])
    mesh=trimesh.Trimesh(vertices,faces,process=False)
    if not mesh.is_watertight or not mesh.is_winding_consistent or abs(mesh.volume-volume)>1e-10*volume:
        raise ValueError('box union is not a closed oriented volume')
    return mesh,dict(centers=centers.tolist(),half_sizes=half.tolist(),volume_m3=float(volume),
        vertices=len(vertices),faces=len(faces),occupied_cells=len(occupied),
        scope='exact_axis_aligned_box_union_boundary')
