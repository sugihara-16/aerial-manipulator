"""Ground-walking task and path references; no learned or stored body designs."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class WalkSpec:
    modules: int = 2
    speed: float = .22
    distance: float = 5.
    curvature: float = 0.

    def __post_init__(self):
        if not isinstance(self.modules,int) or not 1<=self.modules<=8:
            raise ValueError('walking needs 1..8 fixed modules')
        if not np.isfinite([self.speed,self.distance,self.curvature]).all() or self.speed<.2 or self.distance<=0:
            raise ValueError('walking requires at least 0.2 m/s and positive finite distance')

    @property
    def duration(self):
        """Nominal travel time, not a substitute for measured arrival."""
        return self.distance/self.speed

    @property
    def time_limit(self):
        return self.distance/.2

    def path(self,time):
        """Constant-speed straight or broad circular path, starting along +x."""
        distance=self.speed*np.asarray(time);heading=self.curvature*distance
        if abs(self.curvature)<1e-10:
            xy=np.stack([distance,np.zeros_like(distance)],axis=-1)
        else:
            xy=np.stack([np.sin(heading)/self.curvature,(1-np.cos(heading))/self.curvature],axis=-1)
        velocity=self.speed*np.stack([np.cos(heading),np.sin(heading)],axis=-1)
        acceleration=self.speed**2*self.curvature*np.stack([-np.sin(heading),np.cos(heading)],axis=-1)
        return xy,heading,velocity,acceleration


def foot_link(module,port):
    return f'm{module}/'+('pitch_dock_mech' if port.startswith('pitch') else 'yaw_dock_mech')+port[-1]


def requires_face_alignment(port_or_link):
    """Female pitch Docks face the floor; male yaw Docks may tilt."""
    name=port_or_link.rsplit('/',1)[-1]
    if name in ('pitch1','pitch2','pitch_dock_mech1','pitch_dock_mech2'):return True
    if name in ('yaw1','yaw2','yaw_dock_mech1','yaw_dock_mech2'):return False
    raise ValueError(f'unknown walking Dock: {port_or_link}')


def rotation_z(angle):
    c,s=np.cos(angle),np.sin(angle)
    return np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])


def ground_support(rotation,world_vertices):
    """Finite support corners and friction axes that rotate with the foot.

    The inner friction polygon is a numerical approximation of isotropic
    friction. Its orientation and corner selection must not change solely
    because the same periodic assembly turns in the horizontal plane.
    """
    tangent=rotation[:,np.argmin(abs(rotation[2]))].copy();tangent[2]=0.
    tangent/=np.linalg.norm(tangent);normal=np.array([0.,0.,1.])
    basis=np.array([normal,tangent,np.cross(normal,tangent)]).T
    near=np.flatnonzero(world_vertices[:,2]<=world_vertices[:,2].min()+.0005)
    directions=np.array([[1,1],[1,-1],[-1,1],[-1,-1]])
    indices=near[np.argmax((world_vertices[near]@basis[:,1:])@directions.T,axis=0)]
    return basis,indices


def foot_material(plant,links):
    """Actual collision vertices in each Dock frame; no enlarged sole."""
    import mujoco
    if getattr(plant,'_foot_material_model',None) is not plant.mj:
        plant._foot_material_model=plant.mj;plant._foot_material_cache={}
    key=tuple(links)
    if key in plant._foot_material_cache:
        return [(bid,vertices.copy()) for bid,vertices in plant._foot_material_cache[key]]
    m=plant.mj;d=mujoco.MjData(m);mujoco.mj_kinematics(m,d)
    result=[]
    for link in links:
        bid=m.body(link).id;points=[]
        for g in range(m.ngeom):
            if not (m.geom(g).name or '').startswith(link+'/'):continue
            mid=m.geom_dataid[g]
            if m.geom_type[g]!=int(mujoco.mjtGeom.mjGEOM_MESH):raise ValueError('feet require actual mesh')
            vertices=m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid]+m.mesh_vertnum[mid]]
            world=vertices@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g]
            points.append((world-d.xpos[bid])@d.xmat[bid].reshape(3,3))
        if not points:raise ValueError('foot has no collision material')
        result.append((bid,np.unique(np.round(np.vstack(points),9),axis=0)))
    plant._foot_material_cache[key]=[(bid,vertices.copy()) for bid,vertices in result]
    return result
